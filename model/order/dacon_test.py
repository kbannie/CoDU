import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count

# ------------------------- 기본 유틸리티 함수 -------------------------
def parse_bbox(bbox_str):
    """bbox 파싱"""
    if isinstance(bbox_str, str):
        return list(map(float, bbox_str.replace(' ', '').split(',')))
    elif isinstance(bbox_str, list):
        return bbox_str
    else:
        raise ValueError(f"bbox 형식 오류: {bbox_str}")

def iou(boxA, boxB):
    """IoU 계산"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: 
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def normalized_edit_distance(s1, s2):
    """정규화된 편집 거리 (NED)"""
    s1, s2 = str(s1).strip(), str(s2).strip()
    max_len = max(len(s1), len(s2))
    return 0.0 if max_len == 0 else 1 - SequenceMatcher(None, s1, s2).ratio()

def ned_reading_order(gt_list, pred_list):
    """Reading Order NED 계산"""
    if not gt_list and not pred_list:
        return 0.0
    if not gt_list or not pred_list:
        return 1.0
    if len(gt_list) < 2 or len(pred_list) < 2:
        return 1.0 if gt_list != pred_list else 0.0
    gt_str = ','.join(map(str, gt_list))
    pred_str = ','.join(map(str, pred_list))
    return 1 - SequenceMatcher(None, gt_str, pred_str).ratio()

# ------------------------- COCO-style mAP@0.5:0.95 계산 -------------------------
def compute_custom_map(answer_df, pred_df):
    """
    COCO-style mAP 계산 - 완전 개선된 버전
    예측력이 나쁘거나 데이터가 없어도 안정적으로 작동
    """
    IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
    CATEGORIES = sorted(answer_df['category_type'].unique())
    ap_all = []

    for category in CATEGORIES:
        gt_cat = answer_df[answer_df['category_type'] == category].copy()
        pred_cat = pred_df[pred_df['category_type'] == category].copy()

        # GT가 없으면 해당 카테고리의 모든 IoU threshold에서 AP = 0
        if len(gt_cat) == 0:
            for _ in IOU_THRESHOLDS:
                ap_all.append(0.0)
            continue

        gt_by_image = gt_cat.groupby('ID')
        pred_by_image = pred_cat.groupby('ID')

        for iou_thresh in IOU_THRESHOLDS:
            tps, fps = [], []
            total_gts = 0

            for doc_id, gt_group in gt_by_image:
                try:
                    gt_boxes = gt_group['bbox'].apply(parse_bbox).tolist()
                    matched = [False] * len(gt_boxes)
                    total_gts += len(gt_boxes)

                    # 예측 데이터 처리
                    if doc_id in pred_by_image.groups:
                        pred_group = pred_by_image.get_group(doc_id)
                        pred_boxes = pred_group[['bbox', 'confidence_score']].copy()
                        pred_boxes['bbox'] = pred_boxes['bbox'].apply(parse_bbox)
                        # Confidence score 기준 내림차순 정렬 (COCO 표준)
                        pred_boxes = pred_boxes.sort_values('confidence_score', ascending=False).reset_index(drop=True)
                    else:
                        pred_boxes = pd.DataFrame(columns=['bbox', 'confidence_score'])

                    # 각 예측에 대해 TP/FP 판정
                    for _, row in pred_boxes.iterrows():
                        pred_box = row['bbox']
                        matched_flag = False
                        for i, gt_box in enumerate(gt_boxes):
                            if not matched[i] and iou(pred_box, gt_box) >= iou_thresh:
                                matched[i] = True
                                matched_flag = True
                                break
                        if matched_flag:
                            tps.append(1)
                            fps.append(0)
                        else:
                            tps.append(0)
                            fps.append(1)
                            
                except Exception as e:
                    print(f"문서 {doc_id} 처리 중 오류: {e}")
                    continue

            # 전체 GT가 0인 경우 AP = 0
            if total_gts == 0:
                ap_all.append(0.0)
                continue

            # 예측이 아예 없는 경우 AP = 0
            if len(tps) == 0:
                ap_all.append(0.0)
                continue

            tps = np.array(tps)
            fps = np.array(fps)
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            
            # Precision과 Recall 계산
            precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
            recalls = cum_tp / (total_gts + 1e-6)

            # recalls가 비어있는 경우 처리 (추가 안전장치)
            if len(recalls) == 0:
                ap_all.append(0.0)
                continue

            # Monotonic precision 계산 및 101-point interpolation
            precisions = np.maximum.accumulate(precisions[::-1])[::-1]
            recall_points = np.linspace(0, 1, 101)
            
            try:
                interp_precisions = np.interp(recall_points, recalls, precisions, left=0, right=0)
                ap = np.mean(interp_precisions)
                ap_all.append(ap)
            except Exception as e:
                print(f"카테고리 {category}, IoU {iou_thresh:.2f}: interpolation 오류 - {e}")
                ap_all.append(0.0)

    return np.mean(ap_all) if ap_all else 0.0

# ------------------------- OCR & Reading Order 평가 -------------------------
def process_document(args):
    """문서별 OCR 및 Reading Order 평가"""
    doc_id, answer_df, pred_df = args
    OCR_CATS = {'title', 'subtitle', 'text'}
    RO_CATS = {'title', 'subtitle', 'text', 'image', 'table', 'equation'}

    gt_items = answer_df[answer_df['ID'] == doc_id].copy()
    pred_items = pred_df[pred_df['ID'] == doc_id].copy()
    
    # 빈 데이터 처리
    if len(gt_items) == 0:
        return 0.0, 0.0
    
    try:
        gt_items['bbox'] = gt_items['bbox'].apply(parse_bbox)
        pred_items['bbox'] = pred_items['bbox'].apply(parse_bbox)
    except Exception as e:
        print(f"문서 {doc_id} bbox 파싱 오류: {e}")
        return 0.0, 0.0

    matched_gt, matched_pred = set(), set()
    ocr_dist, ro_pairs = [], []

    # GT 기준으로 1:1 매칭
    for i, gt in gt_items.iterrows():
        # order 컬럼 유효성 검사
        gt_order = gt.get('order', None)
        if pd.isna(gt_order):
            gt_order = None
            
        best_iou, best_j = 0, -1
        for j, pred in pred_items.iterrows():
            if j in matched_pred or gt['category_type'] != pred['category_type']:
                continue
                
            iou_val = iou(gt['bbox'], pred['bbox'])
            if iou_val >= 0.5 and iou_val > best_iou:
                best_iou, best_j = iou_val, j
                
        if best_j != -1:
            matched_gt.add(i)
            matched_pred.add(best_j)
            pred = pred_items.loc[best_j]
            
            # OCR 평가
            if gt['category_type'] in OCR_CATS:
                gt_text = gt.get('text', '')
                pred_text = pred.get('text', '')
                ocr_dist.append(normalized_edit_distance(gt_text, pred_text))
            
            # Reading Order 평가 - order 유효성 검사
            if gt['category_type'] in RO_CATS and gt_order is not None:
                pred_order = pred.get('order', None)
                if not pd.isna(pred_order):
                    ro_pairs.append((gt_order, pred_order))
        else:
            # 매칭 실패시 OCR 점수 1.0 추가 (최대 패널티)
            if gt['category_type'] in OCR_CATS:
                ocr_dist.append(1.0)

    # OCR 점수 계산
    ocr_score = 1 - np.mean(ocr_dist) if ocr_dist else 0.0
    print(f"[DEBUG] OCR NED Score: {ocr_score}")

    # Reading Order 점수 계산 - NaN 방지 강화
    if ro_pairs and len(ro_pairs) > 0:
        try:
            ro_pairs.sort(key=lambda x: x[0])  # GT order 기준 정렬
            gt_seq = [g for g, _ in ro_pairs]
            pred_seq = [p for _, p in ro_pairs]
            
            # NaN 체크
            if any(pd.isna(x) for x in gt_seq + pred_seq):
                ro_score = 0.0
            else:
                ned = ned_reading_order(gt_seq, pred_seq)
                
                # Coverage 계산 - 0으로 나누기 방지
                ro_eligible_items = gt_items[gt_items['category_type'].isin(RO_CATS)]
                if len(ro_eligible_items) > 0:
                    coverage = len(ro_pairs) / len(ro_eligible_items)
                    ro_score = (1 - ned) * coverage
                else:
                    ro_score = 0.0
        except Exception as e:
            print(f"문서 {doc_id} Reading Order 계산 오류: {e}")
            ro_score = 0.0
    else:
        ro_score = 0.0

    return ocr_score, ro_score

# ------------------------- 최종 평가 함수 -------------------------
def evaluate_document(answer_df, pred_df):
    """
    OmniDocBench 스타일 문서 평가
    - OCR (NED): 30%
    - Layout Detection (mAP@0.5:0.95): 35%
    - Reading Order (NED): 35%
    """
    ALLOWED_CATEGORIES = {'title', 'subtitle', 'text', 'image', 'table', 'equation'}
    
    # 데이터 유효성 검사
    if len(answer_df) == 0 or len(pred_df) == 0:
        return 0.0
    
    # 필수 컬럼 확인
    required_cols_answer = ['ID', 'category_type', 'order', 'text', 'bbox']
    required_cols_pred = ['ID', 'category_type', 'confidence_score', 'order', 'text', 'bbox']
    
    for col in required_cols_answer:
        if col not in answer_df.columns:
            raise ValueError(f"Answer 데이터에 '{col}' 컬럼이 없습니다.")
    for col in required_cols_pred:
        if col not in pred_df.columns:
            raise ValueError(f"Prediction 데이터에 '{col}' 컬럼이 없습니다.")
    
    # 카테고리 필터링
    answer_df = answer_df[answer_df['category_type'].isin(ALLOWED_CATEGORIES)].copy()
    pred_df = pred_df[pred_df['category_type'].isin(ALLOWED_CATEGORIES)].copy()

    # 병렬 처리로 OCR 및 Reading Order 평가
    n_processes = min(cpu_count(), 4)  # 최대 4개 프로세스
    tasks = [(doc_id, answer_df, pred_df) for doc_id in answer_df['ID'].unique()]

    try:
        with Pool(n_processes) as pool:
            results = pool.map(process_document, tasks)
    except Exception:
        results = [process_document(task) for task in tasks]

    # 결과 집계
    if results:
        ocr_scores, ro_scores = zip(*results)
        valid_ocr_scores = [s for s in ocr_scores if not pd.isna(s)]
        ocr_score = np.mean(valid_ocr_scores) if valid_ocr_scores else 0.0
        valid_ro_scores = [s for s in ro_scores if not pd.isna(s)]
        reading_order_score = np.mean(valid_ro_scores) if valid_ro_scores else 0.0
    else:
        ocr_score = 0.0
        reading_order_score = 0.0

    # Layout Detection 평가
    layout_score = compute_custom_map(answer_df, pred_df)

    # 최종 점수
    final_score = 0.30 * ocr_score + 0.35 * layout_score + 0.35 * reading_order_score

    # ---------------- Debug 출력 ----------------
    print("===================================")
    print(f"OCR Score (30%):          {ocr_score:.4f}")
    print(f"Layout mAP (35%):         {layout_score:.4f}")
    print(f"Reading Order Score (35%):{reading_order_score:.4f}")
    print("-----------------------------------")
    print(f"Final Score:              {final_score:.4f}")
    print("===================================")

    return final_score
