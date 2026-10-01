#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# -------------------- 설정값 --------------------
FULLSPAN_RATIO = 0.60      # 전폭 판단 비율(문서폭의 60% 이상이면 전폭)
GAP_MIN_ABS = 40           # 2컬럼 판정 시 최소 절대 갭(px)
GAP_MIN_RATIO = 0.18       # 2컬럼 판정 시 최소 상대 갭(문서폭의 18%)

# 겹침/포함을 같은 order로 묶는 규칙
MERGE_ON_INTERSECT = True      # 사각형이 조금이라도 겹치면 같은 order
MERGE_IOU_THRESH   = 0.0       # IoU 기준 (0.0이면 교집합만 있어도 묶임)
CONTAIN_RATIO_THR  = 0.85      # 포함: (작은 박스가 교집합으로 덮이는 비율) >= 0.85

# -------------------- 유틸 --------------------
def parse_bbox(bbox_str):
    """'x1, y1, x2, y2' 형태(따옴표/공백 허용) -> (x1,y1,x2,y2)"""
    try:
        if isinstance(bbox_str, str):
            xs = [t for t in re.split(r'[,\s]+', bbox_str.strip()) if t != ""]
        else:
            xs = list(bbox_str)
        x1, y1, x2, y2 = list(map(float, xs))
        # 정렬 및 최소 크기 보정
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        if x2 - x1 < 1: x2 = x1 + 1
        if y2 - y1 < 1: y2 = y1 + 1
        return (x1, y1, x2, y2)
    except Exception:
        return (0.0, 0.0, 1.0, 1.0)

def _estimate_img_size(boxes):
    """bbox로부터 페이지 크기 추정 (폭/높이)"""
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return (max(1.0, float(max(xs2))), max(1.0, float(max(ys2))))

# -------------------- 1/2컬럼 판별 --------------------
def detect_columns(boxes, img_width):
    """
    x중심 좌표 분포로 1/2컬럼 판정.
    return: (num_cols, split_x or None)
      - 2컬럼이면 split_x는 두 클러스터 센터의 중간값
    """
    if len(boxes) < 4:
        return 1, None

    x_centers = np.array([ (b[0]+b[2]) / 2.0 for b in boxes ], dtype=np.float32).reshape(-1,1)
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(x_centers)
    except Exception:
        return 1, None

    centers = np.sort(km.cluster_centers_.flatten())
    gap = centers[1] - centers[0]
    thr = max(GAP_MIN_ABS, img_width * GAP_MIN_RATIO)

    if gap > thr:
        split_x = (centers[0] + centers[1]) / 2.0
        return 2, float(split_x)
    return 1, None

# -------------------- 전폭 섹션 분리 --------------------
def _split_sections_by_fullwidth(indices, boxes, img_width):
    """
    전폭 요소를 경계로 섹션 분할.
    반환: [섹션별 인덱스 리스트]  (전폭 요소는 자신의 섹션 맨 앞에 오도록 구성)
    """
    is_full = np.array([(boxes[i][2]-boxes[i][0]) >= (FULLSPAN_RATIO * img_width) for i in indices], dtype=bool)

    # 전폭 요소 없으면 섹션 하나
    full_idx = [indices[i] for i, f in enumerate(is_full) if f]
    if not full_idx:
        return [indices]

    # 전폭 요소 y1 기준으로 경계 생성
    full_idx_sorted = sorted(full_idx, key=lambda j: boxes[j][1])
    cuts = [boxes[j][1] for j in full_idx_sorted]
    bounds = [-float('inf')] + cuts + [float('inf')]

    sections = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = [j for j in indices if (boxes[j][1] >= lo and boxes[j][1] < hi)]
        fw = [j for j in seg if (boxes[j][2]-boxes[j][0]) >= (FULLSPAN_RATIO * img_width)]
        rest = [j for j in seg if j not in fw]
        fw_sorted = sorted(fw, key=lambda j: boxes[j][1])  # 전폭 먼저
        sections.append(fw_sorted + rest)
    return sections

# -------------------- 겹침/포함 판정 유틸 --------------------
def _area(b):
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def _inter(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2-x1)*(y2-y1)

def _iou(b1, b2):
    inter = _inter(b1, b2)
    if inter <= 0:
        return 0.0
    return inter / (_area(b1)+_area(b2)-inter + 1e-9)

def _is_contained_enough(b_small, b_big, ratio_thr=CONTAIN_RATIO_THR):
    """작은 박스가 큰 박스 안에 충분히 들어가는지 (교집합/작은면적 >= thr)"""
    inter = _inter(b_small, b_big)
    a_s   = _area(b_small) + 1e-9
    return (inter / a_s) >= ratio_thr

def _overlap_or_contain(b1, b2):
    """겹치거나 포함이면 True"""
    if _inter(b1, b2) > 0:
        if MERGE_ON_INTERSECT:
            return True
        # IoU 기준 사용하고 싶을 때
        if _iou(b1, b2) >= MERGE_IOU_THRESH:
            return True
    # 포함 체크(양방향) - 더 작은 쪽이 큰 쪽에 충분히 들어가면 묶음
    if _area(b1) <= _area(b2):
        if _is_contained_enough(b1, b2):
            return True
    else:
        if _is_contained_enough(b2, b1):
            return True
    return False

# -------------------- order 재부여 --------------------
def reorder_boxes(df_group):
    """
    하나의 ID 그룹에 대해 order 재부여.
    - 전폭 섹션 분리 → 섹션 내 정렬
    - 2컬럼 시 split_x로 좌/우 나눠 각 컬럼 y→x 정렬, 좌→우 결합
    - 1컬럼 시 섹션 내 y→x 정렬
    - 마지막에 겹치는 박스들 중 가장 큰 박스만 기존 order 유지, 나머지는 새로운 order 부여
    """
    df_group = df_group.copy()
    boxes = [parse_bbox(b) for b in df_group["bbox"]]
    ids = list(range(len(boxes)))

    img_w, img_h = _estimate_img_size(boxes)
    num_cols, split_x = detect_columns(boxes, img_w)

    new_order_idx = []
    if num_cols == 1:
        # 전폭 섹션 기준으로만 끊고, 섹션 내 y→x
        for seg in _split_sections_by_fullwidth(ids, boxes, img_w):
            fw = [j for j in seg if (boxes[j][2]-boxes[j][0]) >= (FULLSPAN_RATIO * img_w)]
            rest = [j for j in seg if j not in fw]
            rest_sorted = sorted(rest, key=lambda j: (boxes[j][1], boxes[j][0]))
            new_order_idx.extend(fw + rest_sorted)
    else:
        # 2컬럼: 섹션별로 좌/우 분리 후 각자 y→x
        for seg in _split_sections_by_fullwidth(ids, boxes, img_w):
            fw = [j for j in seg if (boxes[j][2]-boxes[j][0]) >= (FULLSPAN_RATIO * img_w)]
            rest = [j for j in seg if j not in fw]
            left  = [j for j in rest if ((boxes[j][0]+boxes[j][2])/2.0) < split_x]
            right = [j for j in rest if j not in left]
            left_sorted  = sorted(left,  key=lambda j: (boxes[j][1], boxes[j][0]))
            right_sorted = sorted(right, key=lambda j: (boxes[j][1], boxes[j][0]))
            new_order_idx.extend(sorted(fw, key=lambda j: boxes[j][1]) + left_sorted + right_sorted)

    # df 재정렬
    df_out = df_group.iloc[new_order_idx].copy()

    # ----- 수정된 order 묶기 로직: 겹치는 것 중 가장 큰 박스만 같은 order 유지 -----
    seq_boxes = [boxes[i] for i in new_order_idx]
    orders = [None] * len(seq_boxes)  # 정확한 길이로 초기화
    cur_order = 0
    processed = [False] * len(seq_boxes)

    for i in range(len(seq_boxes)):
        if processed[i]:
            continue
            
        # 현재 박스와 겹치는 모든 박스들 찾기 (자신 포함)
        overlapping_indices = [i]
        for j in range(i + 1, len(seq_boxes)):
            if not processed[j] and _overlap_or_contain(seq_boxes[i], seq_boxes[j]):
                overlapping_indices.append(j)
        
        if len(overlapping_indices) == 1:
            # 겹치는 박스가 없으면 현재 order 부여
            orders[i] = cur_order
            processed[i] = True
        else:
            # 겹치는 박스들 중에서 가장 큰 박스 찾기
            largest_idx = max(overlapping_indices, key=lambda idx: _area(seq_boxes[idx]))
            
            # 가장 큰 박스는 현재 order 유지
            orders[largest_idx] = cur_order
            processed[largest_idx] = True
            
            # 나머지 겹치는 박스들은 새로운 order들 부여
            for idx in overlapping_indices:
                if idx != largest_idx:
                    cur_order += 1
                    orders[idx] = cur_order
                    processed[idx] = True
        
        cur_order += 1

    df_out["order"] = orders
    return df_out

# -------------------- CSV 단위 실행 --------------------
def reorder_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    required = {"ID", "bbox"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    out_groups = []
    for id_val, g in df.groupby("ID", sort=False):
        out_groups.append(reorder_boxes(g))
    new_df = pd.concat(out_groups).reset_index(drop=True)
    new_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Done] 순서 재정렬 저장 완료: {output_csv}")

# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./output/submission.csv")
    ap.add_argument("--output", default="./output/submission_reorder.csv")
    args = ap.parse_args()

    reorder_csv(args.input, args.output)