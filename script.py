import os
import sys
import json, yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import pathlib
import shutil
import subprocess
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import importlib.util
from typing import List, Dict, Tuple
from model.order.order_again import reorder_csv

# -------------------------
# 설정
# -------------------------
YOLO_WEIGHTS = "./model/yolo/doclayout_yolo_docstructbench.pt"
IMPROVED_YOLO_WEIGHTS = "./model/yolo/doclayout_yolo_improved.pt"  # 있으면 우선 사용
INLINE_YOLO_WEIGHTS = "./model/yolo/formula_yolo.pt"  # 인라인 수식 탐지용(ultralytics)
PIX2TEX_OFFLINE_DIR = r"./model/pix2tex"             # weights.pth, image_resizer.pth 위치

TEST_CSV = "./data/test.csv"
OUTPUT_CSV = "./output/submission.csv"
TEMP_IMG_DIR = "./temp_images_44"
VIZ_DIR = "./output/viz_45"

# 안정화된 인라인 수식 탐지 설정
INLINE_CONF = 0.30  # 더 보수적인 임계값으로 안정성 향상
INLINE_IMGSZ = 1024  # 안정적인 해상도

# 60분 내 실행 + 0.4+ 정확도를 위한 균형 파라미터
DETECTION_CONF_THRESHOLD = 0.06  # 낮춰서 더 많은 검출로 정확도 향상
DETECTION_IOU_THRESHOLD = 0.40   # 낮춰서 겹침 제거 강화
DETECTION_IMGSZ = 1024           # 적당한 해상도로 속도 향상
MAX_DET = 2500                   # 더 많은 검출 허용

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Ultralytics 환경 차단(오프라인/자동설치 방지)
os.environ["YOLO_CONFIG_DIR"] = str(pathlib.Path("./.ultralytics").resolve())
os.environ["YOLO_AUTOINSTALL"] = "0"
os.environ["YOLO_VERBOSE"] = "0"

cfg = pathlib.Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics"
cfg.mkdir(parents=True, exist_ok=True)
sp = cfg / "settings.json"
if not sp.exists():
    sp.write_text(json.dumps({
        "datasets_dir": str((cfg / "datasets").resolve()),
        "runs_dir": str((cfg / "runs").resolve()),
        "sync": False,
        "checks": False,
        "api_key": ""
    }, indent=2), encoding="utf-8")

# -------------------------
# 전역 엔진
# -------------------------
OCR: PaddleOCR | None = None
model = None              # DocLayout-YOLO (YOLOv10 포맷)
inline_model = None       # 인라인 수식용 YOLO(ultralytics)
TEX = None                # pix2tex (LatexOCR)

CATEGORY_COLOR = {
    'title':    (255, 0, 0),      # Red
    'subtitle': (255, 165, 0),    # Orange
    'text':     (0, 255, 0),      # Green
    'image':    (0, 0, 255),      # Blue
    'table':    (128, 0, 128),    # Purple
    'equation': (255, 0, 255),    # Magenta
    'inline_formula': (0, 0, 0)   # Black
}

# DocLayNet → 제출 카테고리 매핑
LABEL_MAP = {
    'plain text': 'text',
    'title': 'title',
    'isolate_formula': 'equation',
    'table': 'table',
    'figure': 'image',
    'subtitle':'subtitle'
}

# -------------------------
# 유틸
# -------------------------
_FONTS_READY = False

def _load_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        return ImageFont.load_default()

def _safe_name(id_val):
    return re.sub(r'[<>:"/\\|?*]', '_', str(id_val))

def _normalize_text(s):
    if s is None:
        return ""
    return " ".join(str(s).split())

def ensure_system_fonts():
    """PPTX/PNG 렌더링 시 한글 폰트 이슈를 줄이기 위한 폰트 등록(없어도 동작)."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    try:
        spec = importlib.util.find_spec("koreanize_matplotlib")
        if spec is None:
            print("[fonts] koreanize_matplotlib 패키지가 없음(무시 가능)")
            _FONTS_READY = True
            return

        import pkg_resources, koreanize_matplotlib  # noqa
        fonts_dir = pkg_resources.resource_filename('koreanize_matplotlib', 'fonts')
        if os.path.exists(fonts_dir):
            sys_fonts_dir = "/usr/share/fonts/truetype/pipfonts"
            os.makedirs(sys_fonts_dir, exist_ok=True)
            for fname in os.listdir(fonts_dir):
                if fname.lower().endswith(".ttf"):
                    src = os.path.join(fonts_dir, fname)
                    dst = os.path.join(sys_fonts_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
            try:
                subprocess.run(["fc-cache", "-f", "-v"], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("[fonts] 시스템 폰트 캐시 업데이트 완료")
            except Exception as e:
                print(f"[fonts] 폰트 캐시 업데이트 실패(무시 가능): {e}")
    except Exception as e:
        print(f"[fonts] 폰트 등록 실패(무시 가능): {e}")
    _FONTS_READY = True

def convert_to_images(input_path, temp_dir, dpi=300):
    """PDF/PPTX/JPG/PNG → PIL Image 리스트"""
    ext = Path(input_path).suffix.lower()
    import time
    unique_temp_dir = os.path.join(temp_dir, f"temp_{int(time.time())}")
    os.makedirs(unique_temp_dir, exist_ok=True)

    try:
        if ext in [".pptx", ".pdf"]:
            ensure_system_fonts()

        if ext == ".pdf":
            try:
                images = convert_from_path(input_path, dpi=dpi, output_folder=unique_temp_dir, fmt="png")
                return images
            except Exception as e:
                raise RuntimeError(f"PDF 변환 실패: {e}")

        elif ext == ".pptx":
            try:
                subprocess.run([
                    "libreoffice", "--headless", "--convert-to", "png",
                    "--outdir", unique_temp_dir, input_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

                pngs = sorted([p for p in os.listdir(unique_temp_dir) if p.lower().endswith(".png")])
                if pngs:
                    images = []
                    for p in pngs:
                        img_path = os.path.join(unique_temp_dir, p)
                        try:
                            img = Image.open(img_path).convert("RGB")
                            images.append(img)
                        except Exception as e:
                            print(f" 이미지 로드 실패: {img_path}, {e}")

                    # 중복 제거(동일 크기 중복 컷 제거)
                    unique_images, seen = [], set()
                    for img in images:
                        size_hash = (img.size, len(img.getdata()))
                        if size_hash not in seen:
                            unique_images.append(img)
                            seen.add(size_hash)
                    return unique_images
                else:
                    raise RuntimeError("PPTX→PNG 변환 결과가 비어있음")
            except subprocess.TimeoutExpired:
                raise RuntimeError("PPTX 변환 시간 초과 (120초)")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"LibreOffice 변환 실패: {e}")

        elif ext in [".jpg", ".jpeg", ".png"]:
            try:
                return [Image.open(input_path).convert("RGB")]
            except Exception as e:
                raise RuntimeError(f"이미지 로드 실패: {e}")
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
    finally:
        try:
            if os.path.exists(unique_temp_dir):
                shutil.rmtree(unique_temp_dir)
        except Exception:
            pass

def scale_bbox_to_target(bbox, current_size, target_size):
    """bbox(현재좌표)를 target_size 좌표계로 매우 정확하게 스케일"""
    if len(bbox) != 4:
        raise ValueError(f"bbox는 4개 값이어야 함: {bbox}")
    x1, y1, x2, y2 = bbox
    cw, ch = current_size
    tw, th = target_size
    if cw <= 0 or ch <= 0:
        raise ValueError(f"현재 크기 무효: {current_size}")
    
    # 정확한 스케일링 비율
    sx, sy = tw / cw, th / ch
    
    # 좌표 변환 (더 정밀한 계산)
    scaled_x1 = x1 * sx
    scaled_y1 = y1 * sy
    scaled_x2 = x2 * sx
    scaled_y2 = y2 * sy
    
    # 경계 검사 및 최소 크기 보장 (더 정밀하게)
    min_size = max(2, min(tw, th) * 0.002)  # 최소 크기를 더 크게 설정
    scaled_x1 = max(0, min(scaled_x1, tw - min_size))
    scaled_y1 = max(0, min(scaled_y1, th - min_size))
    scaled_x2 = max(scaled_x1 + min_size, min(scaled_x2, tw))
    scaled_y2 = max(scaled_y1 + min_size, min(scaled_y2, th))
    
    # 정밀한 반올림
    return [int(round(scaled_x1)), int(round(scaled_y1)), int(round(scaled_x2)), int(round(scaled_y2))]

def refine_bbox(bbox, image_size, min_size=8):
    """텍스트 추출 최적화를 위한 bbox 좌표 정제"""
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size
    
    # 입력값 검증
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return [0, 0, min_size, min_size]
    
    # 좌표 정규화 (min < max 보장)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # 텍스트 포함도를 높이기 위한 적절한 여백
    margin_x = max(2, int(img_w * 0.002))  # 최소 2픽셀, 이미지 폭의 0.2%
    margin_y = max(1, int(img_h * 0.001))  # 최소 1픽셀, 이미지 높이의 0.1%
    
    # 여백 추가
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img_w, x2 + margin_x)
    y2 = min(img_h, y2 + margin_y)
    
    # 이미지 경계 내로 제한
    x1 = max(0, min(x1, img_w - min_size))
    y1 = max(0, min(y1, img_h - min_size))
    x2 = max(x1 + min_size, min(x2, img_w))
    y2 = max(y1 + min_size, min(y2, img_h))
    
    # 최소 크기 보장
    if x2 - x1 < min_size:
        center_x = (x1 + x2) / 2
        x1 = max(0, center_x - min_size / 2)
        x2 = min(img_w, center_x + min_size / 2)
    
    if y2 - y1 < min_size:
        center_y = (y1 + y2) / 2
        y1 = max(0, center_y - min_size / 2)
        y2 = min(img_h, center_y + min_size / 2)
    
    # 좌표 정밀도 향상 (소수점 반올림)
    final_bbox = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    
    # 유효성 재검증
    if (final_bbox[2] <= final_bbox[0] or final_bbox[3] <= final_bbox[1] or
        final_bbox[0] < 0 or final_bbox[1] < 0 or
        final_bbox[2] > img_w or final_bbox[3] > img_h):
        # 안전한 기본값 반환
        return [0, 0, min_size, min_size]
    
    return final_bbox

def visualize_detections(image_pil, overlays, save_path):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    font = _load_font()

    W, H = img.size
    print(f"시각화 이미지 크기: {W}x{H}")

    valid_overlays = 0
    for i, overlay in enumerate(overlays):
        if len(overlay) == 6:
            x1, y1, x2, y2, cat_or_label, score = overlay
            label = f"{cat_or_label} {score:.2f}"
        elif len(overlay) == 7:
            x1, y1, x2, y2, cat_or_label, score, order = overlay
            label = f"[{order}] {cat_or_label} {score:.2f}"
        else:
            print(f"Overlay 형식 오류: {overlay} (길이: {len(overlay)})")
            continue

        try:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        except (ValueError, TypeError) as e:
            print(f"좌표 변환 실패 [{i}]: {overlay} -> {e}")
            continue

        # 좌표 정렬
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        # Clamp to image bounds
        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        x2 = min(int(max(x2, x1 + 1)), W - 1)
        y2 = min(int(max(y2, y1 + 1)), H - 1)

        if x2 - x1 < 3 or y2 - y1 < 3:
            print(f"박스 크기 너무 작음 [{i}]: {x1},{y1},{x2},{y2}")
            continue

        valid_overlays += 1
        color = CATEGORY_COLOR.get(cat_or_label, (255, 255, 0))

        try:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        except Exception as e:
            print(f"박스 그리기 실패 [{i}]: {e}")
            continue

        try:
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                tw, th = draw.textsize(label, font=font)

            label_y = max(0, y1 - th - 2)
            label_x2 = min(W, x1 + tw + 4)
            label_y2 = min(H, label_y + th + 2)

            draw.rectangle([x1, label_y, label_x2, label_y2], fill=color)
            draw.text((x1 + 2, label_y), label, fill=(255, 255, 255), font=font)
        except Exception as e:
            print(f"라벨 그리기 실패 [{i}]: {e}")

    print(f"유효한 overlay: {valid_overlays}/{len(overlays)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)
    print(f"🖼️ 시각화 저장: {save_path}")


# -------------------------
# 초기화
# -------------------------

def initialize_ocr():
    global OCR
    if OCR is not None:
        return

    det_dir = Path("./model/paddleocr/det/PP-OCRv5_server_det")
    rec_dir = Path("./model/paddleocr/rec/korean_PP-OCRv5_mobile_rec_infer")

    
    det_name = "PP-OCRv5_server_det"
    rec_name = "korean_PP-OCRv5_mobile_rec"

    # kwargs = {
    #     "text_detection_model_dir": str(det_dir),
    #     "text_recognition_model_dir": str(rec_dir),
    #     "use_textline_orientation": False,
    #     "lang": "korean"
    # }

    kwargs = {
        "text_detection_model_dir": str(det_dir),
        "text_detection_model_name": det_name,
        "text_recognition_model_dir": str(rec_dir),
        "text_recognition_model_name": rec_name,

        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,

        "lang": "korean"
    }


    print(f" [OCR]\n"
          f"  det_dir={det_dir}\n  rec_dir={rec_dir}")

    from paddleocr import PaddleOCR
    OCR = PaddleOCR(**kwargs)
    print("✅ PaddleOCR 초기화 완료 (경량 구조)")

def initialize_yolo():
    """DocLayout-YOLO(v10) 초기화: 개선 가중치 우선"""
    global model
    if model is not None:
        return
    weights_to_try = []
    if os.path.exists(IMPROVED_YOLO_WEIGHTS):
        weights_to_try.append(IMPROVED_YOLO_WEIGHTS)
    if os.path.exists(YOLO_WEIGHTS):
        weights_to_try.append(YOLO_WEIGHTS)
    if not weights_to_try:
        raise FileNotFoundError("YOLO 가중치를 찾을 수 없습니다.")
    last_err = None
    for w in weights_to_try:
        try:
            from doclayout_yolo import YOLOv10
            model = YOLOv10(w)
            print(f" YOLO 모델 초기화 완료: {w}")
            return
        except Exception as e:
            print(f" YOLO 로드 실패({w}): {e}")
            last_err = e
    raise RuntimeError(f"YOLO 초기화 실패: {last_err}")

def initialize_inline_yolo():
    """인라인 수식 탐지용 ultralytics YOLO 초기화"""
    global inline_model
    if inline_model is not None:
        return
    if not os.path.exists(INLINE_YOLO_WEIGHTS):
        raise FileNotFoundError(f"인라인 수식 YOLO 가중치 없음: {INLINE_YOLO_WEIGHTS}")
    inline_model = YOLO(INLINE_YOLO_WEIGHTS)
    print(" 인라인 수식 YOLO 초기화 완료")

def initialize_pix2tex():
    """pix2tex(선택) 오프라인 가중치 로딩(없으면 None 유지)"""
    global TEX
    if TEX is not None:
        return
    try:
        from pix2tex.cli import LatexOCR
        import pix2tex as _p2t
        ckpt_dir = pathlib.Path(_p2t.__file__).parent / "model" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        src_dir = pathlib.Path(PIX2TEX_OFFLINE_DIR)
        src_w = src_dir / "weights.pth"
        src_r = src_dir / "image_resizer.pth"
        if not src_w.is_file() or not src_r.is_file():
            raise FileNotFoundError(f"pix2tex 가중치가 없습니다: {src_w}, {src_r}")
        if not (ckpt_dir / "weights.pth").exists():
            shutil.copy2(src_w, ckpt_dir / "weights.pth")
        if not (ckpt_dir / "image_resizer.pth").exists():
            shutil.copy2(src_r, ckpt_dir / "image_resizer.pth")
        TEX = LatexOCR()
        print(f" pix2tex 초기화 완료 @ {ckpt_dir}")
    except Exception as e:
        TEX = None
        print(f" pix2tex 초기화 실패(무시 가능): {e}")

# -------------------------
# 품질/안정성 모듈
# -------------------------
def post_process_text(text, category):
    """카테고리별 텍스트 후처리"""
    if not text:
        return ""
    text = text.strip()
    if category == 'title':
        text = text.capitalize()
        text = re.sub(r'[^\w\s가-힣]', '', text)
    elif category == 'subtitle':
        text = text.capitalize()
    elif category == 'text':
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*([A-Z가-힣])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def adjust_confidence_score(original_score, text, category):
    """0.4+ 정확도를 위한 점수 보정 강화"""
    base = float(original_score)
    
    # 빈 텍스트에 대해서는 점수를 낮춤
    if not text or not text.strip():
        return max(0.32, base * 0.75)  # 빈 텍스트는 조금 더 관대하게
    
    text_len = len(text.strip())
    
    # 길이별 보정 (정확도 우선) - 더 관대하게
    if text_len < 2:
        base *= 0.95  # 짧은 텍스트도 조금 더 관대하게
    elif text_len >= 2:
        base *= 1.15  # 2글자 이상이면 더 큰 보너스
    if text_len > 5:
        base *= 1.20  # 5글자 이상이면 더 큰 보너스
    if text_len > 15:
        base *= 1.15   # 긴 텍스트는 추가 보너스
    
    # 카테고리별 보정 (정확도 우선) - 더 관대하게
    if category == 'title':
        if text_len >= 2:
            base *= 1.30  # 더 큰 보너스
        base += 0.10  # 더 큰 기본 보너스
    elif category == 'subtitle':
        if text_len >= 2:
            base *= 1.25  # 더 큰 보너스
        base += 0.08  # 더 큰 기본 보너스
    elif category == 'text':
        if text_len >= 1:
            base *= 1.15  # 더 큰 보너스
        base += 0.07  # 더 큰 기본 보너스
    elif category == 'equation':
        # 수식은 더 관대하게
        if text_len >= 1:
            base *= 1.35  # 더 큰 보너스
        base += 0.12  # 더 큰 기본 보너스
    elif category == 'table':
        if text_len >= 1:
            base *= 1.20  # 더 큰 보너스
        base += 0.08  # 더 큰 기본 보너스
    
    # 언어별 보정 - 더 관대하게
    if re.search(r'[가-힣]', text):
        base *= 1.12  # 한글 보너스 증가
    if re.search(r'[a-zA-Z]', text):
        base *= 1.08  # 영어 보너스 증가
    
    # 숫자 포함 보정 - 더 관대하게
    if re.search(r'\d', text):
        base *= 1.10
    
    # 수식 패턴 보정 - 더 관대하게
    if re.search(r'[=\+\-\*/\^\(\)∫∑∏]', text):
        if category == 'equation':
            base *= 1.25  # equation에 더 큰 보너스
        else:
            base *= 1.08
    
    # 최소/최대 점수 제한 (정확도 우선) - 더 관대하게
    return min(0.99, max(0.32, base))

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def reclassify_after_title(layout_items):
    """title이 하나 나오면 나머지는 subtitle/text로 재분류 (이미지/표/수식 보존)"""
    if not layout_items:
        return layout_items
    
    # title 개수 확인
    title_items = [item for item in layout_items if item['cls'] == 'title']
    
    if len(title_items) == 0:
        return layout_items  # title이 없으면 그대로
    
    if len(title_items) == 1:
        # title이 하나면 나머지 title들만 subtitle로 변경
        for item in layout_items:
            if item['cls'] == 'title':
                # 첫 번째 title은 그대로 유지
                continue
            elif item['cls'] in ['subtitle', 'text', 'image', 'table', 'equation']:
                # 중요한 카테고리는 그대로 유지
                continue
            else:
                # 기타 카테고리만 text로 변경
                item['cls'] = 'text'
    else:
        # title이 여러 개면 첫 번째만 title, 나머지는 subtitle로
        title_items.sort(key=lambda x: x['box'][1])  # y좌표로 정렬
        
        for i, item in enumerate(layout_items):
            if item['cls'] == 'title':
                if item == title_items[0]:
                    # 첫 번째 title만 유지
                    continue
                else:
                    # 나머지 title들은 subtitle로 변경
                    item['cls'] = 'subtitle'
            elif item['cls'] not in ['subtitle', 'text', 'image', 'table', 'equation']:
                # 기타 카테고리만 text로 변경
                item['cls'] = 'text'
    
    return layout_items

def post_process_layouts(layout_items, image_size):
    """텍스트 추출 및 bbox 품질 향상을 위한 후처리"""
    if not layout_items:
        return layout_items
    
    # 0) 모든 bbox 좌표 정제 (깔끔한 크기)
    for item in layout_items:
        item['box'] = refine_bbox(item['box'], image_size, min_size=8)
    
    # 1) 수식 재분류 로직 (text → equation)
    for item in layout_items:
        if item['cls'] == 'text' and item['score'] > 0.1:
            # 수식 패턴이 포함된 텍스트를 equation으로 재분류
            text_content = item.get('text', '')
            if (re.search(r'[=+\-*/^()∫∑∏Δ]', text_content) or 
                re.search(r'[A-Za-z]\s*[=]\s*[A-Za-z]', text_content) or
                len(text_content) < 50 and re.search(r'[A-Za-z]\s*[+\-*/]\s*[A-Za-z]', text_content)):
                item['cls'] = 'equation'
                item['score'] = max(item['score'], 0.15)  # equation 최소 점수 보장
    
    # 2) 카테고리별 신뢰도 필터링 (텍스트 탐지 강화)
    items = []
    for it in layout_items:
        cat = it['cls']
        score = it['score']
        
        # 카테고리별 신뢰도 임계값 (정확도와 속도 균형) - 더 관대하게
        if cat in ['title', 'subtitle'] and score >= 0.06:  # 낮춰서 더 많은 검출
            items.append(it)
        elif cat == 'text' and score >= 0.04:  # 더 낮은 텍스트 임계값
            items.append(it)
        elif cat == 'image' and score >= 0.10:  # 낮춰서 더 많은 검출
            items.append(it)
        elif cat == 'table' and score >= 0.06:  # 낮춰서 더 많은 검출
            items.append(it)
        elif cat == 'equation' and score >= 0.03:  # 더 낮은 equation 임계값
            items.append(it)
        elif score >= 0.06:  # 기타 카테고리 더 낮은 임계값
            items.append(it)
    
    # 3) 최소 크기 조건 (적절한 크기)
    W, H = image_size
    min_area = (W * H) * 0.00005  # 적절한 최소 면적
    items = [it for it in items if (it['box'][2]-it['box'][0])*(it['box'][3]-it['box'][1]) >= min_area]
    
    # 4) 강화된 NMS로 겹침 제거 (inline_formula 중복 제거)
    cats = set(it['cls'] for it in items)
    final_items = []
    
    for cat in cats:
        citems = [it for it in items if it['cls'] == cat]
        if len(citems) <= 1:
            final_items.extend(citems)
            continue
        
        # 점수 기준 내림차순 정렬
        citems.sort(key=lambda x: x['score'], reverse=True)
        
        keep_items = []
        for current_item in citems:
            keep = True
            
            for kept_item in keep_items:
                iou = calculate_iou(current_item['box'], kept_item['box'])
                
                # 카테고리별 IoU 임계값 (겹침 제거 강화) - 더 관대하게
                if cat in ['title', 'subtitle']:
                    threshold = 0.5  # 낮춰서 더 많은 검출 허용
                elif cat == 'text':
                    threshold = 0.4  # 낮춰서 더 많은 검출 허용
                elif cat == 'image':
                    threshold = 0.6  # 낮춰서 더 많은 검출 허용
                elif cat == 'table':
                    threshold = 0.3  # 더 낮은 IoU로 더 많은 검출 허용
                else:  # equation
                    threshold = 0.25  # 더 낮은 IoU (독립 수식 허용)
                
                # 겹침 제거 조건 - 더 관대하게
                score_diff = abs(current_item['score'] - kept_item['score'])
                if iou > threshold:
                    if score_diff > 0.15:  # 점수 차이를 더 작게 해서 더 관대하게
                        if current_item['score'] > kept_item['score']:
                            keep_items.remove(kept_item)
                            continue
                        else:
                            keep = False
                            break
                    else:
                        keep = False
                        break
            
            if keep:
                keep_items.append(current_item)
        
        final_items.extend(keep_items)
    
    # 5) 최종 신뢰도 필터링 (텍스트 탐지 강화)
    filtered_items = []
    for item in final_items:
        cat = item['cls']
        score = item['score']
        
        # 최종 필터링 (정확도와 속도 균형) - 더 관대하게
        if cat == 'equation' and score >= 0.03:  # 더 낮은 equation 임계값
            filtered_items.append(item)
        elif cat in ['title', 'subtitle'] and score >= 0.06:  # 낮춰서 더 많은 검출
            filtered_items.append(item)
        elif cat == 'text' and score >= 0.04:  # 더 낮은 텍스트 임계값
            filtered_items.append(item)
        elif cat == 'image' and score >= 0.10:  # 낮춰서 더 많은 검출
            filtered_items.append(item)
        elif cat == 'table' and score >= 0.06:  # 낮춰서 더 많은 검출
            filtered_items.append(item)
        elif score >= 0.06:  # 기타 카테고리 더 낮은 임계값
            filtered_items.append(item)
    
    # 6) y좌표 정렬
    filtered_items.sort(key=lambda x: x['box'][1])
    return filtered_items

# -------------------------
# 인라인 수식 매칭/병합
# -------------------------
def get_inline_map_no_touch(layouts: List[Dict], image_pil: Image.Image,
                            conf=INLINE_CONF, imgsz=INLINE_IMGSZ,
                            thr_inline=0.60, thr_block=0.50,
                            cov_tie_eps=0.05, border_eps=5):
    """페이지 전체에서 인라인 수식을 검출 → 텍스트형 레이아웃에 배정 (중복 제거 강화)"""
    if inline_model is None:
        raise RuntimeError("인라인 YOLO 초기화 안 됨")

    arr = np.array(image_pil.convert("RGB"))
    res = inline_model.predict(arr, conf=conf, imgsz=imgsz,
                               device=device, verbose=False)[0]
    formulas = []
    if res.boxes is not None:
        for b in res.boxes.xyxy.cpu().numpy():
            formulas.append(b.tolist())

    eq_boxes = [(i, lt['box']) for i, lt in enumerate(layouts) if lt['cls'] == "equation"]
    text_idxs = [i for i, lt in enumerate(layouts) if lt['cls'] in ("text", "title", "subtitle")]
    inline_map: Dict[int, List[List[float]]] = {}

    # 인라인 수식 중복 제거 (겹치는 수식 제거)
    filtered_formulas = []
    for i, f in enumerate(formulas):
        fx1, fy1, fx2, fy2 = f
        f_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
        if f_area == 0:
            continue
            
        # 다른 수식과 겹치는지 확인
        is_duplicate = False
        for other_f in filtered_formulas:
            ofx1, ofy1, ofx2, ofy2 = other_f
            of_area = max(0, ofx2 - ofx1) * max(0, ofy2 - ofy1)
            
            # IoU 계산
            ix1, iy1 = max(fx1, ofx1), max(fy1, ofy1)
            ix2, iy2 = min(fx2, ofx2), min(fy2, ofy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = f_area + of_area - inter
            iou = inter / union if union > 0 else 0
            
            if iou > 0.5:  # 높은 IoU로 중복 제거
                is_duplicate = True
                break
                
        if not is_duplicate:
            filtered_formulas.append(f)

    for f in filtered_formulas:
        fx1, fy1, fx2, fy2 = f
        f_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
        if f_area == 0:
            continue

        # 블록 수식과 많이 겹치면 인라인 아님
        is_block = False
        for _, eq in eq_boxes:
            ex1, ey1, ex2, ey2 = eq
            ix1, iy1 = max(fx1, ex1), max(fy1, ey1)
            ix2, iy2 = min(fx2, ex2), min(fy2, ey2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / f_area >= thr_block:
                is_block = True
                break
        if is_block:
            continue

        # 텍스트형 레이아웃 중 포함율 최대인 대상 찾기
        best_i, best_cov = -1, 0.0
        for i in text_idxs:
            tx1, ty1, tx2, ty2 = layouts[i]['box']
            fexp = [fx1 - 2, fy1 - 1, fx2 + 2, fy2 + 1]  # 경계 보완
            ix1, iy1 = max(fexp[0], tx1), max(fexp[1], ty1)
            ix2, iy2 = min(fexp[2], tx2), min(fexp[3], ty2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            cov = inter / f_area
            if cov > best_cov:
                best_i, best_cov = i, cov

        if best_i == -1:
            continue

        # 포함율 동률이면 y-center 가까운 쪽
        ties = [i for i in text_idxs if abs((best_cov) - (inter / f_area if f_area else 0)) < cov_tie_eps]
        if len(ties) > 1:
            fc_y = (fy1 + fy2) / 2
            best_i = min(ties, key=lambda i: abs(fc_y - (layouts[i]['box'][1] + layouts[i]['box'][3]) / 2))

        # 수평 경계에 바짝 붙은 경우 차선 후보 고려
        tx1, _, tx2, _ = layouts[best_i]['box']
        fc_x = (fx1 + fx2) / 2
        near_border = (abs(fc_x - tx1) < border_eps) or (abs(fc_x - tx2) < border_eps)
        if near_border and len(text_idxs) > 1:
            def overlap_ratio(i):
                t = layouts[i]['box']
                ox = max(0, min(fx2, t[2]) - max(fx1, t[0]))
                oy = max(0, min(fy2, t[3]) - max(fy1, t[1]))
                return (ox * oy) / f_area if f_area > 0 else 0
            ranks = sorted(text_idxs, key=lambda i: -overlap_ratio(i))
            if len(ranks) >= 2:
                best_i = ranks[1]

        # 최종 포함율 기준
        if best_cov >= thr_inline or (best_cov >= (thr_inline - 0.1) and tx1 <= fc_x <= tx2):
            inline_map.setdefault(best_i, []).append(f)

    return inline_map

def build_text_with_inline(image_pil, region_box, inline_boxes, conf_thres=0.0, line_eps=None):
    """텍스트 추출 품질 향상을 위한 폴리곤 중심 좌표 기반 라인/좌우 정렬 + 인라인 수식 LaTeX 병합"""
    if OCR is None:
        raise RuntimeError("OCR이 초기화되지 않았습니다")
    if line_eps is None:
        line_eps = int(round(image_pil.height * 0.012))  # 더 정밀한 라인 구분

    x1, y1, x2, y2 = map(int, region_box)
    crop = image_pil.crop((x1, y1, x2, y2)).copy()

    # 인라인 수식 영역 화이트 마스킹 (더 정확한 마스킹)
    draw = ImageDraw.Draw(crop)
    for b in inline_boxes:
        bx1, by1, bx2, by2 = map(int, b)
        # 좌표를 crop 영역으로 변환
        mask_x1 = max(0, bx1 - x1)
        mask_y1 = max(0, by1 - y1)
        mask_x2 = min(x2 - x1, bx2 - x1)
        mask_y2 = min(y2 - y1, by2 - y1)
        if mask_x2 > mask_x1 and mask_y2 > mask_y1:
            draw.rectangle([mask_x1, mask_y1, mask_x2, mask_y2], fill=(255, 255, 255))

    # 텍스트 조각 (PaddleOCR 단일) - 품질 향상
    text_frags = []
    try:
        res_list = OCR.predict(np.array(crop))  # v3 API
        if res_list:
            R = res_list[0]

            # v3 결과 dict 꺼내기
            data = getattr(R, "json", None)
            # 문서에 따라 json이 바로 필드들을 담거나 {"res": {...}} 형태일 수 있어 둘 다 처리
            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            rec_texts  = (data or {}).get("rec_texts", [])    # 인식 텍스트들
            rec_scores = (data or {}).get("rec_scores", [])  # 각 텍스트 점수
            # 폴리곤: rec_polys가 있으면 그걸, 없으면 dt_polys 사용
            rec_polys  = (data or {}).get("rec_polys", (data or {}).get("dt_polys", []))

            for poly, txt, sc in zip(rec_polys, rec_texts, rec_scores):
                if float(sc) < conf_thres:
                    continue
                # poly는 crop 좌표계라서 원본 좌표로 보정
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                cx, cy = float(np.mean(xs)) + x1, float(np.mean(ys)) + y1
                s = _normalize_text(txt)
                if s and len(s.strip()) > 0:  # 빈 텍스트 제거
                    text_frags.append({'type': 'TXT', 'text': s, 'center': (cx, cy), 'score': float(sc)})
    except Exception as e:
        print(f"[PaddleOCR v3] 오류: {e}")

    # 수식 LaTeX 조각 (품질 향상)
    math_frags = []
    for fb in inline_boxes:
        latex = ""
        if TEX is not None:
            try:
                patch = image_pil.crop(tuple(map(int, fb))).convert("RGB")
                latex = TEX(patch).strip()
                # LaTeX 품질 검증
                if not latex or len(latex) < 2:
                    latex = "formula"
            except Exception as e:
                print(f"[pix2tex] 오류: {e}")
                latex = "formula"
        cx = (fb[0] + fb[2]) / 2.0; cy = (fb[1] + fb[3]) / 2.0
        math_frags.append({'type': 'TEX', 'latex': latex, 'center': (cx, cy)})

    frags = text_frags + math_frags
    if not frags:
        return ""

    # y → x 정렬로 라인 그룹핑 (더 정밀한 그룹핑)
    frags.sort(key=lambda o: (o['center'][1], o['center'][0]))
    lines = []; cur = [frags[0]]
    for f in frags[1:]:
        if abs(f['center'][1] - cur[-1]['center'][1]) <= line_eps:
            cur.append(f)
        else:
            lines.append(cur); cur = [f]
    lines.append(cur)

    out = []
    for ln in lines:
        ln.sort(key=lambda o: o['center'][0])
        parts = []
        for o in ln:
            if o['type'] == 'TXT':
                parts.append(o['text'])
            else:
                # 수식 앞뒤로 공백을 추가하여 텍스트와 분리
                latex_text = o['latex'] if o['latex'] else "formula"
                parts.append(f" ${latex_text}$ ")
        s = "".join([p for p in parts if p]).strip()
        if s and len(s) > 0:  # 빈 라인 제거
            out.append(s)
    return "\n".join(out)


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """
    "x1, y1, x2, y2" 형태(절대좌표)로 들어온 YOLO 결과를 파싱.
    """
    try:
        xs = [t for t in bbox_str.replace('"','').split(',')]
        xs = [float(s.strip()) for s in xs if s.strip() != ""]
        if len(xs) != 4:
            raise ValueError(f"bbox 형식 오류: {bbox_str}")
        x1, y1, x2, y2 = xs
        
        # 좌표 정렬 및 검증
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 최소 크기 보장
        if x2 - x1 < 1:
            x2 = x1 + 1
        if y2 - y1 < 1:
            y2 = y1 + 1
            
        return x1, y1, x2, y2
    except Exception as e:
        print(f"bbox 파싱 오류: {bbox_str} -> {e}")
        return 0.0, 0.0, 1.0, 1.0  # 기본값 반환

def box_center(b: Tuple[float,float,float,float]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def box_size(b: Tuple[float,float,float,float]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return (max(0.0, x2-x1), max(0.0, y2-y1))

def spanning_threshold(width: float) -> float:
    # 전폭 요소 기준: 문서 폭의 0.60 이상이면 전폭으로 간주
    return 0.60 * width

# -------------------------
# 2단 감지
# -------------------------
def detect_two_column(items: List[Dict]) -> Tuple[bool, float]:
    """
    items: {'idx': 원래 행 인덱스, 'bbox':(x1,y1,x2,y2), 'cat':str} 리스트
    return: (is_two_col, x_split)
      - is_two_col=True 이면 2단 문서로 판단
      - x_split은 좌/우 분할 기준 x (문서폭/2 근사)
    """
    if len(items) < 4:
        return False, None

    # 문서 폭/높이 추정(최댓값 기반)
    max_x2 = max(b['bbox'][2] for b in items)
    max_y2 = max(b['bbox'][3] for b in items)
    width, height = max_x2, max_y2
    if width <= 0:
        return False, None

    # 중앙 분할선 초기값
    x_mid = width / 2.0

    # 좌/우 컬럼 후보 분류 (중심점 기준)
    left = []
    right = []
    for it in items:
        cx, cy = box_center(it['bbox'])
        (left if cx < x_mid else right).append(it)

    # 한쪽이 너무 적으면 2단 아닐 가능성 큼
    if len(left) < 2 or len(right) < 2:
        return False, x_mid

    # y-짝짓기: 왼쪽박스 y1과 "비슷한 높이"의 오른쪽박스 y1이 일정 비율 이상 존재하면 2단
    tau = max(20.0, 0.02 * height)  # 높이 차 허용치
    pairs = 0
    used_r = set()
    r_y = np.array([r['bbox'][1] for r in right])
    for l in left:
        ly = l['bbox'][1]
        # 오른쪽 중 가장 가까운 y 찾기
        idx = int(np.argmin(np.abs(r_y - ly)))
        if abs(r_y[idx] - ly) <= tau and idx not in used_r:
            used_r.add(idx)
            pairs += 1

    # 컬럼 간 수평 분리도도 체크(좌 평균 cx < 우 평균 cx)
    mean_lx = np.mean([box_center(it['bbox'])[0] for it in left])
    mean_rx = np.mean([box_center(it['bbox'])[0] for it in right])
    separated = (mean_rx - mean_lx) >= (0.10 * width)

    # 매칭쌍이 일정 수 이상이면 2단으로 인정
    # - 최소 2쌍, 또는 양측 요소 수의 20% 이상
    min_pairs = max(2, int(0.2 * min(len(left), len(right))))
    is_two_col = separated and (pairs >= min_pairs)
    return is_two_col, x_mid

# -------------------------
# 전폭 요소로 섹션 분할
# -------------------------
def split_sections_by_fullwidth(items: List[Dict], doc_width: float) -> List[List[Dict]]:
    """
    전폭(가로폭 >= 0.60*doc_width) 요소를 섹션 경계로 삼아 위→아래로 페이지를 잘라
    각 섹션 내부를 반환. (경계 요소 자체는 '그 섹션'의 전폭 그룹으로 남김)
    """
    full_w = []
    normal = []
    for it in items:
        w, h = box_size(it['bbox'])
        (full_w if w >= spanning_threshold(doc_width) else normal).append(it)

    # y1 기준 정렬
    full_w.sort(key=lambda x: x['bbox'][1])
    normal.sort(key=lambda x: x['bbox'][1])

    if not full_w:
        return [items]  # 섹션 하나

    # 섹션 범위를 구성: (-inf, y_fw1), [y_fw1, y_fw2), ..., [y_last, +inf)
    breaks = [fw['bbox'][1] for fw in full_w]
    sections = []
    prev = -float('inf')
    # 전폭 요소도 같은 섹션에 포함시키기 위해 later merge
    for b in breaks + [float('inf')]:
        seg = []
        # 이번 구간에 속하는 normal 요소
        for it in normal:
            y = it['bbox'][1]
            if prev <= y < b:
                seg.append(it)
        # 구간 내 전폭 요소(경계 y==b인 요소 포함)
        fw_in_seg = [fw for fw in full_w if (prev <= fw['bbox'][1] < b) or (b == float('inf') and fw['bbox'][1] >= prev)]
        # 전폭 먼저, 그 다음 normal
        seg_sorted = sorted(fw_in_seg, key=lambda x: x['bbox'][1]) + sorted(seg, key=lambda x: x['bbox'][1])
        sections.append(seg_sorted)
        prev = b
    return sections

# -------------------------
# 섹션 내부 정렬 (2단 / 1단)
# -------------------------
def order_within_section(section_items: List[Dict], x_split: float, is_two_col: bool) -> List[Dict]:
    if not section_items:
        return []
    if not is_two_col:
        # 1단: y1 → x1
        return sorted(section_items, key=lambda x: (x['bbox'][1], x['bbox'][0]))

    # 2단: 좌→우로 나눠서 각자 y1로 정렬, 그리고 [좌 전체] → [우 전체]
    left, right = [], []
    for it in section_items:
        cx, cy = box_center(it['bbox'])
        (left if cx < x_split else right).append(it)

    left = sorted(left, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    right = sorted(right, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    return left + right

# -------------------------
# 메인: ID별 order 재계산
# -------------------------
def compute_orders_for_group(df_g: pd.DataFrame) -> pd.Series:
    """
    df_g: 하나의 ID에 해당하는 부분 DataFrame
    return: 같은 index 순서로 재배열된 order 시리즈
    """
    # 아이템 변환
    items = []
    for idx, row in df_g.iterrows():
        b = parse_bbox(row["bbox"])
        items.append({
            "idx": idx,
            "bbox": b,
            "cat": str(row.get("category_type", "")).lower()
        })

    if not items:
        return pd.Series([], dtype=int)

    # 문서 폭/높이 추정
    max_x2 = max(b['bbox'][2] for b in items)
    max_y2 = max(b['bbox'][3] for b in items)
    width, height = max_x2, max_y2

    # 2단 감지
    is_two_col, x_split = detect_two_column(items)
    if x_split is None:
        x_split = width / 2.0

    # 전폭 기준으로 섹션 분리 (제목/표 등 전폭 요소가 컬럼보다 앞서도록)
    sections = split_sections_by_fullwidth(items, width)

    # 섹션마다 정렬 후 합치기
    ordered_items: List[Dict] = []
    for sec in sections:
        ordered_items.extend(order_within_section(sec, x_split, is_two_col))

    # 최종 order 부여
    order_map: Dict[int, int] = {}
    for new_order, it in enumerate(ordered_items):
        order_map[it["idx"]] = new_order

    # 원래 인덱스 순서(df_g.index)에 맞춘 Series 반환
    return df_g.index.to_series().map(order_map).astype(int)

def reorder_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    haein.py의 predictions 리스트를 받아서 minhye의 ordering 로직을 적용
    """
    if not predictions:
        return predictions
    
    # DataFrame으로 변환
    df = pd.DataFrame(predictions)
    
    # 필요한 컬럼이 있는지 확인
    required_cols = ['ID', 'bbox', 'category_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for ordering: {missing_cols}")
        return predictions
    
    # ID별로 order 재계산
    new_orders = []
    for id_val, group in df.groupby("ID", sort=False):
        try:
            orders = compute_orders_for_group(group)
            new_orders.append(orders)
        except Exception as e:
            print(f"Warning: Failed to reorder group {id_val}: {e}")
            # 실패시 원래 순서 유지
            fallback_orders = pd.Series(range(len(group)), index=group.index)
            new_orders.append(fallback_orders)
    
    if new_orders:
        all_new_orders = pd.concat(new_orders).sort_index()
        # order 컬럼 업데이트
        df.loc[all_new_orders.index, "order"] = all_new_orders.values
    
    # DataFrame을 다시 dict 리스트로 변환
    return df.to_dict('records')


# -------------------------
# 단일 이미지 추론
# -------------------------
def inference_one_image(id_val, image_pil, original_size, target_size, conf_thres=None, iou_thres=None):
    """mAP 최적화된 단일 이미지 추론"""
    # mAP 최적화 파라미터 사용
    if conf_thres is None:
        conf_thres = DETECTION_CONF_THRESHOLD
    if iou_thres is None:
        iou_thres = DETECTION_IOU_THRESHOLD
    if model is None:
        raise RuntimeError("YOLO 모델이 초기화되지 않았습니다")

    # YOLO 입력을 위해 임시 저장
    import time, random
    os.makedirs(TEMP_IMG_DIR, exist_ok=True)
    temp_filename = f"_temp_{_safe_name(id_val)}_{int(time.time())}_{random.randint(1000,9999)}.png"
    temp_path = os.path.join(TEMP_IMG_DIR, temp_filename)
    image_pil.save(temp_path)

    try:
        print(f"[DEBUG] YOLO 추론 시작 - 파일: {temp_path}")
        print(f"[DEBUG] 모델 타입: {type(model)}")
        print(f"[DEBUG] 파라미터: imgsz={DETECTION_IMGSZ}, conf={conf_thres}, device={device}")
        
        # 60분 내 실행 + 0.38+ 정확도를 위한 추론
        det_res = model.predict(
            temp_path, 
            imgsz=DETECTION_IMGSZ,
            conf=conf_thres,
            iou=0.45,  # 적당한 IoU로 겹침 제거
            device=device,
            verbose=False,  # 속도 향상을 위해 False
            save=False,
            show=False,
            agnostic_nms=False,  # 클래스별 NMS 비활성화로 세밀한 탐지
            max_det=MAX_DET  # 적당한 최대 검출 개수
        )
        
        print(f"[DEBUG] 추론 결과 타입: {type(det_res)}, 길이: {len(det_res) if det_res else 'None'}")
        
        if not det_res or len(det_res) == 0:
            print("[DEBUG] 빈 결과 반환 - 더 낮은 임계값으로 재시도")
            # 낮은 임계값으로 재시도 (속도와 정확도 균형)
            det_res = model.predict(
                temp_path, 
                imgsz=640,  # 더 작은 이미지 크기
                conf=0.02,  # 더 낮은 신뢰도로 더 많은 검출
                iou=0.25,   # 더 낮은 IoU로 겹침 제거 강화
                device=device,
                verbose=False,  # 속도 향상
                agnostic_nms=False,
                max_det=3500  # 더 많은 최대 검출
            )
            print(f"[DEBUG] 재시도 결과: {len(det_res) if det_res else 'None'}")
            
        if not det_res or len(det_res) == 0:
            print("[WARNING] 객체가 검출되지 않았습니다")
            return [], []
            
        result = det_res[0]
        print(f"[DEBUG] result 타입: {type(result)}")
        print(f"[DEBUG] result.boxes: {result.boxes}")
        
        if result.boxes is None or len(result.boxes) == 0:
            print("[WARNING] 박스가 검출되지 않았습니다")
            return [], []
            
        inf_shape = result.orig_shape  # (h, w)
        print(f"원본 크기: {original_size}, 추론 크기: {(inf_shape[1], inf_shape[0])}")
        print(f"[DEBUG] 검출된 박스 수: {len(result.boxes)}")
        
    except Exception as e:
        print(f" YOLOv10 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        return [], []
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

    predictions, overlays = [], []
    order = 0

    try:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        names = result.names if hasattr(result, "names") else {}

        print(f"  YOLO 원시 결과: {len(boxes)}개 박스")
        print(f"  사용 가능한 클래스: {list(names.values()) if names else 'None'}")
        print(f"  LABEL_MAP: {LABEL_MAP}")

        layout_items = []
        for box, score, cls_idx in zip(boxes, scores, classes):
            label = names.get(int(cls_idx), None)
            if label is None or label not in LABEL_MAP:
                continue
            cls = LABEL_MAP[label]
            x1_o, y1_o, x2_o, y2_o = box.tolist()
            
            # bbox 정제 적용
            refined_box = refine_bbox([x1_o, y1_o, x2_o, y2_o], original_size)
            layout_items.append({'cls': cls, 'box': refined_box, 'score': float(score)})

        # 안정화(중복 제거/NMS/정렬)
        layout_items = post_process_layouts(layout_items, original_size)
        
        # title이 하나 나오면 나머지는 subtitle/text로 재분류
        layout_items = reclassify_after_title(layout_items)

        # 인라인 수식 배정
        inline_map = get_inline_map_no_touch(
            layouts=layout_items,
            image_pil=image_pil,
            conf=INLINE_CONF, imgsz=INLINE_IMGSZ,
            thr_inline=0.60, thr_block=0.50,
            cov_tie_eps=0.05, border_eps=5
        )

        for i, lt in enumerate(layout_items):
            cls = lt['cls']; box = lt['box']; score = lt['score']
            overlays.append((*box, cls, score))

            x1_t, y1_t, x2_t, y2_t = scale_bbox_to_target(box, original_size, target_size)

            if cls in ['title', 'subtitle', 'text']:
                inline_here = inline_map.get(i, [])
                merged_text = build_text_with_inline(
                    image_pil=image_pil,
                    region_box=box,
                    inline_boxes=inline_here,
                    conf_thres=0.03,  # 더 낮은 임계값으로 더 많은 텍스트 추출
                    line_eps=int(round(original_size[1] * 0.012)),  # 더 정밀한 라인 구분
                )
                final_text = post_process_text(merged_text, cls)
                adjusted_score = adjust_confidence_score(score, final_text, cls)
                predictions.append({
                    'ID': id_val,
                    'category_type': cls,
                    'confidence_score': float(adjusted_score),
                    'order': order,
                    'text': final_text,
                    'bbox': f'{x1_t}, {y1_t}, {x2_t}, {y2_t}'
                })
                order += 1

                for fb in inline_here:
                    overlays.append((*fb, 'inline_formula', 1.0))
            else:
                predictions.append({
                    'ID': id_val,
                    'category_type': cls,
                    'confidence_score': float(score),
                    'order': order,
                    'text': "",
                    'bbox': f'{x1_t}, {y1_t}, {x2_t}, {y2_t}'
                })
                order += 1

    except Exception as e:
        print(f" 예측 결과 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return [], []

    return predictions, overlays


# -------------------------
# 메인 추론 루프
# -------------------------
def inference(test_csv_path=TEST_CSV, output_csv_path=OUTPUT_CSV, save_visualization=True):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(TEMP_IMG_DIR, exist_ok=True)

    if save_visualization:
        os.makedirs(VIZ_DIR, exist_ok=True)
        print(f"시각화 결과는 {VIZ_DIR}에 저장됩니다.")

    # 모델 초기화
    initialize_yolo()
    initialize_ocr()
    initialize_inline_yolo()
    initialize_pix2tex()

    # CSV 로드
    csv_dir = os.path.dirname(test_csv_path)
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"테스트 CSV 파일을 찾을 수 없습니다: {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    required_cols = ['ID', 'path', 'width', 'height']
    missing_cols = [c for c in required_cols if c not in test_df.columns]
    if missing_cols:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing_cols}")

    all_preds = []
    page_images = {}  # 페이지별 이미지 저장 (ordering 시각화용)

    for idx, row in test_df.iterrows():
        id_val = row['ID']
        raw_path = row['path']
        file_path = os.path.normpath(os.path.join(csv_dir, raw_path))
        try:
            target_w = int(row['width']); target_h = int(row['height'])
        except (ValueError, TypeError) as e:
            print(f"잘못된 크기 값: {id_val}, {e}")
            continue

        print(f"\n처리 중 ({idx+1}/{len(test_df)}): {id_val}")
        print(f" 파일: {file_path}")
        print(f" 타깃 크기: {target_w}x{target_h}")

        if not os.path.exists(file_path):
            print(f"파일 없음: {file_path}")
            continue

        try:
            images = convert_to_images(file_path, TEMP_IMG_DIR)
            original_sizes = [img.size for img in images]
            if len(images) == 0:
                print("이미지 변환 결과 없음")
                continue

            # 여러 페이지 지원: ID에 페이지 번호 접미사
            for p, img in enumerate(images, start=1):
                page_id = id_val if len(images) == 1 else f"{id_val}_{p}"
                
                # 페이지 이미지 저장 (ordering 시각화용)
                page_images[page_id] = img
                
                preds, overlays = inference_one_image(page_id, img, original_sizes[p-1], (target_w, target_h))
                all_preds.extend(preds)

                # 기존 detection 시각화 (ordering 전)
                if save_visualization and overlays:
                    viz_filename = f"{_safe_name(page_id)}_detection.png"
                    viz_path = os.path.join(VIZ_DIR, viz_filename)
                    visualize_detections(img, overlays, viz_path)

            print(f"예측 완료: {len(images)}페이지, 누적 {len(all_preds)}개 객체")

        except Exception as e:
            print(f"처리 실패: {file_path} → {e}")
            import traceback
            traceback.print_exc()

        # 중간 temp 정리
        try:
            for temp_file in os.listdir(TEMP_IMG_DIR):
                if temp_file.startswith("_temp") and temp_file.endswith(".png"):
                    tfp = os.path.join(TEMP_IMG_DIR, temp_file)
                    if os.path.isfile(tfp):
                        os.remove(tfp)
        except Exception:
            pass

    # 결과 저장 및 Ordering 후 시각화
    if all_preds:
        print(f"\n[Ordering] 순서 정렬 적용 중... ({len(all_preds)}개 객체)")
        all_preds = reorder_predictions(all_preds)
        print("[Ordering] 2단 컬럼 기반 순서 정렬 완료")
        
        # Ordering이 반영된 시각화 생성
        if save_visualization and page_images:
            print("\n[Visualization] Ordering 반영 시각화 생성 중...")
            
            # ID별로 그룹화된 예측 결과
            ordered_by_page = {}
            for pred in all_preds:
                page_id = pred['ID']
                if page_id not in ordered_by_page:
                    ordered_by_page[page_id] = []
                ordered_by_page[page_id].append(pred)
            
            # 각 페이지별로 ordering 시각화 생성
            for page_id, page_preds in ordered_by_page.items():
                if page_id in page_images:
                    ordered_viz_filename = f"{_safe_name(page_id)}_ordered.png"
                    ordered_viz_path = os.path.join(VIZ_DIR, ordered_viz_filename)
                    
                    # 예측 결과를 overlay 형태로 변환 (order 포함)
                    ordered_overlays = []
                    for pred in page_preds:
                        try:
                            bbox_coords = [float(x.strip()) for x in pred['bbox'].split(',')]
                            if len(bbox_coords) != 4:
                                print(f"[Ordering Viz] 잘못된 bbox 형식: {pred['bbox']}")
                                continue
                                
                            x1, y1, x2, y2 = bbox_coords
                            
                            # 좌표 정렬 및 검증
                            x1, x2 = min(x1, x2), max(x1, x2)
                            y1, y2 = min(y1, y2), max(y1, y2)
                            
                            # 최소 크기 보장
                            if x2 - x1 < 1:
                                x2 = x1 + 1
                            if y2 - y1 < 1:
                                y2 = y1 + 1
                            
                            cat = pred.get('category_type', 'unknown')
                            score = pred.get('confidence_score', 0.0)
                            order = pred.get('order', -1)
                            
                            # 6개 요소 형태로 생성 (order 포함)
                            ordered_overlays.append((x1, y1, x2, y2, cat, score, order))
                        except Exception as e:
                            print(f"[Ordering Viz] 좌표 파싱 실패: {pred['bbox']}, {e}")
                            continue
                    
                    if ordered_overlays:
                        visualize_detections(page_images[page_id], ordered_overlays, ordered_viz_path)
            
            print(f"[Visualization] Ordering 시각화 완료: {len(ordered_by_page)}개 페이지")
        
        result_df = pd.DataFrame(all_preds)
        result_df.to_csv(output_csv_path, index=False, encoding='UTF-8-sig')
        print(f"\n저장 완료: {output_csv_path}")
        print(f"  총 {len(all_preds)}개 객체 검출")
        if save_visualization:
            print(f"기본 시각화 (Detection): {VIZ_DIR}/*_detection.png")
            print(f"순서 시각화 (Ordered): {VIZ_DIR}/*_ordered.png")
    else:
        print("\n검출된 객체가 없습니다.")
        empty_df = pd.DataFrame(columns=['ID', 'category_type', 'confidence_score', 'order', 'text', 'bbox'])
        empty_df.to_csv(output_csv_path, index=False, encoding='UTF-8-sig')


# -------------------------
# 엔트리포인트
# -------------------------
if __name__ == "__main__":
    print(f"Using device: {device}")
    try:
        inference(save_visualization=True)
        reorder_csv("./output/submission.csv", "./output/submission.csv")
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n 실행 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if os.path.exists(TEMP_IMG_DIR):
                for temp_file in os.listdir(TEMP_IMG_DIR):
                    if temp_file.startswith("_temp"):
                        temp_path = os.path.join(TEMP_IMG_DIR, temp_file)
                        if os.path.isfile(temp_path):
                            os.remove(temp_path)
                if not os.listdir(TEMP_IMG_DIR):
                    os.rmdir(TEMP_IMG_DIR)
        except Exception as e:
            print(f"정리 중 오류: {e}")