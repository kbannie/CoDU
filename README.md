# CoDU — Complex Document Understanding

Document understanding code for the 2025 Samsung AI Challenge. The pipeline combines DocLayout-YOLO layout detection, PaddleOCR text recognition, Pix2Tex formula recognition, bounding-box post-processing, and reading-order reconstruction.

## Files

- `script.py`: document inference and CSV/visualization output.
- `model/order/order_again.py`: reading-order post-processing.
- `model/order/dacon_test.py`: local evaluation utilities.

## Setup

Use Python 3.10+ in a GPU environment. The original requirements target PaddlePaddle GPU with CUDA 12.6; install a compatible PyTorch build for your environment. PDF/PPTX conversion also needs Poppler/LibreOffice, respectively.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pdf2image scikit-learn
```

Model assets and datasets are not included in this repository. Restore the model folders from the original code archive, keeping these paths:

```text
model/yolo/doclayout_yolo_docstructbench.pt
model/yolo/formula_yolo.pt
model/paddleocr/det/PP-OCRv5_server_det/
model/paddleocr/rec/korean_PP-OCRv5_mobile_rec_infer/
model/pix2tex/weights.pth
model/pix2tex/image_resizer.pth
```

## Run

Prepare `data/test.csv` with columns `ID,path,width,height`. Document paths are resolved relative to the CSV directory. Run from the repository root:

```bash
python script.py
```

Results are saved to `output/submission.csv` and `output/viz_45/`. Paths and inference settings can be adjusted near the top of `script.py`.

This archive contains inference code. The `train.py` referenced in the separate training guide is not included. Source syntax has been checked; full GPU inference has not been validated for this upload.
