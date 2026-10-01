# CoDU — Complex Document Understanding

Document understanding code for the 2025 Samsung AI Challenge. The pipeline combines DocLayout-YOLO layout detection, PaddleOCR text recognition, Pix2Tex formula recognition, bounding-box post-processing, and reading-order reconstruction.

## Files

- `script.py`: document inference and CSV/visualization output.
- `model/order/order_again.py`: reading-order post-processing.
- `model/order/dacon_test.py`: local evaluation utilities.
- `model/paddleocr/`: OCR model configurations.

## Setup

Use Python 3.10+ in a GPU environment. The original requirements target PaddlePaddle GPU with CUDA 12.6; install a compatible PyTorch build for your environment. PDF/PPTX conversion also needs Poppler/LibreOffice, respectively.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pdf2image scikit-learn
```

## Model weights

Download the files below individually and save each at the specified path. Create missing folders; for PaddleOCR downloads, rename the file to the exact filename shown in **Save as**. The OCR configuration files are already included in this repository. Repository access is required while this project is private.

| Model | Download | Save as (relative to repository root) |
| --- | --- | --- |
| DocLayout-YOLO | [doclayout_yolo_docstructbench.pt](https://github.com/kbannie/CoDU/releases/download/original-archive/doclayout_yolo_docstructbench.pt) | `model/yolo/doclayout_yolo_docstructbench.pt` |
| Inline formula detection | [formula_yolo.pt](https://github.com/kbannie/CoDU/releases/download/original-archive/formula_yolo.pt) | `model/yolo/formula_yolo.pt` |
| PaddleOCR detection | [paddleocr_det_inference.pdiparams](https://github.com/kbannie/CoDU/releases/download/original-archive/paddleocr_det_inference.pdiparams) | `model/paddleocr/det/PP-OCRv5_server_det/inference.pdiparams` |
| PaddleOCR recognition | [paddleocr_rec_inference.pdiparams](https://github.com/kbannie/CoDU/releases/download/original-archive/paddleocr_rec_inference.pdiparams) | `model/paddleocr/rec/korean_PP-OCRv5_mobile_rec_infer/inference.pdiparams` |
| Pix2Tex | [weights.pth](https://github.com/kbannie/CoDU/releases/download/original-archive/weights.pth) | `model/pix2tex/weights.pth` |
| Pix2Tex image resizer | [image_resizer.pth](https://github.com/kbannie/CoDU/releases/download/original-archive/image_resizer.pth) | `model/pix2tex/image_resizer.pth` |

Pix2Tex weights enable formula transcription; the script can continue without Pix2Tex if its initialization fails.

<details>
<summary>Additional model files from the original archive (not loaded by default)</summary>

| Model | Download | Save as (relative to repository root) |
| --- | --- | --- |
| Additional YOLO checkpoint | [best.pt](https://github.com/kbannie/CoDU/releases/download/original-archive/best.pt) | `model/yolo/best.pt` |
| Additional YOLO checkpoint | [doclaynet_10.pt](https://github.com/kbannie/CoDU/releases/download/original-archive/doclaynet_10.pt) | `model/yolo/doclaynet_10.pt` |
| Additional YOLO checkpoint | [epoch_3.pt](https://github.com/kbannie/CoDU/releases/download/original-archive/epoch_3.pt) | `model/yolo/epoch_3.pt` |
| Additional YOLO checkpoint | [yolov12l-doclaynet.pt](https://github.com/kbannie/CoDU/releases/download/original-archive/yolov12l-doclaynet.pt) | `model/yolo/yolov12l-doclaynet.pt` |
| Additional recognition checkpoint | [korean_PP-OCRv5_mobile_rec_pretrained.pdparams](https://github.com/kbannie/CoDU/releases/download/original-archive/korean_PP-OCRv5_mobile_rec_pretrained.pdparams) | `model/paddleocr/rec/korean_PP-OCRv5_mobile_rec_infer/korean_PP-OCRv5_mobile_rec_pretrained.pdparams` |
| Orientation classifier parameters | [paddleocr_cls_inference.pdiparams](https://github.com/kbannie/CoDU/releases/download/original-archive/paddleocr_cls_inference.pdiparams) | `model/paddleocr/cls/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams` |
| Orientation classifier model | [paddleocr_cls_inference.pdmodel](https://github.com/kbannie/CoDU/releases/download/original-archive/paddleocr_cls_inference.pdmodel) | `model/paddleocr/cls/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel` |
| Orientation classifier metadata | [paddleocr_cls_inference.pdiparams.info](https://github.com/kbannie/CoDU/releases/download/original-archive/paddleocr_cls_inference.pdiparams.info) | `model/paddleocr/cls/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams.info` |

</details>

File checksums: [SHA256SUMS](https://github.com/kbannie/CoDU/releases/download/original-archive/SHA256SUMS). Datasets are not included.

## Run

Prepare `data/test.csv` with columns `ID,path,width,height`. Document paths are resolved relative to the CSV directory. Run from the repository root:

```bash
python script.py
```

Results are saved to `output/submission.csv` and `output/viz_45/`. Paths and inference settings can be adjusted near the top of `script.py`.

This repository contains inference code. The `train.py` referenced in the separate training guide is not included. Source syntax has been checked; full GPU inference has not been validated for this upload.
