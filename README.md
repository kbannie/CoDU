# CoDU — Complex Document Understanding

Document understanding code for the 2025 Samsung AI Challenge. The pipeline combines DocLayout-YOLO layout detection, PaddleOCR text recognition, Pix2Tex formula recognition, bounding-box post-processing, and reading-order reconstruction.

**[Download the complete original ZIP, including all model weights (737 MiB)](https://github.com/kbannie/CoDU/releases/tag/original-archive).** The repository contains source code and model configurations; the release preserves every file from the original archive. Repository access is required while this project is private.

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

Download `haeing2ordering.zip` from the release above and extract it at the repository root to restore all model assets. With GitHub CLI:

```bash
gh release download original-archive --repo kbannie/CoDU --pattern 'haeing2ordering.zip*'
shasum -a 256 -c haeing2ordering.zip.sha256
unzip -n haeing2ordering.zip
```

The archive includes the YOLO, PaddleOCR, and Pix2Tex model folders. Datasets are not included.

## Run

Prepare `data/test.csv` with columns `ID,path,width,height`. Document paths are resolved relative to the CSV directory. Run from the repository root:

```bash
python script.py
```

Results are saved to `output/submission.csv` and `output/viz_45/`. Paths and inference settings can be adjusted near the top of `script.py`.

This archive contains inference code. The `train.py` referenced in the separate training guide is not included. Source syntax has been checked; full GPU inference has not been validated for this upload.
