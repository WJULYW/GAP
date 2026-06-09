# GAP Release Code

This directory contains a cleaned release version of the GAP implementation for:

**Align the GAP: Prior-based Unified Multi-Task Remote Physiological Measurement Framework for Domain Generalization and Personalization**

The file layout follows the original research code:

- `Model.py`: GAP model (`My_model`) with shared ResNet-18 encoder, task gates, four task heads, and person branch.
- `MyDataset.py`: STMap dataset and prior-based augmentation.
- `MyLoss.py`: GAP losses: LSSA, LSDA, LFC, LTIC/LTC, LPE, LP, and LMT.
- `train.py`: GAP-G / MSSDG training entry.
- `tta_train.py`: GAP-P / TTPA adaptation entry.
- `TTA_methods.py`: test-time adaptation wrapper.
- `utils.py`: metrics, signal utilities, and argument parser.

## Environment

The code is intentionally kept free of hard dependencies from the old experimental scripts.

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

## Smoke Test

Run a random-data check before launching experiments:

```bash
python tests/smoke_test.py
python train.py --smoke
python tta_train.py --smoke
```

## Data Layout

The default data root is `./data/STMap`. Override it with `--data-root`.

Expected layout:

```text
DATA_ROOT/
  PURE/
    subject_or_video/
      STMap/STMap.png
      Label/...
  STMap_Index/
    PURE/
      subject_or_video_0.mat
```

Build indexes from STMap folders with:

```bash
python train.py --data-root /path/to/STMap --tgt PURE --reData 1
```

## MSSDG Training

```bash
python train.py --data-root /path/to/STMap --tgt PURE --batch-size 100 --max_iter 20000 --learning-rate 1e-5
```

The target dataset is held out. All other configured datasets are used as source domains.

## TTPA Adaptation

```bash
python tta_train.py --data-root /path/to/STMap --tgt PURE --checkpoint Result_Model/GAP_G_PURE.pth --batch-size 1 --learning-rate 1e-5
```

TTPA resets the model to the same GAP-G checkpoint for each target subject and adapts samples in chronological order.
