# GAP

This is the official implementation for:

**Align the GAP: Prior-based Unified Multi-Task Remote Physiological Measurement Framework for Domain Generalization and Personalization** (IJCV2026)

The file layout follows the original research code:

- `Model.py`: GAP model (`My_model`) with shared ResNet-18 encoder, task gates, four task heads, and person branch.
- `MyDataset.py`: STMap dataset and prior-based augmentation.
- `MyLoss.py`: GAP losses: LSSA, LSDA, LFC, LTIC/LTC, LPE, LP, and LMT.
- `train.py`: GAP-G / MSSDG training entry.
- `tta_train.py`: GAP-P / TTPA adaptation entry.
- `TTA_methods.py`: test-time adaptation wrapper.
- `utils.py`: metrics, signal utilities, and argument parser.


## Data

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

For HMPC-D dataset, please refer to [PhysDrive](https://github.com/WJULYW/PhysDrive-Dataset); For HCW dataset, please refer to [HCW](https://github.com/WJULYW/PhysMLE); For other datasets, as well as the STMap generation, please refer to https://github.com/WJULYW/HSRD.

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

## Citation
Title: [Align the gap: Prior-based unified multi-task remote physiological measurement framework for domain generalization and personalization](https://link.springer.com/article/10.1007/s11263-025-02707-w)

Jiyao Wang, Xiao Yang, Hao Lu, Dengbo He, Kaishun Wu, IJCV, 2026  
```
@article{wang2026align,
  title={Align the gap: Prior-based unified multi-task remote physiological measurement framework for domain generalization and personalization},
  author={Wang, Jiyao and Yang, Xiao and Lu, Hao and He, Dengbo and Wu, Kaishun},
  journal={International Journal of Computer Vision},
  volume={134},
  number={5},
  pages={199},
  year={2026},
  publisher={Springer}
}
```
