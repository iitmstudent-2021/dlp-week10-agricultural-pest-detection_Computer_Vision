# Agricultural Pest Detection — DLP Week 10

## Competition

| | |
|---|---|
| **Platform** | Kaggle |
| **Competition** | [Object Detection Week 10 DLP](https://www.kaggle.com/competitions/object-detetction-week-10-dlp) |
| **Task** | Multi-class object detection — locate and classify agricultural pests |
| **Metric** | Mean Average Precision @ IoU=0.5 (mAP@0.5) |
| **Result** | **Public Leaderboard: Rank 6 · Score 0.6769** |

---

## Project Overview

This project builds a YOLOv8-based object detection pipeline to detect and classify **23 species of agricultural pests** from field images. The model outputs bounding boxes with class labels for every pest visible in a test image.

Agricultural pests cause massive crop losses globally. Automated detection enables precision treatment — reducing chemical use while protecting yield.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle Competition Data](https://www.kaggle.com/competitions/object-detetction-week-10-dlp/data) |
| **Training images** | 12,701 |
| **Test images** | 7,600 |
| **Total annotations** | 102,472 bounding boxes |
| **Image resolution** | 800 × 600 px |
| **Number of classes** | 23 |
| **Format** | JPG images + CSV annotations (xmin, ymin, xmax, ymax) |

### 23 Pest Classes

| # | Class | # | Class |
|---|---|---|---|
| 0 | AsiaticRiceBorer | 12 | RiceLeafRoller |
| 1 | BlackCutworm | 13 | RiceLeafhopper |
| 2 | BrownPlantHopper | 14 | RiceShellPest |
| 3 | CornBorer | 15 | RiceStemfly |
| 4 | GrainSpreaderThrips | 16 | RiceWaterWeevil |
| 5 | Grub | 17 | SmallBrownPlantHopper |
| 6 | LargeCutworm | 18 | WhiteBackedPlantHopper |
| 7 | MoleCricket | 19 | WhiteMarginedMoth |
| 8 | PaddyStemMaggot | 20 | Wireworm |
| 9 | RedSpider | 21 | YellowCutworm |
| 10 | RiceGallMidge | 22 | YellowRiceBorer |
| 11 | RiceLeafCaterpillar | | |

---

## Key EDA Findings

| Finding | Detail | Impact on Model |
|---|---|---|
| **Class imbalance** | RiceWaterWeevil (29,735 samples) vs WhiteMarginedMoth (54 samples) — **551× ratio** | Increased `cls` loss weight; focal loss via YOLO |
| **Small objects** | Mean bbox 34×34 px on 800px images (~4% of width) | Kept `imgsz=640` with mosaic augmentation |
| **7 rare classes** | CornBorer, LargeCutworm, MoleCricket, RiceGallMidge, Wireworm, WhiteMarginedMoth, YellowCutworm all < 500 samples | These had lowest AP50 (WhiteMarginedMoth AP=0.0) |
| **Dense images** | Max 202 objects/image; median 3 | Mosaic augmentation critical for dense scenes |
| **Uniform spatial distribution** | No strong positional bias in bbox centres | Standard anchor configuration sufficient |
| **Square aspect ratio** | Mean W/H ≈ 1.0 | No custom anchor tuning needed |

---

## Model & Pipeline

### Architecture
- **Model**: YOLOv8m (medium) — pretrained on COCO, fine-tuned on pest data
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Input size**: 640 × 640 px

### Training Configuration

```python
MODEL_NAME    = 'yolov8m.pt'
IMG_SIZE      = 640
EPOCHS        = 30
BATCH_SIZE    = 16
OPTIMIZER     = 'AdamW'
LR0           = 0.001
LRF           = 0.01
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3
PATIENCE      = 10       # early stopping
CLOSE_MOSAIC  = 10       # disable mosaic last 10 epochs
CLS           = 0.7      # boosted classification loss for imbalanced classes
LABEL_SMOOTH  = 0.1
CACHE         = False    # Kaggle disk limit ~19GB; cache needs 23GB
```

### Augmentation Strategy

| Augmentation | Value | Reason |
|---|---|---|
| Mosaic | 1.0 | Simulates dense multi-pest scenes |
| Mixup | 0.1 | Improves generalisation |
| HSV jitter | h=0.015, s=0.7, v=0.4 | Handles field lighting variation |
| Horizontal flip | 0.5 | Pests appear in any orientation |
| Scale | 0.5 | Handles varying camera distances |
| Copy-paste | 0.0 | Disabled — invalid for detection tasks |
| Close mosaic | last 10 epochs | Cleaner final fine-tuning |

### Inference

```python
CONF_THRESH = 0.05    # tuned from sweep; default 0.25 gave mAP=0.5691, 0.05 gave 0.6256
NMS_IOU     = 0.5
AUGMENT     = True    # Test-Time Augmentation (TTA)
INFER_BATCH = 32
```

---

## Results

### Validation Metrics (30 epochs, conf=0.05)

| Metric | Score |
|---|---|
| **mAP@0.5** | **0.6331** |
| mAP@0.5:0.95 | 0.3902 |
| Precision | 0.6315 |
| Recall | 0.6400 |

### Per-Class Performance (Worst 5)

| Class | AP@0.5 |
|---|---|
| WhiteMarginedMoth | 0.000 |
| GrainSpreaderThrips | 0.111 |
| Wireworm | 0.215 |
| RiceGallMidge | 0.448 |
| RiceLeafhopper | 0.458 |

### Leaderboard

| Leaderboard | Score | Rank |
|---|---|---|
| Public | 0.6769 | **6 / 140+** |
| Private | — | ~24 |

---

## Kaggle-Specific Engineering

| Problem | Solution |
|---|---|
| Browser tab OOM crash | `matplotlib.use('Agg')` + `plt.savefig()` + `plt.close()` — no inline rendering |
| Training log flood → OOM | `verbose=False` on all YOLO calls |
| Disk OOM (cache=True needs 23GB) | `cache=False` |
| Copying 12,701 images wastes disk | `os.symlink()` instead of `shutil.copy2()` |
| Dataset path varies by account | `KAGGLE_INPUT.rglob('train.csv')` — recursive path discovery |
| submission.csv not in output | Must use **Save & Run All (Commit)**, not interactive run |
| Kaggle rejects empty PredictionString | Fill with dummy `'RiceWaterWeevil 0.001 0 0 1 1'` for no-detection images |

---

## Submission Format

```
ImageID,PredictionString
0000003.jpg,AsiaticRiceBorer 0.8687 396 431 450 501 YellowRiceBorer 0.7882 277 186 315 210 ...
0000010.jpg,YellowRiceBorer 0.8293 357 342 384 383 ...
0000099.jpg,                          ← empty string NOT accepted; use dummy prediction
```

**Format per object**: `ClassName Confidence Xmin Ymin Xmax Ymax` (space-separated, absolute pixel coordinates)

---

## Key Learnings

1. **Confidence threshold matters more than expected** — default `conf=0.25` gave mAP=0.569; tuning to `conf=0.05` gave mAP=0.626 (+5.7% absolute gain, zero extra compute)

2. **Test-Time Augmentation (TTA)** — `augment=True` during inference gives a free ~1-2% mAP boost with no retraining

3. **`close_mosaic`** — disabling mosaic in the last N epochs lets the model fine-tune on clean images, improving final convergence

4. **Class imbalance is the hardest problem** — WhiteMarginedMoth (54 samples) scored AP=0.0; boosting `cls` loss weight helps but rare classes remain difficult without oversampling

5. **Kaggle output files only persist on committed runs** — interactive cell execution produces temporary outputs that vanish when the session ends; always use **Save & Run All**

6. **Symlinks over copies** — `os.symlink()` for dataset preparation is instant and uses zero extra disk vs `shutil.copy2()` which copies 12,701 images (~1.85 GB)

7. **`cache=False` on Kaggle** — disk cache requires ~23 GB but Kaggle provides ~19 GB free; always disable

8. **`verbose=False`** — YOLO's training output is enormous; leaving it on floods the browser and causes tab crashes on Kaggle

---

## Repository Structure

```
Week-10/Assignment/
├── pest_detection.ipynb      ← main notebook (train + inference + CSV)
├── submission_final.csv      ← final submission file (7,600 rows)
├── README.md                 ← this file
└── fix_submission.py         ← utility: fixes null PredictionString rows
```

---

## How to Reproduce

1. Go to the [competition page](https://www.kaggle.com/competitions/object-detetction-week-10-dlp) and join
2. Upload `pest_detection.ipynb` to a new Kaggle notebook
3. Add the competition dataset as input
4. Set accelerator to **GPU T4 x2**
5. Click **Save Version → Save & Run All (Commit)**
6. After ~3 hours, go to Output tab → download `submission.csv`
7. Submit to competition leaderboard

---

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 Training Parameters](https://docs.ultralytics.com/modes/train/)
- [Competition Page](https://www.kaggle.com/competitions/object-detetction-week-10-dlp)
- IIT Madras BS in Data Science — Deep Learning Practice (DLP), Week 10 Assignment, JAN 2026 Term
