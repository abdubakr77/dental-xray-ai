# Dental AI: Panoramic X-Ray Intelligence System

An end-to-end computer vision system for panoramic dental X-rays: a trained
multi-stage detection and classification pipeline, deployed as a full
interactive web application.

> AI-generated results are for research / decision-support purposes only and
> are not a substitute for professional dental diagnosis. All findings must
> be confirmed by a qualified dentist.

---

## Table of Contents

- [Overview](#overview)
- [The Pipeline](#the-pipeline)
- [Application Features](#application-features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Model Training Notes](#model-training-notes)
- [Known Limitations](#known-limitations)
- [Tech Stack](#tech-stack)

---

## Overview

The system takes a full panoramic dental X-ray and runs it through five
sequential stages: locating the four dental quadrants, detecting and
numbering every individual tooth, classifying each tooth as healthy or
unhealthy, identifying the disease type for unhealthy teeth, and, for teeth
diagnosed with caries, classifying severity as caries or deep caries.

Two YOLO models handle the detection stages (quadrants, then teeth within
each quadrant). Three Swin Transformer classifiers handle health status,
disease type, and caries severity. All five stages are wired into one
pipeline (`inference_pipeline.py`), with a full Streamlit interface built
around it rather than leaving it as a notebook.

## The Pipeline

```
Full Panoramic X-Ray
        │
        ▼
   Quadrant Detection  (YOLO)
        │
   ┌────┴────┬────────┬────────┐
 Upper Right  Upper Left  Lower Left  Lower Right
        │
        ▼
   Tooth Detection & Numbering  (YOLO, per quadrant)
   → each tooth gets a class ID 0-7, local to its own quadrant
        │
        ▼
   Health Classification  (Swin Transformer)
   → Healthy  |  Disease Found
        │
        ▼ (unhealthy teeth only)
   Disease Type Classification  (Swin Transformer)
   → Impacted  |  Caries  |  Periapical
        │
        ▼ (Caries teeth only)
   Caries Severity Classification  (Swin Transformer)
   → Caries  |  Deep Caries
```

**Design decisions worth noting:**

- Tooth numbering is a class ID from **0 to 7, local to each quadrant**, not
  a single global numbering scheme across all 32 possible teeth. This
  matches the model's own output and is preserved unchanged everywhere in
  the app, in reports, and in exports.
- During detection, all teeth are treated as a single "Tooth" class; the
  per-quadrant numbering is computed afterward via spatial ordering.
- A tooth's pipeline stops early when the result doesn't require the next
  stage: a Healthy tooth never goes through disease classification, and a
  non-Caries diagnosis (Impacted, Periapical) never goes through severity
  classification.
- Wherever the model produces a full probability distribution, the app
  surfaces every class's probability, not just the top prediction, so close
  calls between two possibilities are visible rather than hidden behind a
  single confidence number.

## Application Features

Built around the trained pipeline as a five-page Streamlit app:

**Home**: Landing page with a pipeline overview and a random sample-image
gallery to jump straight into an analysis.

**Analysis**: The core experience: upload or select a panoramic X-ray and
watch the pipeline run stage by stage. All four quadrants are detected and
classified in parallel (not one after another), tooth detection boxes
appear across all quadrants simultaneously round by round, and each tooth's
result card updates in place as classification resolves, never replaced by
an unrelated element. The final summary includes per-quadrant statistics,
disease distribution, the fully annotated X-ray, and a plain-language
warning when two predictions were close enough to be worth a second look.
Every completed analysis is saved automatically.

**Individual Models**: Run any single model in isolation (just quadrant
detection, just the disease classifier, etc.) with input-type validation, so
a full panoramic image can't be fed to a model expecting a tooth crop by
mistake. Useful for debugging or demonstrating one stage at a time.

**Reports / History**: Every saved analysis, browsable in detail with
per-quadrant and per-tooth breakdowns, exportable as JSON, CSV, PDF, or the
final annotated image.

**Dashboard**: Aggregate statistics across every saved analysis: totals,
disease distribution, most frequent finding, average inference time,
quadrant-level trends, and a confidence trend over time.

**Help / About**: Full usage documentation, FAQ, troubleshooting, and the
actual measured accuracy of each model.

**Debug mode** (sidebar toggle): Surfaces raw coordinates, confidence
values, device info, and the pipeline's internal warnings, kept separate
from the normal polished view.

## Project Structure

```
<PROJECT_ROOT>/
├── configs/
|   ├── stage1.yaml
|   ├── stage2.yaml
|   ├── stage2_continued.yaml
|   ├── stage3.yaml
│   └── trained_models.yaml      # model paths & class counts
├── Data/
│   ├── RAW DATASETS.md
│   ├── Processed/
│   │   ├── Stage 1 (Quadrant Detection)/
│   │   │   ├── data.yaml
│   │   │   ├── main_df (before splitted).pkl
│   │   │   └── ...
│   │   ├── Stage 2 (Enumeration Detection)/
│   │   │   └── ...
│   │   └── Stage 3/
│   │       └── ...
│   └── ...
├── Notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_stage1_quadrant_detection.ipynb
│   └── ...
├── Runs/
│   ├── Data Visualizations/
│   │   ├── Annotation Analysis/
│   │   ├── Boxes Visualizations/
│   │   ├── Dataset Overview/
│   │   └── Disease Analysis/
│   ├── Stage 1/
│   │   └── ...
│   ├── Stage 2/
│   │   └── ...
│   └── Stage 3/
│       └── ...
└── src/
    ├── app.py                   # entry point, navigation
    ├── inference_pipeline.py    # the trained pipeline, source of truth
    ├── model_utils.py           # model loading, detection/classification
    ├── utils.py                 # data prep, splitting, augmentation
    ├── vis.py                   # matplotlib/cv2 visualization helpers
    ├── animation.py             # CSS/SVG animation primitives for the UI
    ├── core/
    │   ├── config.py             # class names, stage metadata, paths
    │   ├── model_registry.py     # cached model loading, device info
    │   ├── single_model_runner.py# per-stage wrappers for isolated runs
    │   └── errors.py             # user-facing error handling
    ├── storage/
    │   ├── reports.py            # save/list/load analysis reports
    │   └── export.py             # CSV / PDF export
    ├── ui/
    │   ├── home.py, analysis.py, models_playground.py,
    │   ├── reports_page.py, dashboard.py, help.py, about.py
    │   └── components.py         # shared widgets (probability bars, etc.)
    ├── sample_images/            # sample X-rays for the Home/Analysis gallery
    ├── reports/                  # saved analysis reports (generated)
    └── temp_uploads/             # temporary upload storage (generated)
```

## Setup

```bash
git clone <repo-url>
cd dental-xray-ai
pip install -r requirements.txt
```

Place your trained model weights and `configs/trained_models.yaml` (one
level above `src/`). See [Configuration](#configuration) for the expected
format. Model weight files are not committed to this repository; host them
separately (e.g. Hugging Face Hub) if deploying.

## Running the App

Run from inside `src/`: the model config's own relative paths (inside the
YAML) are resolved against the working directory the app is launched from:

```bash
cd src
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud. A few things that mattered getting
there:

- `opencv-python` needs `libGL.so.1`, which isn't present on Streamlit
  Cloud's minimal Linux environment; use `opencv-python-headless` instead
  (no GUI dependencies, same API).
- The cloud environment is CPU-only; `load_recommended_models()` falls back
  to CPU automatically when CUDA isn't available.
- Model weight files are not stored in the git repository (GitHub rejects
  anything over 100MB, and it's the wrong place for them regardless); fetch
  them from external storage at startup instead.

## Configuration

`configs/trained_models.yaml` (not included, provide your own):

```yaml
final_recommended_models:
  quadrant_model: <path to quadrant detector weights>
  enumeration_model: <path to tooth enumeration weights>
  teeth_status_model: <path to health classifier weights>
  disease_model: <path to disease classifier weights>
  caries_status_model: <path to caries severity classifier weights>

quadrant_nc: 4
enumeration_nc: 8
teeth_status_nc: 2
disease_nc: 3
caries_status_nc: 2
```

Override the config path with the `DENTAL_AI_MODEL_CONFIG` environment
variable, and the reports directory with `DENTAL_AI_REPORTS_DIR`.

## Model Training Notes

- Trained on the **DENTEX 2023** dataset.
- Hierarchical YOLO detection: quadrant detection → tooth enumeration,
  followed by three Swin Transformer classification stages.
- Dataset parsing handles the DENTEX JSON annotation structure across its
  three annotation subfolder types.
- Ground-truth disease annotations were checked for near-duplicate boxes on
  the same tooth (same class, near-identical coordinates from repeated
  export runs) and deduplicated by IoU before training; genuinely distinct
  multi-disease annotations on the same tooth (different box sizes) were
  left untouched.
- Augmentation via Albumentations, with YOLO's own built-in augmentations
  explicitly disabled to avoid double-augmenting. Detail-destroying
  transforms (blur, sharpen, noise, grid distortion) are grouped so at most
  one applies per image rather than stacking independently, and rotation /
  zoom use reflect padding instead of a black border to avoid introducing
  edge artifacts with no anatomical meaning.
- Class imbalance in the disease classifiers is handled with
  augmentation-based oversampling to match the majority class count exactly
  (not exceeding it), combined with class-weighted loss. Pixel-space SMOTE
  was evaluated and dropped: interpolating raw X-ray pixels produced
  unrealistic blended images rather than plausible teeth.
- An external dental X-ray dataset was evaluated as a way to grow the
  minority disease classes; it was not merged in, since its bounding box
  annotations were inconsistently aligned with the actual finding location
  compared to DENTEX.
- Train/val/test split via `MultilabelStratifiedShuffleSplit`, given the
  multi-label nature of per-tooth disease annotations.
- ViT-B/16 was the first classifier backbone tried; it trained unstably
  even with a learning-rate warmup, and fine-tuning collapsed sharply when
  unfreezing the backbone mid-training (the optimizer has no momentum
  history for newly-unfrozen parameters, so they get optimized as if from
  scratch at the same learning rate as the already-converged head). Swin
  Transformer proved more stable under the same staged fine-tuning approach
  (frozen backbone warmup via `SequentialLR` combining linear warmup and
  cosine annealing, then a sharply reduced learning rate on unfreezing) and
  is the architecture used in the deployed models.
- Developed locally on an RTX 2060 Super (8GB VRAM) and on Kaggle for
  publishing/sharing.

## Known Limitations

- **Rare classes**: Periapical has limited test samples; treat it as a
  screening signal rather than a definitive result. See the About page for
  the actual measured precision/recall per class.
- **Error propagation**: an early-stage mistake (e.g. a missed tooth
  detection) carries forward; later stages can't correct it.
- **Caries vs. deep caries**: the boundary between severity levels is
  continuous in reality, not a hard line; this stage has the lowest
  measured accuracy of the three classifiers.
- **CPU inference**: on deployments without a GPU, analysis is
  meaningfully slower than the numbers shown in local development.

## Tech Stack

Python · PyTorch · torchvision (Swin Transformer) · Ultralytics YOLO ·
OpenCV · Streamlit · pandas · matplotlib/seaborn · scikit-learn