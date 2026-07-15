"""
generate_report.py
-------------------
Builds a PDF training report from a YOLO (Ultralytics) run directory that has
the following subfolder layout (as in the user's project):

    run_dir/
        args.yaml
        results.csv
        results.png
        weights/
            best.pt
            last.pt
        Box Metrices Curves/
            BoxF1_curve.png
            BoxP_curve.png
            BoxPR_curve.png
            BoxR_curve.png
        Confusion Matrix/
            confusion_matrix.png
            confusion_matrix_normalized.png
        Test Outputs Predictions/
            train_76_full_output.png
            train_81_0_Upper Right.png
            train_81_1_Lower Right.png
            ...
        Train Batches/
            train_batch0.jpg
            train_batch1.jpg
            ...
        Val Batches/
            val_batch0_labels.jpg
            val_batch0_pred.jpg
            ...

Usage:
    python generate_report.py --run_dir "Runs/Stage 1" --output "Runs/Stage1_Report.pdf" --title "Stage 1 - Quadrant Detector"
"""

import os
import re
import glob
import argparse
import yaml
import pandas as pd
from collections import defaultdict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage
)


# ----------------------------------------------------------------------
# Subfolder names | edit here if your folder names ever change
# ----------------------------------------------------------------------

SUBFOLDER_BOX_CURVES = "Box Metrices Curves"
SUBFOLDER_CONFUSION_MATRIX = "Confusion Matrix"
SUBFOLDER_TEST_PREDICTIONS = "Test Outputs Predictions"
SUBFOLDER_TRAIN_BATCHES = "Train Batches"
SUBFOLDER_VAL_BATCHES = "Val Batches"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_args(run_dir):
    path = os.path.join(run_dir, "args.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_results(run_dir):
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_first(run_dir, subfolder, filenames):
    """Return the first existing file (exact filename match) inside run_dir/subfolder."""
    for name in filenames:
        path = os.path.join(run_dir, subfolder, name) if subfolder else os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return None


def find_all(run_dir, subfolder, patterns, limit=None):
    """Glob patterns inside run_dir/subfolder."""
    base = os.path.join(run_dir, subfolder) if subfolder else run_dir
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(os.path.join(base, pattern))))
    if limit:
        files = files[:limit]
    return files


