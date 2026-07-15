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


def safe_metric(row, col_candidates, fmt="{:.4f}"):
    for col in col_candidates:
        if col in row.index:
            try:
                return fmt.format(row[col])
            except Exception:
                return str(row[col])
    return "N/A"


def img_flowable(path, max_width, max_height=None):
    if not path or not os.path.exists(path):
        return None
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = max_width / float(w)
    new_w = max_width
    new_h = h * ratio
    if max_height and new_h > max_height:
        ratio = max_height / float(h)
        new_h = max_height
        new_w = w * ratio
    return RLImage(path, width=new_w, height=new_h)


def parse_test_prediction_files(files):
    """
    Groups test prediction files by base image ID.
    Handles:
      train_76_full_output.png
      train_81_0_Upper Right.png
      train_81_1_Lower Right.png
      ...
    """
    grouped = defaultdict(lambda: {"full": None, "quadrants": []})

    full_pattern = re.compile(r"^(train_\d+)_full_output\.(png|jpg)$", re.IGNORECASE)
    quad_pattern = re.compile(r"^(train_\d+)_(\d+)_(.+)\.(png|jpg)$", re.IGNORECASE)

    for path in files:
        fname = os.path.basename(path)

        m_full = full_pattern.match(fname)
        if m_full:
            grouped[m_full.group(1)]["full"] = path
            continue

        m_quad = quad_pattern.match(fname)
        if m_quad:
            base_id = m_quad.group(1)
            quad_idx = int(m_quad.group(2))
            quad_name = m_quad.group(3)
            grouped[base_id]["quadrants"].append((quad_idx, quad_name, path))

    for base_id in grouped:
        grouped[base_id]["quadrants"].sort(key=lambda x: x[0])

    return grouped


# ----------------------------------------------------------------------
# Report builder
# ----------------------------------------------------------------------

def generate_report(run_dir, output_path, title="YOLO Training Report"):

    if not output_path.lower().endswith(".pdf"):
        raise ValueError(f"output_path must end with '.pdf', got: {output_path}")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=15, spaceBefore=18,
                               spaceAfter=8, textColor=colors.HexColor("#1a1a1a"),
                               fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="SubText", fontSize=9.5, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="Caption", fontSize=8.5, textColor=colors.HexColor("#666666"),
                               spaceAfter=10, alignment=1))
    styles.add(ParagraphStyle(name="ImageID", fontSize=11, spaceBefore=10, spaceAfter=4,
                               fontName="Helvetica-Bold"))

    story = []
    page_width = A4[0] - 2 * 2 * cm

    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Run directory: {run_dir}", styles["SubText"]))
    story.append(Spacer(1, 16))

    # ---------------- 1. Configuration ----------------
    args = load_args(run_dir)
    if args:
        story.append(Paragraph("Configuration", styles["SectionHeader"]))
        config_keys = ["model", "epochs", "imgsz", "batch", "optimizer",
                       "lr0", "lrf", "patience", "device", "workers", "auto_augment"]
        rows = [["Parameter", "Value"]]
        for k in config_keys:
            if k in args:
                rows.append([k, str(args[k])])
        table = Table(rows, colWidths=[6 * cm, 8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

    # ---------------- 2. Training Summary ----------------
    results_df = load_results(run_dir)
    if results_df is not None and len(results_df) > 0:
        story.append(Paragraph("Training Summary", styles["SectionHeader"]))

        map50_col_candidates = ["metrics/mAP50(B)", "metrics/mAP50"]
        map5095_col_candidates = ["metrics/mAP50-95(B)", "metrics/mAP50-95"]
        precision_col_candidates = ["metrics/precision(B)", "metrics/precision"]
        recall_col_candidates = ["metrics/recall(B)", "metrics/recall"]

        best_map_col = next((c for c in map50_col_candidates if c in results_df.columns), None)
        best_row = results_df.loc[results_df[best_map_col].idxmax()] if best_map_col else results_df.iloc[-1]

        total_epochs = int(results_df["epoch"].iloc[-1]) + 1 if "epoch" in results_df.columns else len(results_df)
        best_epoch = int(best_row["epoch"]) + 1 if "epoch" in best_row.index else "N/A"

        summary_rows = [
            ["Metric", "Value"],
            ["Total epochs run", str(total_epochs)],
            ["Best epoch", str(best_epoch)],
            ["mAP50 (best)", safe_metric(best_row, map50_col_candidates)],
            ["mAP50-95 (best)", safe_metric(best_row, map5095_col_candidates)],
            ["Precision (best)", safe_metric(best_row, precision_col_candidates)],
            ["Recall (best)", safe_metric(best_row, recall_col_candidates)],
        ]
        table = Table(summary_rows, colWidths=[6 * cm, 8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

        img = img_flowable(os.path.join(run_dir, "results.png"), max_width=page_width, max_height=9 * cm)
        if img:
            story.append(img)
            story.append(Paragraph("Figure: loss and metric curves over training epochs.", styles["Caption"]))

    story.append(PageBreak())

    # ---------------- 3. Confusion Matrix ----------------
    story.append(Paragraph("Per-Class Performance", styles["SectionHeader"]))
    cm_path = find_first(run_dir, SUBFOLDER_CONFUSION_MATRIX,
                          ["confusion_matrix.png", "confusion_matrix_normalized.png"])
    img = img_flowable(cm_path, max_width=page_width, max_height=11 * cm)
    if img:
        story.append(img)
        story.append(Paragraph("Figure: confusion matrix on the validation set.", styles["Caption"]))
    else:
        story.append(Paragraph("No confusion matrix found in run directory.", styles["SubText"]))

    story.append(PageBreak())

    # ---------------- 4. Box Metric Curves ----------------
    story.append(Paragraph("Precision / Recall / F1 Curves", styles["SectionHeader"]))
    curve_files = [f for f in [
        find_first(run_dir, SUBFOLDER_BOX_CURVES, ["BoxP_curve.png"]),
        find_first(run_dir, SUBFOLDER_BOX_CURVES, ["BoxR_curve.png"]),
        find_first(run_dir, SUBFOLDER_BOX_CURVES, ["BoxF1_curve.png"]),
        find_first(run_dir, SUBFOLDER_BOX_CURVES, ["BoxPR_curve.png"]),
    ] if f]
    for path in curve_files:
        img = img_flowable(path, max_width=page_width * 0.85, max_height=8 * cm)
        if img:
            story.append(img)
            story.append(Paragraph(os.path.basename(path), styles["Caption"]))
    if not curve_files:
        story.append(Paragraph("No box metric curve images found.", styles["SubText"]))

    story.append(PageBreak())

    # ---------------- 5. Train / Val Batches ----------------
    story.append(Paragraph("Sample Training Batches", styles["SectionHeader"]))
    train_batches = find_all(run_dir, SUBFOLDER_TRAIN_BATCHES,
                              ["train_batch*.jpg", "train_batch*.png"], limit=2)
    for path in train_batches:
        img = img_flowable(path, max_width=page_width, max_height=9 * cm)
        if img:
            story.append(img)
            story.append(Paragraph(os.path.basename(path), styles["Caption"]))
    if not train_batches:
        story.append(Paragraph("No train batch samples found.", styles["SubText"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Sample Validation Batches", styles["SectionHeader"]))
    val_batches = find_all(run_dir, SUBFOLDER_VAL_BATCHES,
                            ["val_batch*_pred.jpg", "val_batch*_pred.png"], limit=2)
    for path in val_batches:
        img = img_flowable(path, max_width=page_width, max_height=9 * cm)
        if img:
            story.append(img)
            story.append(Paragraph(os.path.basename(path), styles["Caption"]))
    if not val_batches:
        story.append(Paragraph("No validation batch samples found.", styles["SubText"]))

    story.append(PageBreak())

    # ---------------- 6. Test Set Predictions ----------------
    story.append(Paragraph("Test Set Predictions", styles["SectionHeader"]))
    story.append(Paragraph(
        "Qualitative predictions on held-out images not used in train/val "
        "(disease-only annotated images kept aside for visual inspection).",
        styles["SubText"]))
    story.append(Spacer(1, 8))

    all_pngs = find_all(run_dir, SUBFOLDER_TEST_PREDICTIONS, ["train_*.png", "train_*.jpg"])
    test_pred_files = [
        f for f in all_pngs
        if "_full_output" in os.path.basename(f) or re.search(r"train_\d+_\d+_", os.path.basename(f))
    ]

    if test_pred_files:
        grouped = parse_test_prediction_files(test_pred_files)

        def sort_key(base_id):
            m = re.search(r"\d+", base_id)
            return int(m.group()) if m else 0

        for base_id in sorted(grouped.keys(), key=sort_key):
            entry = grouped[base_id]
            story.append(Paragraph(f"Image: {base_id}", styles["ImageID"]))

            if entry["full"]:
                img = img_flowable(entry["full"], max_width=page_width, max_height=9 * cm)
                if img:
                    story.append(img)
                    story.append(Paragraph("Full pipeline output", styles["Caption"]))

            if entry["quadrants"]:
                story.append(Paragraph("Quadrant-level predictions:", styles["SubText"]))
                row = []
                pending_rows = []
                for idx, quad_name, path in entry["quadrants"]:
                    cell_img = img_flowable(path, max_width=page_width / 2 - 0.5 * cm, max_height=6 * cm)
                    row.append([cell_img, Paragraph(quad_name, styles["Caption"])] if cell_img else [Paragraph("Missing image", styles["Caption"])])
                    if len(row) == 2:
                        pending_rows.append(row)
                        row = []
                if row:
                    pending_rows.append(row)

                for pair in pending_rows:
                    t = Table([pair], colWidths=[page_width / 2] * len(pair))
                    t.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]))
                    story.append(t)

            story.append(Spacer(1, 14))
    else:
        story.append(Paragraph("No test prediction images found in run directory.", styles["SubText"]))

    story.append(PageBreak())

    # ---------------- 7. Artifacts ----------------
    story.append(Paragraph("Model Artifacts", styles["SectionHeader"]))
    weights_dir = os.path.join(run_dir, "weights")
    best_pt = os.path.join(weights_dir, "best.pt")
    last_pt = os.path.join(weights_dir, "last.pt")

    artifact_rows = [["File", "Exists", "Size (MB)"]]
    for label, path in [("best.pt", best_pt), ("last.pt", last_pt)]:
        exists = os.path.exists(path)
        size_mb = f"{os.path.getsize(path) / (1024*1024):.1f}" if exists else "-"
        artifact_rows.append([label, "Yes" if exists else "No", size_mb])

    table = Table(artifact_rows, colWidths=[5 * cm, 4 * cm, 4 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title=title,
    )
    doc.build(story)
    print(f"Report saved to: {output_path}")


