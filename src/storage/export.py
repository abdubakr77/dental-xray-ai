"""
Export a saved report to CSV or PDF. JSON export needs no conversion - the
saved report.json IS the export.

No new dependencies are introduced: pandas and matplotlib are already used
elsewhere in the existing pipeline code.
"""

import io
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def report_to_csv_bytes(report: dict) -> bytes:
    """One row per detected tooth."""
    rows = []
    for t in report.get("teeth", []):
        rows.append({
            "Quadrant": t.get("quadrant"),
            "Tooth Class ID": t.get("tooth_class_id"),
            "Health Status": t.get("health_status"),
            "Disease": t.get("disease"),
            "Caries Severity": t.get("caries_severity"),
        })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def report_to_json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2).encode("utf-8")


def report_to_pdf_bytes(report: dict) -> bytes:
    """
    A short, presentable PDF summary: header + metrics, annotated image if
    available, and a table of diseased teeth. Built with matplotlib's
    PdfPages so no extra PDF dependency is required.
    """
    meta = report["metadata"]
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.suptitle("Dental AI - Analysis Report", fontsize=16, fontweight="bold", y=0.98)

        info_lines = [
            f"Analysis ID: {meta['report_id']}",
            f"Date / Time: {meta['date']} {meta['time']}",
            f"Pipeline mode: {meta['pipeline_mode']}",
            f"Processing time: {meta['processing_seconds']}s",
            f"Total teeth: {meta['total_teeth']}   Healthy: {meta['healthy_teeth']}   "
            f"Diseased: {meta['diseased_teeth']}",
        ]
        fig.text(0.08, 0.92, "\n".join(info_lines), fontsize=10, va="top")

        if report.get("annotated_image_path"):
            ax_img = fig.add_axes([0.08, 0.45, 0.84, 0.4])
            img = plt.imread(report["annotated_image_path"])
            ax_img.imshow(img)
            ax_img.set_title("Final Detected Disease Summary", fontsize=11)
            ax_img.axis("off")

        diseased_rows = [
            [t.get("quadrant"), f"#{t.get('tooth_class_id')}",
             t.get("caries_severity") or t.get("disease") or "-"]
            for t in report.get("teeth", [])
            if t.get("disease")
        ]
        if diseased_rows:
            ax_table = fig.add_axes([0.08, 0.05, 0.84, 0.35])
            ax_table.axis("off")
            table = ax_table.table(
                cellText=diseased_rows,
                colLabels=["Quadrant", "Tooth #", "Finding"],
                loc="center", cellLoc="left"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.4)

        fig.text(0.08, 0.02,
                  "AI-generated results are for research / decision-support purposes only "
                  "and are not a substitute for professional dental diagnosis.",
                  fontsize=7, style="italic")

        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()
