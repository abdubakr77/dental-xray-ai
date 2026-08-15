"""
Report persistence.

Layout on disk (as specified):
    reports/
        analysis_<id>/
            metadata.json     -- everything needed for the history list view
            report.json       -- full detailed report (quadrants, teeth, probs)
            original.png
            annotated.png     -- final image, boxes only on diseased teeth

Design notes:
  - IDs are timestamp + short uuid, so they sort chronologically and never
    collide, which also means we never overwrite an existing folder.
  - We do NOT persist each tooth's raw crop (`t['image']` ndarrays) inside
    report.json - that would bloat every report with redundant pixel data
    that's already visible in annotated.png / the quadrant images. Only
    detection/classification results (boxes, class ids, probabilities) are
    stored, which is everything the pipeline actually produced besides pixels.
  - Every numeric value coming out of the pipeline may be a numpy scalar
    (np.float32, np.int64, ...), which json.dump can't serialize directly  - 
    `_to_native` recursively converts those, and nothing else.
"""

import os
import json
import uuid
import shutil
from datetime import datetime

import cv2
import numpy as np

from vis import draw_infrence_boxes
from core.config import REPORTS_DIR


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items() if k != 'image'}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _new_report_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _report_dir(report_id: str) -> str:
    return os.path.join(REPORTS_DIR, f"analysis_{report_id}")


def _build_final_annotated_image(result: dict) -> np.ndarray:
    """Full panoramic image with boxes only on diseased teeth (final result view)."""
    boxes = []
    for t in result.get('diseased_teeth', []):
        quad_key = t['quad_key'].split('_')[-1]
        if quad_key not in result.get('quadrant_boxes', {}):
            continue
        qx1, qy1, _, _ = result['quadrant_boxes'][quad_key]
        tx1, ty1, tx2, ty2 = t['box']
        final_name = t.get('caries_severity') or t.get('disease', 'Unknown')
        boxes.append((qx1 + tx1, qy1 + ty1, qx1 + tx2, qy1 + ty2, f"#{t['class_name']} - {final_name}"))
    return draw_infrence_boxes(result['original_image'], boxes, color=(255, 0, 0))


def save_report(result: dict, warnings: list, stage: str, image_path: str,
                 duration_seconds: float, models_used: dict) -> str:
    """
    Persist a completed analysis. Returns the new report_id.
    Only fields the pipeline actually produced are written - nothing is
    fabricated for fields the pipeline doesn't return (e.g. no per-detection
    ground truth, no clinical validation status).
    """
    report_id = _new_report_id()
    out_dir = _report_dir(report_id)
    os.makedirs(out_dir, exist_ok=False)  # never overwrite

    original = result.get('original_image')
    h, w = (original.shape[0], original.shape[1]) if original is not None else (None, None)

    if original is not None:
        cv2.imwrite(os.path.join(out_dir, "original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

    has_final_view = original is not None and 'quadrant_boxes' in result
    if has_final_view:
        annotated = _build_final_annotated_image(result)
        cv2.imwrite(os.path.join(out_dir, "annotated.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    all_teeth = result.get('all_teeth', [])
    diseased = result.get('diseased_teeth', [])
    disease_counts = {}
    for t in diseased:
        name = t.get('caries_severity') or t.get('disease', 'Unknown')
        disease_counts[name] = disease_counts.get(name, 0) + 1

    metadata = {
        "report_id": report_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "processing_seconds": round(duration_seconds, 3),
        "pipeline_mode": stage,
        "models_used": models_used,
        "image_filename": os.path.basename(image_path) if image_path else None,
        "image_width": w,
        "image_height": h,
        "status": "completed_with_warnings" if warnings else "completed",
        "total_teeth": len(all_teeth),
        "healthy_teeth": len(all_teeth) - len(diseased),
        "diseased_teeth": len(diseased),
        "disease_counts": disease_counts,
        "has_annotated_image": has_final_view,
    }

    quadrants_out = []
    for qname, box in result.get('quadrant_boxes', {}).items():
        teeth_here = result.get('teeth_per_quadrant', {})
        n_teeth = 0
        for qk, teeth in teeth_here.items():
            if qk.endswith(qname):
                n_teeth = len(teeth)
        quadrants_out.append({
            "quadrant": qname,
            "bbox_xyxy": list(box),
            "n_teeth_detected": n_teeth,
        })

    teeth_out = []
    for t in all_teeth:
        teeth_out.append({
            "tooth_class_id": t.get('class_name'),
            "quadrant": t.get('quad_key'),
            "bbox_in_quadrant_xyxy": t.get('box'),
            "health_status": t.get('status'),
            "health_probs": t.get('status_probs'),
            "disease": t.get('disease'),
            "disease_probs": t.get('disease_probs'),
            "caries_severity": t.get('caries_severity'),
            "caries_severity_probs": t.get('caries_severity_probs'),
        })

    report_full = {
        "metadata": metadata,
        "warnings": warnings,
        "quadrants": quadrants_out,
        "teeth": teeth_out,
    }
    report_full = _to_native(report_full)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(_to_native(metadata), f, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report_full, f, indent=2)

    return report_id


def list_reports() -> list:
    """Return metadata for every saved report, newest first."""
    if not os.path.isdir(REPORTS_DIR):
        return []
    out = []
    for folder in sorted(os.listdir(REPORTS_DIR), reverse=True):
        meta_path = os.path.join(REPORTS_DIR, folder, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def load_report(report_id: str) -> dict:
    """Load the full report.json for one analysis, plus image paths if present."""
    out_dir = _report_dir(report_id)
    report_path = os.path.join(out_dir, "report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"No report found for id {report_id}")
    with open(report_path) as f:
        report = json.load(f)
    report["original_image_path"] = os.path.join(out_dir, "original.png")
    annotated_path = os.path.join(out_dir, "annotated.png")
    report["annotated_image_path"] = annotated_path if os.path.exists(annotated_path) else None
    return report
