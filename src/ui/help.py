"""
Help page - comprehensive documentation, FAQ, and troubleshooting.

Everything here is either static explanatory text about how the app works,
or pulled directly from core.config (class names, quadrant names, per-model
input requirements) so it can't drift out of sync with the actual pipeline.
"""

import streamlit as st

from core.config import CLASS_NAMES, QUADRANT_NAMES, STAGE_INPUT_HELP, DISCLAIMER
from animation import inject_animation_css


def render(debug: bool = False):
    inject_animation_css()
    st.title("❓ Help")
    st.caption(f"ℹ️ {DISCLAIMER}")

    tabs = st.tabs(["General", "Input", "Pipeline", "Individual Models", "Results", "Reports", "FAQ", "Troubleshooting"])

    with tabs[0]:
        st.markdown("### What is Dental AI?")
        st.write(
            "A computer-vision system for panoramic dental X-rays. It locates the four dental "
            "quadrants, detects and numbers every individual tooth, and classifies each tooth for "
            "health status and, where relevant, disease type and caries severity."
        )
        st.markdown("### What does the complete pipeline do?")
        st.write(
            "Given one full panoramic X-ray, it runs five stages end to end: quadrant detection, "
            "tooth detection/numbering, healthy-vs-unhealthy classification, disease classification, "
            "and (for teeth classified as Caries) a severity classification into Caries or Deep Caries."
        )
        st.markdown("### Limitations")
        st.write(
            "This is a research/decision-support tool, not a diagnostic device. Detection and "
            "classification accuracy varies by stage and by how rare a finding is in the training "
            "data (rare classes are harder for any model to learn reliably) - see the **About** page "
            "for the actual measured accuracy of each model. Every result should be reviewed by a "
            "qualified dentist before being acted on."
        )

    with tabs[1]:
        st.markdown("### What image should I upload?")
        st.write("A full panoramic dental X-ray (sometimes called an OPG - orthopantomogram) for the "
                  "complete pipeline and the Quadrant Detection model. Other individual models expect "
                  "a narrower crop - see the **Individual Models** tab.")
        st.markdown("### Supported formats")
        st.write("PNG, JPG, or JPEG.")
        st.markdown("### What if I select the wrong kind of image?")
        st.write(
            "On the **Individual Models** page, the app shows exactly what input the selected model "
            "expects before you upload, and validates that the file is a readable image. It can't "
            "always tell a panoramic X-ray from a tooth crop by content alone, though - if a result "
            "looks wrong, double check the input matched what that model expects."
        )
        st.markdown("### Image quality")
        st.write("A clear, unobstructed, properly-exposed panoramic X-ray gives the most reliable "
                  "results. Heavily cropped, rotated, or very low-resolution images may reduce "
                  "detection quality at every downstream stage.")

    with tabs[2]:
        st.markdown("### The complete pipeline")
        st.write("Full Panoramic X-Ray → 4 Quadrants → Individual Teeth → Healthy / Unhealthy → "
                  "Disease Type → Caries / Deep Caries (Caries teeth only).")
        st.markdown("### Quadrants")
        st.write("The four detected regions, in the model's own convention: " + ", ".join(QUADRANT_NAMES) + ".")
        st.markdown("### Tooth numbering")
        st.write(
            "Each tooth gets a class ID from **0 to 7**, local to its own quadrant (not a global "
            "numbering across the whole mouth). This is the exact ID the model outputs - it's shown "
            "unchanged everywhere in the app, in reports, and in exports."
        )
        st.markdown("### Disease classes")
        st.write("The disease classifier's exact classes: " + ", ".join(CLASS_NAMES['disease']) + ".")
        st.markdown("### Caries severity")
        st.write("Only runs for teeth the disease classifier called **Caries**: "
                  + " vs. ".join(CLASS_NAMES['caries_severity']) + ".")

    with tabs[3]:
        st.write("The **Individual Models** page runs exactly one model at a time - useful for "
                 "debugging or demonstrating a single stage. Each model's required input:")
        labels = {
            'quadrant': "Quadrant Detection", 'teeth': "Tooth Detection (Enumeration)",
            'healthy_unhealthy': "Healthy / Unhealthy Classification",
            'disease': "Disease Classification", 'caries_severity': "Caries / Deep Caries Classification",
        }
        for key, label in labels.items():
            st.markdown(f"**{label}**")
            st.caption(STAGE_INPUT_HELP.get(key, "—"))

    with tabs[4]:
        st.markdown("### Bounding boxes & class IDs")
        st.write("Boxes mark detected regions (quadrants or teeth). The number on a tooth box is its "
                  "model class ID (0-7, local to its quadrant) - never renamed or renumbered.")
        st.markdown("### Confidence & probabilities")
        st.write(
            "Wherever the model produces a full probability distribution (disease type, caries "
            "severity), the app shows every class's probability, not just the top prediction - so "
            "you can see when two possibilities were close rather than one being clearly certain."
        )
        st.markdown("### Health, disease, and severity results")
        st.write(
            "A tooth's pipeline stops at Healthy. An Unhealthy tooth continues to disease "
            "classification; a tooth classified as Caries continues once more into the "
            "Caries vs. Deep Caries classifier. The final label shown for a tooth is always "
            "the most specific one available (severity if it ran, otherwise disease type)."
        )

    with tabs[5]:
        st.markdown("### What gets saved")
        st.write("Every completed full-pipeline run: the original image, the final annotated image, "
                 "and a JSON report with every quadrant, every tooth, and every classification result "
                 "the pipeline actually produced. Nothing is saved for **Individual Models** runs - "
                 "those are for one-off testing, not history.")
        st.markdown("### Where reports live")
        st.write("Open **Reports / History** to browse every saved analysis, or the **Dashboard** for "
                 "aggregate statistics across all of them.")
        st.markdown("### Exports")
        st.write("From a report's detail view: the JSON report as-is, a CSV summary (one row per "
                 "tooth), a short PDF summary, and the final annotated image.")

    with tabs[6]:
        with st.expander("Does this replace a dentist?"):
            st.write("No. It's a research/decision-support tool. Every finding needs confirmation by "
                     "a qualified dentist.")
        with st.expander("Why does the pipeline take a while to run?"):
            st.write("It's running five real model inference stages (two detection models, three "
                     "classifiers) in sequence, not a lookup - runtime depends on your hardware, "
                     "especially whether a GPU is available.")
        with st.expander("Can I stop a run partway through?"):
            st.write("Yes, with the Stop button in the sidebar while a run is in progress. It "
                     "detaches the app from that run immediately; the underlying computation, once "
                     "started, finishes quietly in the background rather than being forcibly killed "
                     "(there's no safe way to interrupt a live model computation mid-call).")
        with st.expander("Why do some teeth show no disease result?"):
            st.write("Only teeth classified as Unhealthy go on to disease classification - a tooth "
                     "classified Healthy has nothing further to show by design.")
        with st.expander("Why do only some diseased teeth show a severity result?"):
            st.write("Only teeth the disease classifier calls Caries go on to the severity "
                     "classifier - Impacted and Periapical teeth stop at the disease stage.")

    with tabs[7]:
        st.markdown("**Invalid or corrupted image** — re-export or re-save the file and try again; "
                    "the app validates that an upload is a readable image before running anything.")
        st.markdown("**Model failed to load** — usually a missing or misconfigured model file; the "
                    "app shows the exact config path it looked for rather than a raw crash.")
        st.markdown("**Wrong input format for an individual model** — check that tab's required "
                    "input above; a full X-ray and a single tooth crop aren't interchangeable "
                    "between models.")
        st.markdown("**No detections at all** — usually an image that isn't actually a panoramic "
                    "dental X-ray, or one that's heavily cropped/rotated/low-resolution.")
        st.markdown("**Unsupported input for an individual model** — the app validates file type "
                    "and readability up front; a mismatched *content* (right file type, wrong kind "
                    "of image) may still run but produce a meaningless result.")
        st.markdown("**Result looks off / a debug view is needed** — turn on **Debug mode** in the "
                    "sidebar to see raw coordinates, confidence values, and the pipeline's own "
                    "internal warnings.")
