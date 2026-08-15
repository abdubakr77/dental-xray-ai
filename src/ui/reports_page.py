"""
Reports / History page.

Lists every analysis saved by storage.reports.save_report (called
automatically at the end of a full run in ui/analysis.py), and lets the user
open one for a detailed stage-by-stage view plus export.
"""

import streamlit as st

from storage.reports import list_reports, load_report
from storage.export import report_to_csv_bytes, report_to_json_bytes, report_to_pdf_bytes
from ui.components import metric_row, probability_bars, disclaimer_banner


def render(debug: bool = False):
    st.title("📊 Reports / History")
    disclaimer_banner()
    st.divider()

    reports = list_reports()
    if not reports:
        st.info("No analyses saved yet. Run a full analysis on the **Analysis** page - "
                 "it's saved here automatically when it completes.")
        return

    if "open_report_id" not in st.session_state:
        st.session_state.open_report_id = None

    if st.session_state.open_report_id is None:
        st.subheader(f"📈 {len(reports)} saved {'analysis' if len(reports) == 1 else 'analyses'}")
        for meta in reports:
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
                c1.markdown(f"**{meta['report_id']}**")
                c2.caption(f"{meta['date']} {meta['time']} · {meta['pipeline_mode']}")
                c3.metric("Teeth", meta.get('total_teeth', ' - '))
                c4.metric("Diseased", meta.get('diseased_teeth', ' - '))
                if c5.button("Open", key=f"open_{meta['report_id']}"):
                    st.session_state.open_report_id = meta['report_id']
                    st.rerun()
        return

    # ---- detail view ----
    if st.button("← Back to list"):
        st.session_state.open_report_id = None
        st.rerun()

    try:
        report = load_report(st.session_state.open_report_id)
    except FileNotFoundError as e:
        st.error(str(e))
        st.session_state.open_report_id = None
        return

    meta = report["metadata"]
    st.subheader(f"Analysis `{meta['report_id']}`")
    st.caption(f"{meta['date']} {meta['time']} · Pipeline mode: {meta['pipeline_mode']} · "
               f"Processing time: {meta['processing_seconds']}s")

    metric_row([
        ("Total Teeth", meta.get('total_teeth', 0)),
        ("Healthy", meta.get('healthy_teeth', 0)),
        ("Diseased", meta.get('diseased_teeth', 0)),
    ])

    if report.get("warnings"):
        with st.expander(f"⚠️ {len(report['warnings'])} warning(s) during processing"):
            for w in report["warnings"]:
                st.caption(f"• {w}")

    tabs = st.tabs(["Original Image", "Quadrant Analysis", "Tooth Detection", "Findings", "Final Result"])

    with tabs[0]:
        if report.get("original_image_path"):
            st.image(report["original_image_path"], width='stretch')
        else:
            st.caption("Original image not available for this report.")

    with tabs[1]:
        st.dataframe(report.get("quadrants", []), width='stretch')

    with tabs[2]:
        for t in report.get("teeth", []):
            st.caption(f"#{t['tooth_class_id']} - {t.get('quadrant', ' - ')} - "
                       f"{t.get('health_status', 'unclassified')}")

    with tabs[3]:
        diseased = [t for t in report.get("teeth", []) if t.get("disease")]
        if not diseased:
            st.success("No diseased teeth in this analysis.")
        for t in diseased:
            with st.container(border=True):
                st.markdown(f"**Tooth #{t['tooth_class_id']}** - {t.get('quadrant', ' - ')}")
                st.markdown(f"Disease: **{t.get('disease')}**")
                probability_bars(t.get("disease_probs"), title="Disease probability distribution")
                if t.get("caries_severity"):
                    st.markdown(f"Caries Severity: **{t.get('caries_severity')}**")
                    probability_bars(t.get("caries_severity_probs"), title="Severity probability distribution")

    with tabs[4]:
        if report.get("annotated_image_path"):
            st.image(report["annotated_image_path"], caption="Diseased teeth only", width='stretch')
        else:
            st.caption("Final annotated image not available for this report.")

    st.divider()
    st.subheader("⬇ Export")
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button("JSON report", data=report_to_json_bytes(report),
                        file_name=f"{meta['report_id']}_report.json", mime="application/json")
    e2.download_button("CSV summary", data=report_to_csv_bytes(report),
                        file_name=f"{meta['report_id']}_summary.csv", mime="text/csv")
    e3.download_button("PDF report", data=report_to_pdf_bytes(report),
                        file_name=f"{meta['report_id']}_report.pdf", mime="application/pdf")
    if report.get("annotated_image_path"):
        with open(report["annotated_image_path"], "rb") as f:
            e4.download_button("Annotated image", data=f.read(),
                                file_name=f"{meta['report_id']}_annotated.png", mime="image/png")
