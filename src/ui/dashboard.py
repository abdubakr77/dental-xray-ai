"""
Dashboard: aggregate statistics computed strictly from saved reports.
If there are no reports yet, this says so instead of showing fabricated charts.
"""

import pandas as pd
import streamlit as st

from storage.reports import list_reports
from ui.components import metric_row, disclaimer_banner


def render(debug: bool = False):
    st.title("🦷 Dental AI - Dashboard")
    disclaimer_banner()
    st.divider()

    reports = list_reports()
    if not reports:
        st.info("No analyses yet. Once you run analyses on the **Analysis** page, "
                 "aggregate statistics will appear here.")
        return

    df = pd.DataFrame(reports)
    total_teeth = int(df.get('total_teeth', pd.Series(dtype=int)).sum())
    healthy = int(df.get('healthy_teeth', pd.Series(dtype=int)).sum())
    diseased = int(df.get('diseased_teeth', pd.Series(dtype=int)).sum())
    avg_time = df.get('processing_seconds', pd.Series(dtype=float)).mean()

    metric_row([
        ("Total Analyses", len(df)),
        ("Total Teeth Analyzed", total_teeth),
        ("Healthy Teeth", healthy),
        ("Diseased Teeth", diseased),
    ])

    disease_counts = {}
    for counts in df.get('disease_counts', []):
        if isinstance(counts, dict):
            for name, n in counts.items():
                disease_counts[name] = disease_counts.get(name, 0) + n

    most_common = max(disease_counts, key=disease_counts.get) if disease_counts else " - "
    metric_row([
        ("Most Common Finding", most_common),
        ("Avg. Inference Time", f"{avg_time:.2f}s" if pd.notna(avg_time) else " - "),
    ])

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Disease Distribution")
        if disease_counts:
            st.bar_chart(pd.Series(disease_counts, name="Count"))
        else:
            st.caption("No diseased teeth recorded yet.")

    with col2:
        st.subheader("Healthy vs Diseased")
        if total_teeth > 0:
            st.bar_chart(pd.DataFrame({"Count": [healthy, diseased]}, index=["Healthy", "Diseased"]))
        else:
            st.caption("No teeth recorded yet.")

    st.subheader("Analyses Over Time")
    if 'date' in df.columns:
        by_date = df.groupby('date').size().rename("Analyses")
        st.bar_chart(by_date)

    st.subheader("Recent Analyses")
    cols = [c for c in ['report_id', 'date', 'time', 'pipeline_mode', 'total_teeth', 'diseased_teeth']
            if c in df.columns]
    st.dataframe(df[cols].head(10), width='stretch', hide_index=True)
