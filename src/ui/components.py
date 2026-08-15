"""
Reusable UI building blocks shared across pages.
"""

import streamlit as st


def pipeline_status(steps_done: list, steps_all: list, current: str = None):
    """
    Render a compact status line like:
      ✓ Quadrants   ✓ Teeth   ⟳ Health   ○ Disease   ○ Severity
    """
    parts = []
    for step in steps_all:
        if step == current:
            parts.append(f"⟳ {step}")
        elif step in steps_done:
            parts.append(f"✓ {step}")
        else:
            parts.append(f"○ {step}")
    st.markdown(
        f"<div style='font-size:0.95rem; letter-spacing:0.5px;'>{'&nbsp;&nbsp;&nbsp;'.join(parts)}</div>",
        unsafe_allow_html=True
    )


def probability_bars(probs: dict, title: str = None, highlight_top: bool = True):
    """
    Render a full probability distribution as horizontal bars, sorted
    descending. Never collapses this to just the top class - showing the
    full distribution (including close second-place calls) is a hard
    requirement of this app.
    """
    if not probs:
        st.caption("No probability data available for this prediction.")
        return

    if title:
        st.markdown(f"**{title}**")

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_value = ranked[0][1] if ranked else 0

    for i, (cls, p) in enumerate(ranked):
        pct = float(p) if p <= 1.0001 else float(p) / 100.0
        pct = max(0.0, min(1.0, pct))
        is_top = highlight_top and i == 0
        bar_color = "#4caf50" if is_top else "#78909c"
        label = f"**{cls}**" if is_top else cls

        # flag close calls (second place within 10 points of the top)
        close_call = (i == 1 and ranked and (top_value - p) < 0.10) if len(ranked) > 1 else False

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f"<div style='background:#eceff1; border-radius:6px; height:20px; position:relative;'>"
                f"<div style='background:{bar_color}; width:{pct*100:.1f}%; height:100%; border-radius:6px;'></div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.caption(label + (" ⚠️ close call" if close_call else ""))
        with col2:
            st.markdown(f"**{pct*100:.1f}%**")


def tooth_summary_card(tooth: dict):
    """
    A compact card for one tooth's full journey through the pipeline:
    ID -> health -> disease -> severity, each with its probability set.
    """
    with st.container(border=True):
        st.markdown(f"### Tooth #{tooth.get('tooth_class_id', tooth.get('class_name', '?'))}")
        st.caption(f"Quadrant: {tooth.get('quadrant', tooth.get('quad_key', ' - '))}")

        status = tooth.get('health_status', tooth.get('status'))
        if status:
            st.markdown(f"**Health:** {status}")
            probability_bars(tooth.get('health_probs', tooth.get('status_probs')), title=None)

        disease = tooth.get('disease')
        if disease:
            st.markdown(f"**Disease (model prediction):** {disease}")
            probability_bars(tooth.get('disease_probs'), title=None)

        severity = tooth.get('caries_severity')
        if severity:
            st.markdown(f"**Caries Severity:** {severity}")
            probability_bars(tooth.get('caries_severity_probs'), title=None)


def metric_row(items: list):
    """items: list of (label, value) tuples, rendered as st.metric columns."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def disclaimer_banner():
    from core.config import DISCLAIMER
    st.caption(f"ℹ️ {DISCLAIMER}")
