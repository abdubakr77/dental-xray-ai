"""
Reusable UI building blocks shared across pages.
"""

import re
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


def find_close_calls(labeled_probs: list, threshold: float = 0.10) -> list:
    """
    Given [(tooth_label, probs_dict), ...], return entries where the top two
    predictions are within `threshold` of each other - i.e. genuinely
    ambiguous calls worth flagging, not every prediction the model made.
    """
    close = []
    for label, probs in labeled_probs:
        if not probs or len(probs) < 2:
            continue
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_cls, top_p = ranked[0]
        second_cls, second_p = ranked[1]
        top_p = top_p if top_p <= 1.0001 else top_p / 100.0
        second_p = second_p if second_p <= 1.0001 else second_p / 100.0
        if (top_p - second_p) < threshold:
            close.append({'label': label, 'top': (top_cls, top_p), 'second': (second_cls, second_p)})
    return close


def render_uncertainty_notice(close_calls: list):
    """
    A single clean, professional notice summarizing genuinely ambiguous
    predictions - the user-facing replacement for dumping raw per-tooth
    warning strings into the main flow. Belongs in the final summary only.
    """
    if not close_calls:
        return
    lines = []
    for c in close_calls[:5]:
        top_cls, top_p = c['top']
        second_cls, second_p = c['second']
        lines.append(f"**{c['label']}** - {top_cls} ({top_p*100:.0f}%) vs {second_cls} ({second_p*100:.0f}%)")
    more = f"<br><span style='opacity:0.7;'>+ {len(close_calls) - 5} more</span>" if len(close_calls) > 5 else ""

    st.markdown(
        "<div class='uncertainty-card'>"
        "<div class='uncertainty-title'>⚠ Close Predictions</div>"
        "<div style='font-size:0.85rem; margin-top:6px; opacity:0.9;'>"
        "The model found closely competing possibilities for the following - the second "
        "option is close enough to the top prediction to be worth a second look:</div>"
        f"<div style='font-size:0.85rem; margin-top:8px; line-height:1.7;'>{'<br>'.join(lines)}{more}</div>"
        "</div>",
        unsafe_allow_html=True
    )


# Maps the pipeline's internal event names (see model_utils.py's exported
# log dataframes) to one friendly, non-technical sentence each. Counts are
# appended live; the raw per-tooth detail stays in Debug Mode only.
_WARNING_FRIENDLY_TEXT = {
    'low_confidence': "detection{s} had lower confidence and may be worth a second look",
    'missing_teeth': "expected tooth position{s} could not be confidently detected",
    'needs_manual_review': "tooth box{es} may span more than one tooth and {was_were} flagged for review",
    'missing_quad': "quadrant{s} could not be confidently located",
    'duplicate_quad': "quadrant detection produced a duplicate that was resolved automatically",
}


def friendly_warning_summary(raw_warnings: list) -> list:
    """
    Turn raw pipeline warning strings (e.g. "low_confidence: tooth 6
    (confidence: 0.51)") into a handful of deduplicated, human-readable
    lines by category, with a count - never the raw per-item text. Unknown
    event types are skipped here (they still appear verbatim in Debug Mode).
    """
    counts = {}
    for w in raw_warnings or []:
        event = re.split(r'[:\s]', w.strip(), maxsplit=1)[0]
        counts[event] = counts.get(event, 0) + 1

    lines = []
    for event, template in _WARNING_FRIENDLY_TEXT.items():
        n = counts.get(event, 0)
        if n == 0:
            continue
        text = template.format(s="s" if n != 1 else "", es="es" if n != 1 else "",
                                was_were="were" if n != 1 else "was")
        lines.append(f"{n} {text}.".replace("1 detections", "1 detection"))
    return lines


def disclaimer_banner():
    from core.config import DISCLAIMER
    st.caption(f"ℹ️ {DISCLAIMER}")
