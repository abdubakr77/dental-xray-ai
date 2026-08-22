"""
CSS/SVG animation helpers for the Streamlit app.

Design choice: animation is done with SVG + CSS (not redrawing images with
opencv/PIL every frame), because CSS transitions run in the browser with no
server round-trip, which stays smooth even on a slow connection. The Python
side only decides WHEN a new element should appear (by adding one more box
per rerun); the actual glow/fade motion is entirely CSS.
"""

import re
import streamlit as st


def _flatten_html(html: str) -> str:
    """
    Collapse a multi-line, indented HTML string onto effectively one line.

    Root cause of "animation shows raw code instead of rendering": every
    HTML fragment below is built as an indented multi-line f-string (for
    readability in the Python source). When several such fragments are
    joined, the join points are blank lines followed by 8-12 spaces of
    indentation before the next tag. Streamlit's markdown renderer follows
    CommonMark, which treats a blank line followed by 4+ spaces of
    indentation as the START OF AN INDENTED CODE BLOCK - so instead of
    rendering as HTML, that fragment gets displayed as literal syntax-
    highlighted code. Stripping indentation/newlines before handing the
    string to st.markdown avoids ever triggering that rule.

    Lines are joined with a single space, not '' - some tags split their
    attributes across lines (e.g. `viewBox="..."` on one line,
    `preserveAspectRatio="..."` on the next); joining with no separator
    glues them into one malformed attribute. A trailing regex collapses any
    doubled-up whitespace that results.
    """
    joined = ' '.join(line.strip() for line in html.strip().splitlines())
    return re.sub(r' {2,}', ' ', joined)

# Injected once per page load. Defines the glow + fade-in keyframes and the
# base look of every animated element (box outline, label chip, counters).
ANIMATION_CSS = """
<style>
@keyframes fadeInGlow {
    0%   { 
        opacity: 0; 
        filter: drop-shadow(0 0 0px currentColor) brightness(1);
        stroke-dasharray: 1000;
        stroke-dashoffset: 1000;
    }
    40%  { 
        opacity: 1; 
        filter: drop-shadow(0 0 12px currentColor) brightness(1.1);
        stroke-dashoffset: 0;
    }
    100% { 
        opacity: 1; 
        filter: drop-shadow(0 0 3px currentColor) brightness(1);
        stroke-dashoffset: 0;
    }
}

@keyframes fadeOutShrink {
    0%   { 
        opacity: 1; 
        transform: scale(1) translateY(0);
        filter: drop-shadow(0 0 0px currentColor);
    }
    50%  {
        opacity: 0.5;
        filter: drop-shadow(0 0 2px rgba(200,200,200,0.3));
    }
    100% { 
        opacity: 0; 
        transform: scale(0.7) translateY(10px);
        filter: drop-shadow(0 0 0px transparent);
    }
}

@keyframes popIn {
    0%   { opacity: 0; transform: scale(0.5) rotate(-5deg); }
    70%  { opacity: 1; transform: scale(1.08) rotate(0deg); }
    100% { opacity: 1; transform: scale(1) rotate(0deg); }
}

@keyframes fadeUpIn {
    0%   { opacity: 0; transform: translateY(4px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes slideInUp {
    0%   { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Pure opacity fade - no motion, no scale, no delay between a box and its
   own label. Used for anything that should ease from fully hidden to fully
   visible and nothing else: detection boxes+labels, quadrant cards, tooth
   cards. */
@keyframes fadeInOnly {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes countUp {
    0%   { opacity: 0; transform: scale(0.5); }
    50%  { transform: scale(1.1); }
    100% { opacity: 1; transform: scale(1); }
}

/* The box (rect) and its class-ID label (text) are children of the same
   <g class="anim-box">; animating opacity on the GROUP fades both in
   together, at the same instant, from fully hidden - rather than the box
   popping in instantly (previously only its glow intensity was animated,
   not its visibility) followed by the label doing its own separate
   scale/bounce entrance a moment later. */
.anim-box.entering {
    animation: fadeInOnly 0.3s ease-out both;
}
.anim-box.settled {
    opacity: 1;
}

.anim-box rect {
    transition: stroke-width 0.3s ease;
}

.anim-box:hover rect {
    stroke-width: 6 !important;
}

.anim-box text {
    font-weight: 700;
    text-shadow: 0 0 3px rgba(0, 0, 0, 0.5);
}

.anim-label {
    display: inline-block;
    animation: fadeUpIn 0.35s ease-out forwards;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    margin: 3px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.anim-label-healthy {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1));
    color: #4caf50;
    animation: fadeUpIn 0.35s ease-out forwards,
              fadeOutShrink 0.6s ease-in forwards;
    animation-delay: 0s, 0.9s;
    border: 1px solid rgba(76, 175, 80, 0.3);
}

.anim-label-disease {
    background: linear-gradient(135deg, rgba(244, 67, 54, 0.25), rgba(244, 67, 54, 0.1));
    color: #ff5252;
    border: 1px solid rgba(244, 67, 54, 0.4);
    font-weight: 800;
}

.anim-counter {
    font-size: 1.3rem;
    font-weight: 800;
    animation: countUp 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    padding: 8px 16px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 2px solid currentColor;
    display: inline-block;
}

.tooth-card {
    display: inline-block;
    text-align: center;
    margin: 6px;
    animation: slideInUp 0.5s ease-out forwards;
    transition: transform 0.3s ease;
}

.tooth-card:hover {
    transform: translateY(-5px);
}

.tooth-card img {
    border-radius: 12px;
    border: 3px solid transparent;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    animation: popIn 0.5s ease-out forwards;
}

.tooth-card.disease img {
    border-color: #ff5252;
    box-shadow: 0 4px 20px rgba(244, 67, 54, 0.4);
}

.tooth-card.disease:hover img {
    box-shadow: 0 6px 25px rgba(244, 67, 54, 0.6);
    transform: scale(1.05);
}

.tooth-card.healthy img {
    border-color: rgba(76, 175, 80, 0.4);
    opacity: 0.55;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}

.tooth-card.healthy:hover img {
    opacity: 0.7;
}

.overlay-container {
    position: relative;
    width: 100%;
    border-radius: 12px;
    overflow: hidden;
    background: #f5f5f5;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    animation: settleIn 0.6s ease-out;
}

.overlay-svg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

/* ---- scanning pulse: shown briefly before the first real result is drawn,
   communicates "the AI is looking at this" without faking progress ---- */
@keyframes scanPulse {
    0%, 100% { opacity: 1; filter: brightness(1); }
    50%      { opacity: 0.72; filter: brightness(1.15); }
}
.scan-pulse {
    animation: scanPulse 1.6s ease-in-out infinite;
}

/* ---- settle-in: plays once whenever a container is (re)inserted into the
   DOM - used for the panoramic image's transition into its compact,
   "reference" state after quadrant detection, and generally for any block
   that should ease in rather than pop in ---- */
@keyframes settleIn {
    from { opacity: 0; transform: translateY(-4px) scale(1.01); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.settled {
    max-width: 640px !important;
}

/* ---- quadrant cards: pending -> active -> done ---- */
.quadrant-card {
    border-radius: 14px;
    padding: 6px;
    transition: border-color 0.4s ease, box-shadow 0.4s ease;
    border: 2px solid transparent;
}
.quadrant-card.card-entering {
    animation: fadeInOnly 0.3s ease-out backwards;
}
.quadrant-card .quadrant-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-align: center;
    padding: 4px 0 6px 0;
    letter-spacing: 0.3px;
}
.quadrant-card.pending {
    opacity: 0.55;
    filter: grayscale(0.5);
}
.quadrant-card.pending .quadrant-title {
    color: #78909c;
}
.quadrant-card.active {
    opacity: 1;
    filter: none;
    border-color: #ffab00;
    box-shadow: 0 0 18px rgba(255, 171, 0, 0.35);
}
.quadrant-card.active .quadrant-title {
    color: #ffab00;
}
.quadrant-card.active .overlay-container {
    animation: scanPulse 1.4s ease-in-out infinite;
}
.quadrant-card.done {
    opacity: 1;
    filter: none;
    border-color: #4caf50;
    box-shadow: 0 0 14px rgba(76, 175, 80, 0.25);
}
.quadrant-card.done .quadrant-title {
    color: #4caf50;
}
.quadrant-card.done .quadrant-title::after {
    content: " ✓";
}

/* ---- tooth detail cards: neutral (just detected) -> healthy / unhealthy
   -> disease name -> severity, all on the SAME card, never replaced ---- */
@keyframes cardStateChange {
    from { opacity: 0; }
    to   { opacity: 1; }
}
.tooth-detail-card {
    border-radius: 10px;
    padding: 5px;
    text-align: center;
    border: 2px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.03);
    transition: border-color 0.5s ease, box-shadow 0.5s ease, background 0.5s ease;
    min-width: 68px;
    max-width: 68px;
}
.tooth-detail-card.card-entering {
    animation: fadeInOnly 0.3s ease-out backwards;
}
.tooth-detail-card img {
    border-radius: 6px;
    width: 58px;
    height: 74px;
    object-fit: cover;
    display: block;
    margin: 0 auto 4px auto;
}
.tooth-detail-card .tooth-id {
    font-size: 0.72rem;
    font-weight: 700;
    opacity: 0.85;
}
.tooth-detail-card .tooth-status {
    font-size: 0.66rem;
    font-weight: 700;
    margin-top: 2px;
}
.tooth-detail-card .tooth-status.status-entering {
    animation: cardStateChange 0.3s ease-out both;
}
.tooth-detail-card.neutral {
    border-color: rgba(255, 255, 255, 0.15);
}
.tooth-detail-card.healthy {
    border-color: rgba(76, 175, 80, 0.5);
    background: rgba(76, 175, 80, 0.06);
    box-shadow: 0 0 10px rgba(76, 175, 80, 0.15);
}
.tooth-detail-card.healthy .tooth-status { color: #4caf50; }
.tooth-detail-card.unhealthy {
    border-color: rgba(255, 152, 0, 0.5);
    background: rgba(255, 152, 0, 0.06);
}
.tooth-detail-card.unhealthy .tooth-status { color: #ffab00; }
.tooth-detail-card.disease {
    border-color: rgba(244, 67, 54, 0.55);
    background: rgba(244, 67, 54, 0.07);
    box-shadow: 0 0 12px rgba(244, 67, 54, 0.2);
}
.tooth-detail-card.disease .tooth-status { color: #ff5252; }

/* The "unhealthy -> diagnosed" shift: plays ONCE, exactly on the render
   where a card's diagnosis just resolved (state is now 'disease' AND
   status-entering is set) - starts from the unhealthy card's own amber
   colors and eases into the disease card's red ones, so the transition
   itself is visible rather than the card just appearing already red. */
@keyframes unhealthyToDiagnosed {
    0%   { border-color: rgba(255, 152, 0, 0.5); background: rgba(255, 152, 0, 0.06); box-shadow: none; }
    100% { border-color: rgba(244, 67, 54, 0.55); background: rgba(244, 67, 54, 0.07);
           box-shadow: 0 0 12px rgba(244, 67, 54, 0.2); }
}
.tooth-detail-card.disease.status-entering {
    animation: unhealthyToDiagnosed 0.6s ease-out forwards;
}

.tooth-detail-card .tooth-sub {
    font-size: 0.6rem;
    opacity: 0.7;
    margin-top: 2px;
}
.tooth-detail-card .tooth-sub.status-entering {
    animation: cardStateChange 0.3s ease-out both;
}
.tooth-detail-card .tooth-sub.checking {
    animation: scanPulse 1.1s ease-in-out infinite;
    font-style: italic;
}

/* ---- page-level entrance: applied to the whole page container so moving
   between nav pages feels like a soft transition rather than an instant
   swap. Best-effort - relies on Streamlit's st.container(key=...) exposing
   a stable "st-key-<key>" class; harmless no-op if that class name ever
   changes in a future Streamlit release. ---- */
@keyframes pageFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.st-key-page_fade, [class*="st-key-page_fade_"] {
    animation: pageFadeIn 0.35s ease-out;
}

/* The brief placeholder frame app.py renders for one rerun right after
   nav_page changes, before the new page's own content/fade-in - this is
   what gives the OLD page an actual "recede" beat instead of the switch
   only ever animating the arrival of the new one. */
@keyframes pageFadeOut {
    from { opacity: 1; }
    to   { opacity: 0; }
}
.st-key-page_transition_out {
    animation: pageFadeOut 0.16s ease-in;
}

/* Home page sample thumbnails - gentle fade with a slight cascade across
   the row instead of popping in all at once. */
[class*="st-key-home_sample_card_0"] { animation: pageFadeIn 0.5s ease-out both; }
[class*="st-key-home_sample_card_1"] { animation: pageFadeIn 0.5s ease-out 0.1s both; }
[class*="st-key-home_sample_card_2"] { animation: pageFadeIn 0.5s ease-out 0.2s both; }
[class*="st-key-home_sample_card_3"] { animation: pageFadeIn 0.5s ease-out 0.3s both; }

/* ---- polished completion / save-confirmation card ---- */
@keyframes checkPop {
    0%   { transform: scale(0.6); opacity: 0; }
    60%  { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}
.completion-card {
    border-radius: 16px;
    padding: 20px 24px;
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.12), rgba(76, 175, 80, 0.04));
    border: 1px solid rgba(76, 175, 80, 0.35);
    animation: slideInUp 0.5s ease-out both;
    text-align: center;
}
.completion-card .check {
    font-size: 2.2rem;
    animation: checkPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.completion-card .title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #4caf50;
    margin-top: 4px;
}
.completion-card .subtitle {
    font-size: 0.85rem;
    opacity: 0.75;
    margin-top: 4px;
}

/* ---- clean, non-technical uncertainty notice (replaces raw warning dumps
   in the main flow - see ui/components.py:render_uncertainty_notice) ---- */
.uncertainty-card {
    border-radius: 12px;
    padding: 14px 18px;
    background: rgba(255, 171, 0, 0.08);
    border: 1px solid rgba(255, 171, 0, 0.3);
    animation: slideInUp 0.45s ease-out both;
}
.uncertainty-card .uncertainty-title {
    font-weight: 800;
    color: #ffab00;
    font-size: 0.95rem;
}
</style>
"""


def inject_animation_css():
    """Call once near the top of the app, before any animated element renders."""
    st.markdown(ANIMATION_CSS, unsafe_allow_html=True)


def svg_box_overlay(image_width, image_height, boxes, box_color="#00e676", entering_index=None):
    """
    Builds an SVG overlay of bounding boxes. Only the box at `entering_index`
    (if given) gets the fade-in animation class; every other box is marked
    'settled' (fully visible, no animation property at all).

    This distinction matters because the whole SVG string is rebuilt and
    re-inserted into the DOM on every rerun as one more box is added - if
    every box always carried the animation class, already-shown boxes would
    replay their fade from scratch on every subsequent rerun (their fade
    never gets a chance to finish before the DOM node is torn down and
    recreated again), which looks like instant popping rather than a smooth
    one-at-a-time reveal. A box with no animation property simply renders at
    its final appearance immediately, so re-inserting it causes no visible
    disturbance - true animation only ever happens for the one box that
    just became visible this render.

    Args:
        image_width, image_height: the underlying image's pixel size, so the
                                     SVG viewBox lines up with it exactly
        boxes: list of dicts: {'x1','y1','x2','y2','label','confidence'}
        box_color: stroke color for the boxes and labels in this overlay
        entering_index: index into `boxes` of the one box that just became
                         visible this render (None = none of them are new,
                         e.g. a static final-summary image)

    Returns:
        str: an <svg> element as a string, meant to be layered over the image
             with absolute positioning (see render_image_with_overlay below)
    """
    elements = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        label = b.get('label', '')
        conf = b.get('confidence')
        label_text = f"{label} {conf:.0%}" if conf is not None else label

        # Clamp text position to keep it inside the image
        text_x = max(x1, 0)
        text_y = max(y1 - 8, 24)
        if text_y + 24 > image_height:
            text_y = y2 + 16

        box_state = "entering" if i == entering_index else "settled"
        elements.append(f'''
            <g class="anim-box {box_state}" style="color:{box_color}">
                <rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}"
                      fill="none" stroke="{box_color}" stroke-width="3" rx="4" ry="4" />
                <text x="{text_x}" y="{text_y}" fill="{box_color}"
                      font-size="18" font-weight="700" font-family="Arial, sans-serif"
                      text-anchor="start">{label_text}</text>
            </g>
        ''')

    return _flatten_html(f'''
        <svg class="overlay-svg" viewBox="0 0 {image_width} {image_height}"
             preserveAspectRatio="xMidYMid meet"
             style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;">
            {''.join(elements)}
        </svg>
    ''')


def render_image_with_overlay(image_base64, image_width, image_height, boxes, box_color="#00e676",
                               max_width_px=900, extra_class="", entering_index=None):
    """
    Renders a base image with an animated SVG box overlay on top, using a
    container with proper aspect ratio preservation.

    Args:
        image_base64: the image already encoded as a base64 PNG string
        image_width, image_height: original pixel size of the image
        boxes: same format as svg_box_overlay
        box_color: stroke color for this overlay
        max_width_px: cap on the container's width (smaller for quadrant/tooth
                       cards than for the full panoramic image)
        extra_class: additional CSS class(es) on the container, e.g. "settled"
                      for the compact post-quadrant-stage panoramic image
        entering_index: forwarded to svg_box_overlay - the one box (if any)
                         that should play its fade-in this render
    """
    svg = svg_box_overlay(image_width, image_height, boxes, box_color, entering_index=entering_index)
    # NOTE: fixed 2026-08 - previously used the CSS padding-bottom percentage
    # trick, which stayed fragile even after correcting the direction of the
    # ratio (still produced a tall, partly-empty box in some layouts). Native
    # CSS `aspect-ratio` states the box's proportions directly - there's no
    # percentage-of-width math to get backwards, so this is both simpler and
    # more robust. Supported in every browser this app would realistically
    # run in (Chrome/Firefox/Safari/Edge since 2021).
    st.markdown(_flatten_html(f'''
        <div class="overlay-container {extra_class}"
             style="aspect-ratio: {image_width} / {image_height}; max-width: {max_width_px}px; margin: 0 auto;">
            <img src="data:image/png;base64,{image_base64}"
                 style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                        border-radius: 12px; object-fit: contain; display: block;" />
            {svg}
        </div>
    '''), unsafe_allow_html=True)


def render_quadrant_card(title, image_base64, image_width, image_height, boxes,
                          state="pending", box_color="#ffab00", card_entering=False,
                          entering_box_index=None):
    """
    One quadrant's card: state is 'pending' (grayed out, not started yet),
    'active' (currently being analyzed - glow + scan pulse, teeth boxes
    fill in as they're passed in `boxes`), or 'done' (green, settled, all
    teeth boxes present). The image and its boxes never disappear once
    shown - later stages only ever ADD to `boxes` or change `state`.

    card_entering: True only on the one render where this card itself first
                    appears - plays its own fade-in once, then every later
                    render (even though the whole card is rebuilt each time)
                    renders statically so it doesn't keep re-flickering.
    entering_box_index: forwarded to svg_box_overlay - the one tooth box (if
                          any) that should fade in this render.
    """
    svg = svg_box_overlay(image_width, image_height, boxes, box_color,
                           entering_index=entering_box_index) if boxes else ""
    inner_img = f'''
        <div class="overlay-container" style="aspect-ratio: {image_width} / {image_height};">
            <img src="data:image/png;base64,{image_base64}"
                 style="position:absolute; top:0; left:0; width:100%; height:100%;
                        border-radius: 10px; object-fit: contain; display:block;" />
            {svg}
        </div>
    ''' if image_base64 else ""

    card_appear_class = "card-entering" if card_entering else "card-settled"
    st.markdown(_flatten_html(f'''
        <div class="quadrant-card {state} {card_appear_class}">
            <div class="quadrant-title">{title}</div>
            {inner_img}
        </div>
    '''), unsafe_allow_html=True)


def _tooth_detail_card_html(tooth_id, image_base64, state="neutral", status_text="", sub_text="",
                             card_entering=False, status_entering=False, checking=False):
    card_appear_class = "card-entering" if card_entering else "card-settled"
    outer_change_class = "status-entering" if status_entering else ""
    status_class = "status-entering" if status_entering else ""
    sub_class = ("status-entering" if status_entering else "") + (" checking" if checking else "")
    return f'''
        <div class="tooth-detail-card {state} {card_appear_class} {outer_change_class}">
            <img src="data:image/png;base64,{image_base64}" />
            <div class="tooth-id">#{tooth_id}</div>
            {f'<div class="tooth-status {status_class}">{status_text}</div>' if status_text else ''}
            {f'<div class="tooth-sub {sub_class}">{sub_text}</div>' if sub_text else ''}
        </div>
    '''


def render_tooth_detail_card(tooth_id, image_base64, state="neutral", status_text="", sub_text="",
                              card_entering=False, status_entering=False, checking=False):
    """
    One tooth's persistent detail card, rendered on its own. Prefer
    render_tooth_detail_grid when showing several teeth together (e.g. all
    of one quadrant's teeth) - a run of individual st.markdown calls each
    becomes its own block-level element and stacks one per line regardless
    of the card's own size, which is why this alone isn't the compact-grid
    layout callers usually want.

    state: 'neutral' (just detected, not yet classified) | 'healthy' |
           'unhealthy' (disease pending) | 'disease' (final disease/severity
           name known)
    card_entering / status_entering: see render_tooth_detail_grid.
    """
    st.markdown(_flatten_html(_tooth_detail_card_html(
        tooth_id, image_base64, state, status_text, sub_text, card_entering, status_entering, checking
    )), unsafe_allow_html=True)


def render_tooth_detail_grid(cards: list):
    """
    Renders several tooth detail cards together as one compact, wrapping
    grid (small, clear gaps - not one card per line). `cards` is a list of
    dicts with keys tooth_id, image_base64, state, status_text, sub_text
    (same meaning as render_tooth_detail_card), plus:

      card_entering:   True only the one render where this card first
                        appears at all. The whole grid gets rebuilt into one
                        HTML string every rerun (so it can lay out as a
                        compact wrapping grid rather than one block per
                        card - see the module docstring on _flatten_html for
                        why single-card st.markdown calls don't achieve
                        that), which means every card would otherwise
                        replay its entrance fade on every single rerun for
                        as long as it stays on screen. Gating this to only
                        the true first appearance is what stops that.
      status_entering:  True only the one render where status_text/sub_text
                        just changed (health just resolved, disease name
                        just replaced "checking...", etc.) - same reasoning,
                        scoped to the inner text instead of the whole card.
      checking:         True while sub_text is a transient "Checking..."
                        placeholder - gives it a gentle pulse so a classifier
                        genuinely still being awaited reads as "in progress"
                        rather than looking identical to a settled result.

    Pass cards already in the order you want them to appear - this function
    doesn't reorder them.
    """
    html = ''.join(_tooth_detail_card_html(
        c['tooth_id'], c['image_base64'], c.get('state', 'neutral'),
        c.get('status_text', ''), c.get('sub_text', ''),
        c.get('card_entering', False), c.get('status_entering', False), c.get('checking', False),
    ) for c in cards)
    st.markdown(_flatten_html(f'''
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:4px;">
            {html}
        </div>
    '''), unsafe_allow_html=True)


def render_completion_card(title, subtitle):
    st.markdown(_flatten_html(f'''
        <div class="completion-card">
            <div class="check">✅</div>
            <div class="title">{title}</div>
            <div class="subtitle">{subtitle}</div>
        </div>
    '''), unsafe_allow_html=True)


def render_tooth_grid(teeth, status_key='status', highlight='disease'):
    """
    Renders a horizontal wrapping grid of tooth crop thumbnails, each in its
    own fade-in card. Teeth whose status_key value matches `highlight`
    (case-insensitively contains 'disease') stay fully visible with a red
    border; anything else fades out and shrinks (see .anim-label-healthy /
    fadeOutShrink), which is what produces the "healthy teeth disappear,
    diseased teeth stay" effect requested for the healthy/unhealthy stage.

    Args:
        teeth: list of dicts, each with 'image_base64', 'class_name', status_key
        status_key: which key in each tooth dict holds its current label
        highlight: substring that, if found in the status value, keeps the
                   tooth visible; anything else gets the fade-out treatment
    """
    cards = []
    for t in teeth:
        status = str(t.get(status_key, ''))
        is_disease = highlight.lower() in status.lower()
        css_class = 'disease' if is_disease else 'healthy'

        cards.append(_flatten_html(f'''
            <div class="tooth-card {css_class}">
                <img src="data:image/png;base64,{t['image_base64']}" width="90" height="90" />
                <div class="anim-label {'anim-label-disease' if is_disease else 'anim-label-healthy'}">
                    #{t['class_name']} - {status}
                </div>
            </div>
        '''))

    st.markdown(f'<div style="display:flex; flex-wrap:wrap;">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_live_counter(healthy_count, disease_count):
    """A small animated counter shown while the healthy/unhealthy stage plays out."""
    st.markdown(_flatten_html(f'''
        <div style="display:flex; gap:24px; margin:8px 0;">
            <div class="anim-counter" style="color:#4caf50;">✓ Healthy: {healthy_count}</div>
            <div class="anim-counter" style="color:#f44336;">⚠ Disease Found: {disease_count}</div>
        </div>
    '''), unsafe_allow_html=True)