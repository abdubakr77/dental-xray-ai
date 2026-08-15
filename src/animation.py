"""
CSS/SVG animation helpers for the Streamlit app.

Design choice: animation is done with SVG + CSS (not redrawing images with
opencv/PIL every frame), because CSS transitions run in the browser with no
server round-trip, which stays smooth even on a slow connection. The Python
side only decides WHEN a new element should appear (by adding one more box
per rerun); the actual glow/fade motion is entirely CSS.
"""

import streamlit as st

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

@keyframes rectGlow {
    0%   { filter: drop-shadow(0 0 0px currentColor); }
    30%  { filter: drop-shadow(0 0 15px currentColor); }
    100% { filter: drop-shadow(0 0 3px currentColor); }
}

@keyframes textPulse {
    0%   { opacity: 0; transform: scale(0.8); }
    50%  { opacity: 1; transform: scale(1.05); }
    100% { opacity: 1; transform: scale(1); }
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

@keyframes slideInUp {
    0%   { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes countUp {
    0%   { opacity: 0; transform: scale(0.5); }
    50%  { transform: scale(1.1); }
    100% { opacity: 1; transform: scale(1); }
}

.anim-box rect {
    animation: rectGlow 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
    transition: stroke-width 0.3s ease;
}

.anim-box:hover rect {
    stroke-width: 6 !important;
}

.anim-box text {
    animation: textPulse 0.6s ease-out forwards;
    animation-delay: 0.2s;
    font-weight: 700;
    text-shadow: 0 0 3px rgba(0, 0, 0, 0.5);
}

.anim-label {
    display: inline-block;
    animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
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
    animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards,
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
    display: inline-block;
    border-radius: 12px;
    overflow: hidden;
    background: #f5f5f5;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.overlay-container img {
    display: block;
    width: 100%;
    height: auto;
}

.overlay-svg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}
</style>
"""


def inject_animation_css():
    """Call once near the top of the app, before any animated element renders."""
    st.markdown(ANIMATION_CSS, unsafe_allow_html=True)


def svg_box_overlay(image_width, image_height, boxes, box_color="#00e676"):
    """
    Builds an SVG overlay of bounding boxes, each with its own fade-in-glow
    animation. Boxes already in `boxes` all render at once (their individual
    CSS animation plays on mount), so progressive reveal is done by calling
    this again with one more box added each rerun, not by re-animating
    existing ones.

    Args:
        image_width, image_height: the underlying image's pixel size, so the
                                     SVG viewBox lines up with it exactly
        boxes: list of dicts: {'x1','y1','x2','y2','label','confidence'}
        box_color: stroke color for the boxes and labels in this overlay

    Returns:
        str: an <svg> element as a string, meant to be layered over the image
             with absolute positioning (see render_image_with_overlay below)
    """
    elements = []
    for b in boxes:
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        label = b.get('label', '')
        conf = b.get('confidence')
        label_text = f"{label} {conf:.0%}" if conf is not None else label

        # Clamp text position to keep it inside the image
        text_x = max(x1, 0)
        text_y = max(y1 - 8, 20)
        if text_y + 20 > image_height:
            text_y = y2 + 16

        elements.append(f'''
            <g class="anim-box" style="color:{box_color}">
                <rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}"
                      fill="none" stroke="{box_color}" stroke-width="3" rx="4" ry="4" />
                <text x="{text_x}" y="{text_y}" fill="{box_color}"
                      font-size="14" font-weight="700" font-family="Arial, sans-serif"
                      text-anchor="start">{label_text}</text>
            </g>
        ''')

    return f'''
        <svg class="overlay-svg" viewBox="0 0 {image_width} {image_height}"
             preserveAspectRatio="xMidYMid meet"
             style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;">
            {''.join(elements)}
        </svg>
    '''


def render_image_with_overlay(image_base64, image_width, image_height, boxes, box_color="#00e676"):
    """
    Renders a base image with an animated SVG box overlay on top, using a
    container with proper aspect ratio preservation.

    Args:
        image_base64: the image already encoded as a base64 PNG string
        image_width, image_height: original pixel size of the image
        boxes: same format as svg_box_overlay
        box_color: stroke color for this overlay
    """
    svg = svg_box_overlay(image_width, image_height, boxes, box_color)
    # NOTE: fixed 2026-08 - CSS `padding-bottom` percentages are resolved against
    # the element's WIDTH, so this must be height/width (not width/height) to make
    # the aspect-ratio box actually match the image. The inverted version made wide
    # panoramic X-rays render inside a tall, mostly-empty container.
    aspect_ratio = (image_height / image_width) * 100 if image_width > 0 else 100
    
    st.markdown(f'''
        <div class="overlay-container" style="padding-bottom: {aspect_ratio}%; max-width: 900px; margin: 0 auto;">
            <img src="data:image/png;base64,{image_base64}" 
                 style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                        border-radius: 12px; object-fit: contain; display: block;" />
            {svg.replace('position:absolute', 'position:absolute').replace('width:100%', 'width: 100%').replace('height:100%', 'height: 100%')}
        </div>
    ''', unsafe_allow_html=True)


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

        cards.append(f'''
            <div class="tooth-card {css_class}">
                <img src="data:image/png;base64,{t['image_base64']}" width="90" height="90" />
                <div class="anim-label {'anim-label-disease' if is_disease else 'anim-label-healthy'}">
                    #{t['class_name']} - {status}
                </div>
            </div>
        ''')

    st.markdown(f'<div style="display:flex; flex-wrap:wrap;">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_live_counter(healthy_count, disease_count):
    """A small animated counter shown while the healthy/unhealthy stage plays out."""
    st.markdown(f'''
        <div style="display:flex; gap:24px; margin:8px 0;">
            <div class="anim-counter" style="color:#4caf50;">✓ Healthy: {healthy_count}</div>
            <div class="anim-counter" style="color:#f44336;">⚠ Disease Found: {disease_count}</div>
        </div>
    ''', unsafe_allow_html=True)