"""
Analysis page: a single persistent, progressively-building canvas.

Architecture (replaces the old "replace this section with that section"
elif-chain): every visual element the pipeline can produce is assigned a
position in one flat, ordered "reveal sequence" once the real result is
available (see _build_reveal_sequence). A single integer pointer says how
far into that sequence we are. Every block below (panoramic image,
quadrant cards, tooth cards, final summary) checks the SAME pointer against
each element's own position to decide: not yet shown / just revealed /
already settled. Nothing already on screen is ever removed - later stages
only add to what's visible or update a tooth card's own state in place.

This means the reveal is purely a VISUALIZATION of results run_pipeline()
already computed (health/disease/severity are already sitting in `result`
by the time this page ever sees it) - the pointer never triggers new
inference, it only controls what's currently shown.
"""

import os
import time
import threading
import base64
import io

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference_pipeline import run_pipeline
from src.animation import (
    inject_animation_css, render_image_with_overlay, render_quadrant_card,
    render_tooth_detail_grid, render_completion_card,
)
from src.core.config import (
    CLASS_NAMES, STAGE_REQUIRED_INPUT, SAMPLE_IMAGES_DIR, QUADRANT_NAMES,
    TEMP_UPLOADS_DIR, DETECTION_CONF_THRESHOLD_DEFAULT,
)
from src.core.model_registry import get_models
from src.core.errors import AppError, show_error
from src.storage.reports import save_report
from src.ui.components import (
    disclaimer_banner, metric_row, find_close_calls, render_uncertainty_notice,
    friendly_warning_summary,
)

_QUAD_PRIORITY = [q.replace(' ', '') for q in QUADRANT_NAMES]

_DEFAULTS = {
    'result': None, 'warnings': [], 'running': False, 'stop_requested': False,
    'selected_image_path': None, 'run_started_at': None, 'last_report_id': None,
    'pipeline_thread': None, 'pipeline_holder': None,
    'reveal_steps': None, 'reveal_index_maps': None, 'reveal_pointer': 0,
}

STEP_DELAYS = {
    'scan_start': 0.4, 'quadrant_box': 0.18, 'quadrant_active_all': 0.15,
    'tooth_round': 0.12, 'health_round': 0.1, 'disease_round': 0.35, 'severity_round': 0.35,
}


def _init_state():
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_run():
    st.session_state.result = None
    st.session_state.warnings = []
    st.session_state.running = False
    st.session_state.stop_requested = False
    st.session_state.last_report_id = None
    st.session_state.reveal_steps = None
    st.session_state.reveal_index_maps = None
    st.session_state.reveal_pointer = 0
    # Drop the reference so a still-running background thread (if any) is
    # abandoned rather than waited on - see _pipeline_worker's docstring.
    st.session_state.pipeline_thread = None
    st.session_state.pipeline_holder = None


def _image_to_base64(img_array) -> str:
    pil_img = Image.fromarray(img_array.astype('uint8'))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _list_sample_images() -> list:
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg')))


def _sorted_quad_names(quad_names: list) -> list:
    """Anatomical order (Upper Right, Upper Left, Lower Left, Lower Right),
    matching the model's own quadrant convention - never re-derived or
    guessed, just used to lay the 4 cards out in a sensible fixed order."""
    def key(q):
        suffix = q.split('_')[-1]
        return _QUAD_PRIORITY.index(suffix) if suffix in _QUAD_PRIORITY else 99
    return sorted(quad_names, key=key)


def _sorted_teeth(teeth: list) -> list:
    """Sort a quadrant's teeth by class ID (0-7) - the pipeline's own raw
    order comes from label-file line order, not class ID, which is why
    boxes/cards were appearing in an arbitrary sequence."""
    try:
        return sorted(teeth, key=lambda t: int(t['class_name']))
    except (ValueError, TypeError):
        return teeth


def _pipeline_worker(image_path, models, class_names, conf_threshold, holder):
    """
    Runs the real (blocking) run_pipeline() call on a background thread so
    the main thread stays free to poll and process a Stop click within a
    fraction of a second, instead of being blocked inside one long call for
    the whole run. See the Stop button in render_sidebar_controls.

    This does NOT abort work already in flight inside PyTorch/YOLO - there's
    no safe way to preempt that without killing the process. Clicking Stop
    detaches the UI from this thread; its result is discarded when it
    eventually finishes on its own.
    """
    try:
        result, warnings = run_pipeline(
            image_path=image_path, models=models, class_names=class_names,
            device='cuda', conf_threshold=conf_threshold,
        )
        holder['result'] = result
        holder['warnings'] = warnings
    except Exception as e:
        holder['error'] = e
    finally:
        holder['done'] = True


# ---------------------------------------------------------------------------
# Reveal sequence: the single source of truth for "what's visible right now"
# ---------------------------------------------------------------------------

def _build_reveal_sequence(result: dict):
    quad_names = _sorted_quad_names(list(result.get('quadrant_images', {}).keys()))
    steps = []
    idx = {'quadrant_box': {}, 'quadrant_active': {}, 'quadrant_done': {},
           'tooth_box': {}, 'health': {}, 'disease': {}, 'severity': {}}

    # A genuine "nothing shown yet, AI is looking at the image" frame before
    # the first quadrant box appears - without this, the very first render
    # would already satisfy "box step <= pointer" at pointer 0.
    steps.append({'type': 'scan_start'})

    for q in quad_names:
        idx['quadrant_box'][q] = len(steps)
        steps.append({'type': 'quadrant_box', 'quad': q})
    idx['quadrant_box_wave_end'] = len(steps) - 1 if quad_names else 0

    # Sorted (class-ID 0-7) tooth list per quadrant, reused below for both
    # the detection wave and the classification waves.
    per_quad_teeth = {q: _sorted_teeth(result.get('teeth_per_quadrant', {}).get(q, [])) for q in quad_names}
    max_teeth = max((len(t) for t in per_quad_teeth.values()), default=0)

    if quad_names:
        active_step = len(steps)
        steps.append({'type': 'quadrant_active_all'})
        for q in quad_names:
            idx['quadrant_active'][q] = active_step

        # ONE step per round, shared by every quadrant - tooth index 0 in
        # ALL 4 quadrants becomes visible on the exact same step, then
        # tooth index 1 in all 4, and so on. This is what makes them
        # genuinely simultaneous rather than taking turns quickly.
        last_round_step = {}
        for round_i in range(max_teeth):
            round_step = len(steps)
            steps.append({'type': 'tooth_round', 'round': round_i})
            for q in quad_names:
                if round_i < len(per_quad_teeth[q]):
                    idx['tooth_box'][(q, round_i)] = round_step
                    last_round_step[q] = round_step

        # A quadrant is "done" the instant its OWN last tooth's round
        # happens - a quadrant with fewer teeth genuinely finishes sooner,
        # which is honest to the real per-quadrant counts rather than
        # forcing every quadrant to wait for the one with the most teeth.
        for q in quad_names:
            idx['quadrant_done'][q] = last_round_step.get(q, active_step)
    idx['quad_wave_end'] = len(steps) - 1 if quad_names else idx['quadrant_box_wave_end']

    all_teeth = result.get('all_teeth', [])
    idx['tooth_i_by_identity'] = {id(t): i for i, t in enumerate(all_teeth)}

    # Same "one step per round, shared across all quadrants" treatment for
    # health/disease/severity - Tooth-by-Tooth Analysis's 4 columns resolve
    # their Nth tooth together, not one quadrant finishing before the next
    # even starts.
    for round_i in range(max_teeth):
        round_step = len(steps)
        any_this_round = False
        for q in quad_names:
            teeth = per_quad_teeth[q]
            if round_i < len(teeth):
                t_i = idx['tooth_i_by_identity'].get(id(teeth[round_i]))
                if t_i is not None:
                    idx['health'][t_i] = round_step
                    any_this_round = True
        if any_this_round:
            steps.append({'type': 'health_round', 'round': round_i})
    idx['health_wave_end'] = len(steps) - 1 if all_teeth else idx['quad_wave_end']

    for round_i in range(max_teeth):
        round_step = len(steps)
        any_this_round = False
        for q in quad_names:
            teeth = per_quad_teeth[q]
            if round_i < len(teeth) and teeth[round_i].get('disease'):
                t_i = idx['tooth_i_by_identity'].get(id(teeth[round_i]))
                if t_i is not None:
                    idx['disease'][t_i] = round_step
                    any_this_round = True
        if any_this_round:
            steps.append({'type': 'disease_round', 'round': round_i})
    idx['disease_wave_end'] = len(steps) - 1 if idx['disease'] else idx['health_wave_end']

    for round_i in range(max_teeth):
        round_step = len(steps)
        any_this_round = False
        for q in quad_names:
            teeth = per_quad_teeth[q]
            if round_i < len(teeth) and teeth[round_i].get('caries_severity'):
                t_i = idx['tooth_i_by_identity'].get(id(teeth[round_i]))
                if t_i is not None:
                    idx['severity'][t_i] = round_step
                    any_this_round = True
        if any_this_round:
            steps.append({'type': 'severity_round', 'round': round_i})
    idx['severity_wave_end'] = len(steps) - 1 if idx['severity'] else idx['disease_wave_end']

    return steps, idx


def _tooth_card_state(tooth: dict, tooth_i: int, pointer: int, idx: dict):
    """What a tooth's persistent detail card should show right now.
    Returns (state, status_text, sub_text, just_changed, checking)."""
    health_i = idx['health'].get(tooth_i)
    if health_i is None or pointer < health_i:
        return 'neutral', '', '', False, False

    if tooth.get('status') == 'Healthy':
        return 'healthy', '✓ Healthy', '', pointer == health_i, False

    disease_i = idx['disease'].get(tooth_i)
    if disease_i is None or pointer < disease_i:
        return 'unhealthy', '⚠ Unhealthy', 'Checking disease type...', pointer == health_i, True

    disease_name = tooth.get('disease', 'Unknown')
    severity_i = idx['severity'].get(tooth_i)
    if severity_i is not None:
        if pointer < severity_i:
            return 'disease', disease_name, 'Checking severity...', pointer == disease_i, True
        return ('disease', tooth.get('caries_severity', disease_name), f"({disease_name})",
                pointer == severity_i, False)
    return 'disease', disease_name, '', pointer == disease_i, False


# ---------------------------------------------------------------------------
# Render blocks - each one is idempotent given (result, pointer, idx): call
# it again on any rerun and it draws the exact same "settled so far" state
# plus whatever just became newly visible.
# ---------------------------------------------------------------------------

def _render_panoramic_block(result: dict, pointer: int, idx: dict):
    img = result['original_image']
    h, w = img.shape[:2]
    boxes = []
    entering_index = None
    for q, step_i in idx['quadrant_box'].items():
        if step_i <= pointer:
            suffix = q.split('_')[-1]
            if suffix in result.get('quadrant_boxes', {}):
                x1, y1, x2, y2 = result['quadrant_boxes'][suffix]
                if step_i == pointer:
                    entering_index = len(boxes)
                boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': suffix})

    settled = pointer > idx.get('quadrant_box_wave_end', -1)
    scanning = not boxes and not settled

    st.subheader("🔎 Panoramic Overview" if not settled else "🔎 Panoramic Overview (reference)")
    render_image_with_overlay(
        _image_to_base64(img), w, h, boxes, box_color="#00e676",
        max_width_px=640 if settled else 900,
        extra_class="settled" if settled else ("scan-pulse" if scanning else ""),
        entering_index=entering_index,
    )


def _render_quadrant_cards_block(result: dict, pointer: int, idx: dict):
    quad_names = _sorted_quad_names(list(result.get('quadrant_images', {}).keys()))
    if not quad_names:
        return
    st.markdown("#### 🦷 Quadrant Analysis")
    st.caption("Each quadrant is analyzed in turn - active quadrant glows amber, completed ones settle to green.")
    # A quadrant card first appears the render right after the panoramic
    # box-reveal wave ends - only THAT render should play its fade-in.
    first_card_render = pointer == idx.get('quadrant_box_wave_end', -1) + 1
    cols = st.columns(len(quad_names))
    for col, q in zip(cols, quad_names):
        active_i = idx['quadrant_active'].get(q, 0)
        done_i = idx['quadrant_done'].get(q, 0)
        if pointer < active_i:
            state = 'pending'
        elif pointer < done_i:
            state = 'active'
        else:
            state = 'done'

        qimg = result['quadrant_images'][q]
        h, w = qimg.shape[:2]
        boxes = []
        entering_box_index = None
        if state != 'pending':
            for t_idx, t in enumerate(_sorted_teeth(result.get('teeth_per_quadrant', {}).get(q, []))):
                step_i = idx['tooth_box'].get((q, t_idx))
                if step_i is not None and step_i <= pointer:
                    x1, y1, x2, y2 = t['box']
                    if step_i == pointer:
                        entering_box_index = len(boxes)
                    boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': f"#{t['class_name']}"})

        with col:
            render_quadrant_card(q.split('_')[-1], _image_to_base64(qimg), w, h, boxes,
                                  state=state, box_color="#ffab00", card_entering=first_card_render,
                                  entering_box_index=entering_box_index)


def _render_tooth_cards_block(result: dict, pointer: int, idx: dict):
    all_teeth = result.get('all_teeth', [])
    if not all_teeth:
        return
    quad_names = _sorted_quad_names(list(result.get('quadrant_images', {}).keys()))
    st.markdown("#### 🔬 Tooth-by-Tooth Analysis")
    st.caption("Each tooth's card updates in place as classification resolves - detected → healthy/unhealthy → disease → severity.")
    # Block C first appears the render right after the quadrant detection
    # wave ends - only THAT render should play the cards' entrance fade.
    first_block_render = pointer == idx.get('quad_wave_end', -1) + 1
    tooth_i_by_identity = idx['tooth_i_by_identity']
    cols = st.columns(len(quad_names))
    for col, q in zip(cols, quad_names):
        with col:
            st.caption(f"**{q.split('_')[-1]}**")
            cards = []
            for t in _sorted_teeth(result.get('teeth_per_quadrant', {}).get(q, [])):
                tooth_i = tooth_i_by_identity.get(id(t))
                if tooth_i is None:
                    continue
                state, status_text, sub_text, just_changed, checking = _tooth_card_state(t, tooth_i, pointer, idx)
                cards.append({
                    'tooth_id': t['class_name'], 'image_base64': _image_to_base64(t['image']),
                    'state': state, 'status_text': status_text, 'sub_text': sub_text,
                    'card_entering': first_block_render, 'status_entering': just_changed,
                    'checking': checking,
                })
            if cards:
                render_tooth_detail_grid(cards)


def _render_final_summary_block(result: dict, warnings: list, debug: bool):
    st.divider()
    st.subheader("✅ Analysis Complete")

    all_teeth = result.get('all_teeth', [])
    diseased = result.get('diseased_teeth', [])
    healthy_n = len(all_teeth) - len(diseased)

    metric_row([
        ("🦷 Total Teeth", len(all_teeth)),
        ("✅ Healthy", healthy_n),
        ("⚠️ Diseased", len(diseased)),
    ])

    quad_names = _sorted_quad_names(list(result.get('quadrant_images', {}).keys()))
    if quad_names:
        st.markdown("##### By Quadrant")
        cols = st.columns(len(quad_names))
        for col, q in zip(cols, quad_names):
            teeth_q = result.get('teeth_per_quadrant', {}).get(q, [])
            diseased_q = sum(1 for t in teeth_q if t.get('status') == 'Disease Found')
            with col:
                st.metric(q.split('_')[-1], f"{len(teeth_q)} teeth",
                          delta=f"{diseased_q} diseased" if diseased_q else "all healthy",
                          delta_color="inverse" if diseased_q else "off")

    if diseased:
        st.markdown("##### Disease Distribution")
        counts = {}
        for t in diseased:
            name = t.get('caries_severity') or t.get('disease', 'Unknown')
            counts[name] = counts.get(name, 0) + 1
        st.bar_chart(pd.Series(counts, name="Count"))

    if all_teeth:
        st.markdown("##### Final Detected Findings")
        st.caption("Boxes shown only for teeth with a disease finding.")
        boxes = []
        for t in diseased:
            suffix = t['quad_key'].split('_')[-1]
            if suffix not in result.get('quadrant_boxes', {}):
                continue
            qx1, qy1, _, _ = result['quadrant_boxes'][suffix]
            tx1, ty1, tx2, ty2 = t['box']
            final_label = t.get('caries_severity') or t.get('disease', 'Unknown')
            boxes.append({'x1': qx1 + tx1, 'y1': qy1 + ty1, 'x2': qx1 + tx2, 'y2': qy1 + ty2,
                          'label': f"#{t['class_name']} - {final_label}"})
        img = result['original_image']
        h, w = img.shape[:2]
        render_image_with_overlay(_image_to_base64(img), w, h, boxes, box_color="#f44336", max_width_px=760)

    # Clean, professional uncertainty notice (never raw warning dumps)
    labeled_probs = []
    for t in all_teeth:
        if t.get('status_probs'):
            labeled_probs.append((f"Tooth #{t['class_name']} health", t['status_probs']))
    for t in diseased:
        if t.get('disease_probs'):
            labeled_probs.append((f"Tooth #{t['class_name']} disease", t['disease_probs']))
        if t.get('caries_severity_probs'):
            labeled_probs.append((f"Tooth #{t['class_name']} severity", t['caries_severity_probs']))
    render_uncertainty_notice(find_close_calls(labeled_probs))

    friendly = friendly_warning_summary(warnings)
    if friendly:
        with st.expander("ℹ️ A few things worth reviewing"):
            for line in friendly:
                st.caption(f"• {line}")

    if debug and warnings:
        with st.expander("🛠 Debug: raw pipeline warnings"):
            for w in warnings:
                st.code(w, language=None)

    st.divider()
    render_completion_card(
        "Report Saved",
        f"Analysis ID: {st.session_state.get('last_report_id', '—')} · "
        f"open it anytime from Reports / History"
    )


# ---------------------------------------------------------------------------
# Sidebar + page entry point
# ---------------------------------------------------------------------------

def render_sidebar_controls(debug: bool) -> tuple:
    """Returns (image_path, run_clicked, stop_clicked, reset_clicked)."""
    st.subheader("🦷 Run Analysis")

    with st.expander("📋 Input Requirements", expanded=True):
        st.info(f"**Expected input:** {STAGE_REQUIRED_INPUT['full']}")
        st.caption("Want to test one model in isolation (just quadrant detection, just the "
                   "disease classifier, etc.)? Use the **Individual Models** page - that's "
                   "where single-stage runs actually happen.")

    st.divider()
    st.subheader("📸 Select Image")

    samples = _list_sample_images()
    if 'sample_select' not in st.session_state and st.session_state.get('pending_sample_image'):
        pending = st.session_state.pop('pending_sample_image')
        if pending in samples:
            st.session_state['sample_select'] = pending
    sample_choice = st.selectbox("Sample images", ["-- none --"] + samples,
                                  key="sample_select", label_visibility="collapsed")
    uploaded = st.file_uploader("Or upload your own", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

    image_path = None
    if uploaded is not None:
        os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
        image_path = os.path.join(TEMP_UPLOADS_DIR, uploaded.name)
        with open(image_path, 'wb') as f:
            f.write(uploaded.getbuffer())
    elif sample_choice != "-- none --":
        image_path = os.path.join(SAMPLE_IMAGES_DIR, sample_choice)

    if image_path and os.path.exists(image_path):
        st.image(image_path, caption="Selected image", width='stretch')

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        run_clicked = st.button("▶ Run", type="primary", disabled=(image_path is None), width='stretch')
    with col2:
        stop_clicked = st.button("⏹ Stop", disabled=not st.session_state.running, width='stretch')
    with col3:
        reset_clicked = st.button("↻ Reset", width='stretch')

    return image_path, run_clicked, stop_clicked, reset_clicked


def render(debug: bool = False):
    inject_animation_css()
    _init_state()

    with st.sidebar:
        image_path, run_clicked, stop_clicked, reset_clicked = render_sidebar_controls(debug)

    if run_clicked:
        # Deliberately NOT calling _reset_run() here: the previous analysis
        # (if any) should stay fully visible, unchanged, for the whole time
        # the new one is processing - it only gets replaced the instant the
        # new result is actually ready (see the swap below), not the moment
        # Run is clicked.
        st.session_state.selected_image_path = image_path
        st.session_state.running = True
        st.session_state.stop_requested = False
        st.session_state.pipeline_thread = None
        st.session_state.pipeline_holder = None
        st.rerun()
    if stop_clicked:
        st.session_state.stop_requested = True
        st.session_state.running = False
        # Detach rather than join: the background thread (if any) keeps
        # computing until it naturally finishes, but we stop waiting on it
        # and its eventual result is discarded - see _pipeline_worker's
        # docstring for why a hard cancel isn't possible here. The previous
        # analysis (if any) is untouched and stays exactly as it was.
        st.session_state.pipeline_thread = None
        st.session_state.pipeline_holder = None
        st.rerun()
    if reset_clicked:
        _reset_run()
        st.rerun()

    st.title("🦷 Panoramic X-Ray Analysis")
    disclaimer_banner()
    st.divider()

    needs_poll_rerun = False

    # ---- run the real pipeline once, off the main thread so Stop is responsive ----
    if st.session_state.running and not st.session_state.stop_requested:
        try:
            if st.session_state.pipeline_thread is None:
                models = get_models()
                holder = {'result': None, 'warnings': None, 'error': None, 'done': False}
                st.session_state.pipeline_holder = holder
                thread = threading.Thread(
                    target=_pipeline_worker,
                    args=(st.session_state.selected_image_path, models, CLASS_NAMES,
                          DETECTION_CONF_THRESHOLD_DEFAULT, holder),
                    daemon=True,
                )
                st.session_state.run_started_at = time.time()
                st.session_state.pipeline_thread = thread
                thread.start()

            holder = st.session_state.pipeline_holder
            elapsed = time.time() - (st.session_state.run_started_at or time.time())
            st.info(f"⏳ Running a new analysis... ({elapsed:.0f}s elapsed) - quadrants → teeth → "
                    f"health → disease → severity, in one pass. Click **Stop** in the sidebar to cancel.")
            if st.session_state.result is not None:
                st.caption("The previous analysis below stays as-is until this one finishes.")

            if not holder['done']:
                needs_poll_rerun = True
            elif holder['error'] is not None:
                st.session_state.running = False
                st.session_state.pipeline_thread = None
                st.session_state.pipeline_holder = None
                show_error(holder['error'], debug=debug)
            else:
                # The new result is ready - THIS is the one moment the old
                # analysis (if any) actually gets replaced, quietly: the new
                # panoramic block's own entrance fade (see _render_panoramic_
                # block / .overlay-container's settleIn animation) is what
                # makes the swap read as easing in rather than a hard cut.
                st.session_state.result = holder['result']
                st.session_state.warnings = holder['warnings'] or []
                steps, idx = _build_reveal_sequence(st.session_state.result)
                st.session_state.reveal_steps = steps
                st.session_state.reveal_index_maps = idx
                st.session_state.reveal_pointer = 0
                st.session_state.running = False
                st.session_state.pipeline_thread = None

                duration = time.time() - (st.session_state.run_started_at or time.time())
                try:
                    report_id = save_report(
                        result=holder['result'], warnings=holder['warnings'] or [], stage='full',
                        image_path=st.session_state.selected_image_path,
                        duration_seconds=duration, models_used={"class_names": CLASS_NAMES},
                    )
                    st.session_state.last_report_id = report_id
                except Exception as e:
                    st.session_state.warnings.append(f"Report could not be saved: {e}")
                st.session_state.pipeline_holder = None

        except AppError as e:
            st.session_state.running = False
            st.session_state.pipeline_thread = None
            st.session_state.pipeline_holder = None
            show_error(e, debug=debug)
        except Exception as e:
            st.session_state.running = False
            st.session_state.pipeline_thread = None
            st.session_state.pipeline_holder = None
            show_error(e, debug=debug)

    result = st.session_state.result

    if result is not None and not st.session_state.stop_requested:
        steps = st.session_state.reveal_steps
        idx = st.session_state.reveal_index_maps
        pointer = st.session_state.reveal_pointer

        _render_panoramic_block(result, pointer, idx)

        if pointer > idx.get('quadrant_box_wave_end', -1):
            _render_quadrant_cards_block(result, pointer, idx)

        if pointer > idx.get('quad_wave_end', -1):
            _render_tooth_cards_block(result, pointer, idx)

        if pointer >= len(steps):
            _render_final_summary_block(result, st.session_state.warnings, debug)
        elif not st.session_state.running:
            # Only auto-advance the reveal animation when there's no new run
            # in progress - if one is, the poll loop below drives reruns
            # instead, and we don't want two independent sources of rerun.
            delay = STEP_DELAYS.get(steps[pointer]['type'], 0.2)
            time.sleep(delay)
            st.session_state.reveal_pointer += 1
            st.rerun()

    elif st.session_state.stop_requested:
        st.info("⏹ Processing stopped. Choose another image to continue.")
        if st.button("Reset & Start Over"):
            _reset_run()
            st.rerun()

    else:
        st.markdown("Select an image from the sidebar to begin, or visit **Home** for a quick overview "
                     "of how the pipeline works.")

    if needs_poll_rerun:
        time.sleep(0.3)
        st.rerun()
