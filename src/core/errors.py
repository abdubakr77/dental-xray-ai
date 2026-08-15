"""
User-facing error handling.

`AppError` carries a short, human message plus optional technical detail.
`show_error` renders the human message in the main UI and tucks the technical
detail behind an expander (or into Debug mode) instead of a raw traceback.
"""

import traceback
import streamlit as st


class AppError(Exception):
    """An error with a message safe to show a non-technical user directly."""
    def __init__(self, message: str, technical: str = ""):
        super().__init__(message)
        self.message = message
        self.technical = technical


def show_error(err: Exception, debug: bool = False):
    """Render an error without dumping a raw traceback into the main UI."""
    if isinstance(err, AppError):
        st.error(f"❌ {err.message}")
        if err.technical:
            with st.expander("Technical details"):
                st.code(err.technical)
    else:
        st.error(f"❌ Something went wrong: {err}")
        if debug:
            with st.expander("Technical details (debug mode)"):
                st.code(traceback.format_exc())
        else:
            st.caption("Enable Debug Mode in the sidebar to see the full technical trace.")


def validate_image_file(uploaded_file) -> None:
    """Raise AppError if an uploaded file isn't a readable image."""
    if uploaded_file is None:
        raise AppError("No image was uploaded.")

    valid_ext = ('.png', '.jpg', '.jpeg')
    if not uploaded_file.name.lower().endswith(valid_ext):
        raise AppError(f"Unsupported file type. Please upload one of: {', '.join(valid_ext)}")

    try:
        from PIL import Image
        import io
        Image.open(io.BytesIO(uploaded_file.getvalue())).verify()
    except Exception as e:
        raise AppError("The uploaded file could not be read as a valid image (it may be corrupted).",
                        technical=str(e))
