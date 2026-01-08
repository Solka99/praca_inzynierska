from streamlit_drawable_canvas import st_canvas
def render_canvas():
    canvas_result = st_canvas(
        fill_color="#0935b8",
        stroke_width=5,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=256,
        width=276,
        drawing_mode="freedraw",
        key="canvas",
    )
    return canvas_result