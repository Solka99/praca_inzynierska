import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from kanji_evaluation import evaluate_kanji, load_image_as_np


# --------------------------
# Helper: prepare ghost background
# --------------------------

def prepare_ghost_background(template_path: str, size=(256, 256), alpha: int = 80):
    """
    Tworzy delikatne, półprzezroczyste tło kanji ("ghost"),
    po którym użytkownik może odrysowywać znak.
    """

    # 1. Wczytaj i konwertuj do grayscale
    img = Image.open(template_path).convert("L")

    # 2. Resize do wielkości canvasu
    img = img.resize(size, Image.LANCZOS)

    # 3. Autokontrast, żeby znak był równy
    img = ImageOps.autocontrast(img)

    # 4. Jasnoszare kanji (duch)
    arr = np.array(img).astype(np.float32)
    arr = 180 + (arr / 255.0) * (255 - 180)     # jasne 180–255
    arr = arr.clip(0, 255).astype(np.uint8)
    img_gray = Image.fromarray(arr)

    # 5. Kanał alpha
    alpha_channel = Image.new("L", size, alpha)  # 80 = delikatne 30% krycia

    # 6. Składamy RGBA
    ghost_rgba = Image.merge("RGBA", (img_gray, img_gray, img_gray, alpha_channel))

    return ghost_rgba  # zwracamy jako numpy array


# --------------------------
# UI Page
# --------------------------

st.title("✏️ Kanji Practice – Ćwiczenie pisania znaków")

st.write("""
Ten moduł pozwala ćwiczyć ręczne odrysowywanie znaków kanji.
Tłem obszaru rysowania jest **delikatnie wyszarzony wzór (ghost kanji)**.
Po narysowaniu znaku możesz uzyskać ocenę opartą na metrykach podobieństwa.
""")

# --------------------------
# Wybór kanji
# --------------------------

kanji_list = ["木", "日", "人", "大", "水", "山", "口", "心"]  # przykład — dodaj własne
kanji_selected = st.selectbox("Wybierz znak do ćwiczenia:", kanji_list)

template_path = "app/test_images/ushi_template.png"

# Template do ewaluacji (normalny obraz)
template_eval = load_image_as_np(template_path)


# Przygotowanie tła — ghost kanji
ghost_bg = prepare_ghost_background(template_path, size=(256, 256), alpha=80)


st.subheader(f"Napisz znak: {kanji_selected}")

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=4,
    stroke_color="black",
    # background_image=ghost_bg,   # <<< TU JEST GHOST KANJI
    update_streamlit=True,
    height=256,
    width=256,
    drawing_mode="freedraw",
    key="kanji_practice_canvas",
)

# --------------------------
# Ewaluacja
# --------------------------

if st.button("🔍 Oceń mój zapis"):
    if np.all(canvas_result.image_data == 0):
        st.warning("Najpierw narysuj znak 🙂")
    else:
        user_img = canvas_result.image_data

        # if user_img.dtype != np.uint8:
        #     user_img = user_img.astype("uint8")
        # KLUCZOWA ZMIANA:
        if user_img.dtype != np.uint8:
            user_img = (user_img * 255).astype("uint8")


        plt.imshow(user_img, cmap='gray')
        plt.savefig("aaaaa.png")  # save instead of showing

        print('shown')

        result = evaluate_kanji(user_img, template_eval)
        print(result)
        st.markdown("### Wynik oceny")

        st.metric("Ocena końcowa", f"{result['final_score']:.1f} / 100")

        st.write("**Szczegółowe metryki:**")
        st.write(f"- SSIM: `{result['ssim']:.3f}`")
        st.write(f"- IoU: `{result['iou']:.3f}`")
        st.write(f"- Hu score: `{result['hu_score']:.3f}`")
        st.write(f"- Chamfer score: `{result['chamfer_score']:.3f}`")

