import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from kanji_evaluation import evaluate_kanji, load_image_as_np

# --------------------------
# UI Page
# --------------------------

st.title("✏️ Kanji Practice – Ćwiczenie pisania znaków")

st.write("""
Ten moduł pozwala ćwiczyć ręczne odrysowywanie znaków kanji.
Po narysowaniu znaku możesz uzyskać ocenę opartą na metrykach podobieństwa.
""")

# --------------------------
# Wybór kanji
# --------------------------
path_to_labels = Path().resolve()/ "data" / "ETL8G"/ "ETL8G_01_unpack"/"meta.csv"
labels_df = pd.read_csv(path_to_labels)
labels = labels_df["char"]
labels = labels[:956].tolist()

def create_path_label_list():
    path_label_list = []
    etl_dir = Path().resolve() / "data" / "ETL8G"
    for folder in os.listdir(etl_dir):
        if "unpack" in folder:
            folder_path = os.path.join(etl_dir, folder)

            # go through each png in folder
            for fname in os.listdir(folder_path):
                if fname.endswith(".png"):
                    fpath = os.path.join(folder_path, fname)
                    # take filename without extension
                    idx = int(os.path.splitext(fname)[0])
                    # compute label
                    label = idx % 956
                    path_label_list.append((fpath, label))
    return path_label_list

def find_path_to_image_by_index():
    path_to_templates= Path().resolve() / "data" / "ETL8G" / "ETL8G_33_unpack"

    # path_label_list = create_path_label_list()
    path_label_df = pd.DataFrame(labels, columns=["kanji"])

    index_by_kanji = np.where(path_label_df["kanji"]==kanji_selected)[0][0]
    path_to_specific_template = path_to_templates /f'0000{index_by_kanji}.png' if index_by_kanji<10 else path_to_templates /f'000{index_by_kanji}.png'
    return path_to_specific_template
# print(path_label_list[956])




# kanji_list = ["木", "日", "人", "大", "水", "山", "口", "心"]  # przykład — dodaj własne
kanji_selected = st.selectbox("Wybierz znak do ćwiczenia:", labels)


template_path = find_path_to_image_by_index()



# Template do ewaluacji (normalny obraz)
template_eval = load_image_as_np(template_path)



st.subheader(f"Napisz znak: {kanji_selected}")
path_label_df = pd.DataFrame(labels, columns=["kanji"])

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=5,
    stroke_color="black",
    # background_image=ghost_bg,   # <<< TU JEST GHOST KANJI
    update_streamlit=True,
    height=254,
    width=276,
    drawing_mode="freedraw",
    key="kanji_practice_canvas",
)
st.image(template_eval, caption="Wzór", width=256)
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


        result = evaluate_kanji(user_img, template_eval)
        print(result)
        st.markdown("### Wynik oceny")

        st.metric("Ocena końcowa", f"{result['final_score']:.1f} / 100")

        st.write("**Szczegółowe metryki:**")
        st.write(f"- SSIM: `{result['ssim']:.3f}`")
        st.write(f"- Chamfer score: `{result['chamfer_score']:.3f}`")

