from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image, ImageOps
import streamlit as st
import pandas as pd
import numpy as np
from evaluation.kanji_evaluation import evaluate_kanji, load_image_as_np, preprocess_min
from classification.model_architecture import load_model, predict
import torchvision.transforms as T
from drawing.canvas import render_canvas


st.title("Kanji Practice – Ćwiczenie pisania znaków")

path_to_labels = Path().resolve()/"meta.csv"
labels_df = pd.read_csv(path_to_labels)
labels = labels_df["char"]
labels = labels[:956].tolist()

def create_path_label_list():
    path_label_list = []
    etl_dir = Path().resolve() / "data" / "ETL8G"
    for folder in os.listdir(etl_dir):
        if "unpack" in folder:
            folder_path = os.path.join(etl_dir, folder)

            for fname in os.listdir(folder_path):
                if fname.endswith(".png"):
                    fpath = os.path.join(folder_path, fname)
                    idx = int(os.path.splitext(fname)[0])
                    label = idx % 956
                    path_label_list.append((fpath, label))
    return path_label_list

def find_path_to_image_by_index():
    path_to_templates= Path().resolve() / "data" / "ETL8G" / "ETL8G_33_unpack"
    # path_to_templates= Path().resolve().parent / "data" / "ETL8G" / "ETL8G_33_unpack"

    path_label_df = pd.DataFrame(labels, columns=["kanji"])

    index_by_kanji = np.where(path_label_df["kanji"]==kanji_selected)[0][0]
    if index_by_kanji<10:
        path_to_specific_template = path_to_templates / f'0000{index_by_kanji}.png'
    elif index_by_kanji>=10 and index_by_kanji<100:
        path_to_specific_template = path_to_templates / f'000{index_by_kanji}.png'
    else:
        path_to_specific_template = path_to_templates / f'00{index_by_kanji}.png'

    return path_to_specific_template
# print(path_label_list[956])

model = load_model()

st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        background-color: #697a5a;  /* background */
        color: white;               /* text */
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
kanji_selected = st.selectbox("Wybierz znak do ćwiczenia:", labels)


template_path = find_path_to_image_by_index()



template_eval = load_image_as_np(template_path)


col_draw, col_example= st.columns(2)
with col_example:
    st.image(~template_eval, caption="Wzór", width=256)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    results_slot = st.empty()
with col_draw:
    path_label_df = pd.DataFrame(labels, columns=["kanji"])

    canvas_result = render_canvas()


    if st.button("Oceń mój zapis", type="primary", width=276):
        if np.all(canvas_result.image_data == 0):
            st.warning("Najpierw narysuj znak 🙂")
        else:
            user_img = canvas_result.image_data

            if user_img.dtype != np.uint8:
                user_img = (user_img * 255).astype("uint8")


            plt.imshow(user_img, cmap='gray')
            # plt.savefig("aaaaa.png")


            result = evaluate_kanji(user_img, template_eval)
            # print(result)
            st.markdown("### Wynik oceny")

            st.metric("Ocena końcowa", f"{result['final_score']:.1f} / 100")

            st.write("**Szczegółowe metryki: [Im metryki są bliższe 1 tym lepiej]**")
            st.write(f"- Chamfer score: {result['chamfer_score']:.3f}")
            st.write(f"- Stroke penalty: {result['stroke_penalty']:.3f}")

            img = Image.fromarray(user_img[:, :, :3])


            with col_example:
                kanji_list, prob_list = predict(model, img)
                with results_slot.container():
                    st.subheader("Rozpoznanie")
                    print(kanji_selected)
                    print(kanji_list)
                    if kanji_selected in kanji_list[0]:
                        st.success(f'Model rozpoznał kanji {kanji_selected} !  \n Dobra robota!')
                    else:
                        st.warning("Model nie rozpoznał kanji.  \n Spróbuj narysować jeszcze raz.")


