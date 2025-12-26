from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image, ImageOps
import streamlit as st
import pandas as pd
import numpy as np
from kanji_evaluation import evaluate_kanji, load_image_as_np
from model_architecture import load_model, predict
import torchvision.transforms as T

# --------------------------
# UI Page
# --------------------------

st.title("Kanji Practice – Ćwiczenie pisania znaków")

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

# def pil_to_tensor2_debug(img_pil, prefix="debug"):
#     img = img_pil.convert("L")
#     img = img.point(lambda x: 255 - x)
#     # 1️⃣ ZAPIS: po konwersji do grayscale
#     img.save(f"{prefix}_01_gray.png")
#
#     resize = T.Resize((64, 64))
#     img_resized = resize(img)
#
#     # 2️⃣ ZAPIS: po resize (TO JEST KLUCZOWE)
#     img_resized.save(f"{prefix}_02_resized.png")
#
#     to_tensor = T.ToTensor()
#     tensor = to_tensor(img_resized)  # [1,64,64]
#
#     # 3️⃣ ZAPIS: tensor → obraz (PRZED normalizacją)
#     img_from_tensor = T.ToPILImage()(tensor)
#     img_from_tensor.save(f"{prefix}_03_tensor_before_norm.png")
#
#     normalize = T.Normalize(mean=[0.5], std=[0.5])
#     tensor_norm = normalize(tensor)
#
#     # 4️⃣ ZAPIS: tensor → obraz (PO normalizacji)
#     # cofamy normalizację, żeby dało się zapisać
#     tensor_denorm = tensor_norm * 0.5 + 0.5
#     img_after_norm = T.ToPILImage()(tensor_denorm.clamp(0, 1))
#     img_after_norm.save(f"{prefix}_04_tensor_after_norm.png")
#
#     # batch dim
#     return tensor_norm.unsqueeze(0)  # [1,1,64,64]

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
# kanji_list = ["木", "日", "人", "大", "水", "山", "口", "心"]  # przykład — dodaj własne
kanji_selected = st.selectbox("Wybierz znak do ćwiczenia:", labels)


template_path = find_path_to_image_by_index()



# Template do ewaluacji (normalny obraz)
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
    # st.markdown(f"Napisz znak: {kanji_selected}")
    path_label_df = pd.DataFrame(labels, columns=["kanji"])

    canvas_result = st_canvas(
        fill_color="#0935b8",
        stroke_width=5,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=256,
        width=276,
        drawing_mode="freedraw",
        key="canvas_evaluate",
    )

    # --------------------------
    # Ewaluacja
    # --------------------------

    if st.button("Oceń mój zapis", type="primary", width=276):
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
            plt.savefig("aaaaa.png")


            result = evaluate_kanji(user_img, template_eval)
            # print(result)
            st.markdown("### Wynik oceny")

            st.metric("Ocena końcowa", f"{result['final_score']:.1f} / 100")

            st.write("**Szczegółowe metryki: [Im metryki są bliższe 1 tym lepiej]**")
            st.write(f"- SSIM: `{result['ssim']:.3f}`")
            st.write(f"- Chamfer score: `{result['chamfer_score']:.3f}`")
            st.write(f"- Stroke penalty: `{result['stroke_penalty']:.3f}`")

            img = Image.fromarray(user_img[:, :, :3])

            # kanji_list, prob_list = predict(model, img)
            # for k, p in zip(kanji_list, prob_list):
            #     st.success(f"Rozpoznano: **{k}** (pewność {p:.2f})")

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

                    # for k, p in zip(kanji_list, prob_list):
                    #     st.success(f"Rozpoznano: **{k}** (pewność {p:.2f})")

# with col_example:
#     st.image(~template_eval, caption="Wzór", width=256)
#     results_slot = st.empty()