import numpy as np
import torch
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from kanji_info.kanji_alive_connection import get_kanji_info, show_kanji
from classification.model_architecture import load_model, load_labels, predict
import torchvision.transforms as T
import matplotlib.pyplot as plt
from drawing.canvas import render_canvas

# MODEL_PATH = "best_model3.pt"
INPUT_SIZE = 64
CLASS_TO_KANJI = load_labels()



def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    img = img_pil.convert("L")
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return t


def pil_to_tensor2(img_pil):
    img = img_pil.convert("L")
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor




def main():
    st.set_page_config(page_title="Kanji — minimal demo", page_icon="🈶", layout="centered")
    st.title("Rysuj znak i rozpoznaj")

    model = load_model()

    col_draw, col_animation, col_info, col_examples = st.columns(4)


    canvas = render_canvas()
    if np.any(canvas.image_data != 0) and st.button("Rozpoznaj", type="primary", width=256):
        plt.imshow(canvas.image_data, cmap='gray')
        img = Image.fromarray(canvas.image_data[:, :, :3].astype("uint8"))

        kanji_list, prob_list = predict(model, img)
        st.success(f"Rozpoznano znak: **{kanji_list[0]}**")



        print(prob_list[0])
        data = get_kanji_info(kanji_list[0])
        show_kanji(data)


        # preprocess_kanji(img)




if __name__ == "__main__":
    main()
