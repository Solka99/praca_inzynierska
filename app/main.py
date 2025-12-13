import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from pathlib import Path
import json
# from user_accuracy import preprocess_kanji
from app_helpers.kanji_alive_connection import get_kanji_info, show_kanji

import torch.nn as nn
import torch.nn.functional as F

class CNN_1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 64x64 -> 62x62 (no padding)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0, stride=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 62→31

        # 31x31 -> 15x15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 15x15 -> 7x7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.drop  = nn.Dropout(0.2)
        self.fc    = nn.Linear(128 * 7 * 7, num_classes)


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)      # [B, 128*7*7]
        x = self.drop(x)
        return self.fc(x)

def load_labels():

    path_to_labels = Path().resolve() / "data" / "ETL8G" / "ETL8G_01_unpack" / "meta.csv"
    labels_df = pd.read_csv(path_to_labels)

    labels = labels_df["char"].tolist()

    all_labels = labels.copy()
    labels_copy = all_labels.copy()

    for i in range(31):
        labels_copy_2 = labels_copy.copy()
        all_labels.extend(labels_copy_2)

    all_labels.extend(labels_copy[:956])
    return all_labels

# --- KONFIG (dostosuj) ---
MODEL_PATH = "C:\\Users\\alicj\\PycharmProjects\\KanjiRecognitionModel\\best_model.pt"   # ścieżka do Twojego modelu
INPUT_SIZE = 64                      # rozmiar wejścia modelu
CLASS_TO_KANJI = load_labels()  # mapowanie indeks->kanji (użyj własnej listy)


@st.cache_resource
def load_model():
    # Jeśli masz torchscript -> torch.jit.load, inaczej torch.load
    model = CNN_1(956)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    # grayscale, odwrócenie (jeśli rysujesz czarne linie na białym tle, możesz usunąć invert)
    img = img_pil.convert("L")
    img = ImageOps.invert(img)   # usuń jeśli niepotrzebne
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5      # prosta normalizacja
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return t


def predict(model, img_pil):
    kanji_list = []
    probs_list = []
    with torch.no_grad():
        x = pil_to_tensor(img_pil)
        logits = model(x) #surowe wyjścia sieci neuronowej
        probs = F.softmax(logits, dim=1).cpu().numpy()[0] #Softmax to funkcja, która przekształca liczby na rozkład prawdopodobieństwa (suma = 1)
        # idx = int(np.argmax(probs))
        idx_array = np.argsort(probs)
        idx_array = idx_array[-3:]
        idx_array = idx_array[::-1]
        print(len(idx_array))
        print(type(idx_array))
        for idx in idx_array:
            kanji = CLASS_TO_KANJI[idx] if idx < len(CLASS_TO_KANJI) else f"cls_{idx}"
            kanji_list.append(kanji)
            probs_list.append(float(probs[idx]))
        # kanji = CLASS_TO_KANJI[idx] if idx < len(CLASS_TO_KANJI) else f"cls_{idx}"
        return kanji_list, probs_list


def save_kanji_to_json(kanji):
    print(kanji)
    print(type(kanji))
    FILE_PATH = "C:\\Users\\alicj\\PycharmProjects\\KanjiRecognitionModel\\app\\saved_kanji.json"
    data = json.load(open(FILE_PATH))
    print(type(data['saved_kanji']))
    data['saved_kanji'] = data['saved_kanji'].append(kanji)
    print("data", data)
    print('in')

    try:
        print('ala')
        with open(FILE_PATH, "w") as f:
            json.dump(data, f)
            print('saved')

    except FileNotFoundError:
        print("Error: The file 'data.json' was not found.")



def main():
    st.set_page_config(page_title="Kanji — minimal demo", page_icon="🈶")
    st.title("Rysuj znak i rozpoznaj")

    model = load_model()

    # Płótno do rysowania
    canvas = st_canvas(
        fill_color="#0935b8",
        stroke_width=4,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=256,
        width=256,
        drawing_mode="freedraw",
        key="canvas",
    )
    if np.any(canvas.image_data != 0) and st.button("Rozpoznaj"):
        # canvas.image_data: (H,W,4) RGBA -> bierzemy RGB
        img = Image.fromarray(canvas.image_data[:, :, :3].astype("uint8"))
        kanji_list, prob_list = predict(model, img)
        for k,p in zip(kanji_list, prob_list):
            st.success(f"Rozpoznano: **{k}** (pewność {p:.2f})")
        st.image(img, caption="Twój rysunek", width=128)


        print(prob_list[0])
        data = get_kanji_info(kanji_list[0])
        show_kanji(data)
        # st.json(data)  # szybki podgląd całej odpowiedz


        # preprocess_kanji(img)

    if st.button("💾 Zapisz znak"):
        st.write('kliknięte')
        print('klikniete')
        #użytkownik jest proszony o wybranie kanji z 3 rozpoznanych opcji
        save_kanji_to_json('愛')
        st.success(f"Znak został zapisany!")
        print('jej')




if __name__ == "__main__":
    main()
