import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import torch
from PIL import Image, ImageOps
import streamlit as st
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)

class CNN_Improved(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 64x64 -> 32x32
        self.block1 = ConvBlock(1, 32)
        # 32x32 -> 16x16
        self.block2 = ConvBlock(32, 64)
        # 16x16 -> 8x8
        self.block3 = ConvBlock(64, 128)

        self.gap  = nn.AdaptiveAvgPool2d(1)   # 8x8 -> 1x1
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)          # [B, 128, 8, 8]
        x = self.gap(x)             # [B, 128, 1, 1]
        x = torch.flatten(x, 1)     # [B, 128]
        x = self.drop(x)
        x = self.fc(x)
        return x

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

MODEL_PATH = "C:\\Users\\alicj\\PycharmProjects\\KanjiRecognitionModel\\best_model3.pt"  # ścieżka do Twojego modelu
INPUT_SIZE = 64                      # rozmiar wejścia modelu
CLASS_TO_KANJI = load_labels()  # mapowanie indeks->kanji (użyj własnej listy)

@st.cache_resource
def load_model():
    # Jeśli masz torchscript -> torch.jit.load, inaczej torch.load
    model = CNN_Improved(956)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def center_pil_by_bbox(img_L: Image.Image, thresh: int = 10) -> Image.Image:
    """
    Centruje obiekt na obrazie PIL w trybie 'L' (grayscale).
    Zakładamy: tło ~0 (ciemne), kreski >0 (jasne) – np. po odwróceniu.

    thresh: próg do wykrycia „tuszu” (żeby ignorować bardzo słaby szum).
    """
    if img_L.mode != "L":
        img_L = img_L.convert("L")

    arr = np.array(img_L, dtype=np.uint8)

    # maska obiektu (kreski)
    ys, xs = np.where(arr > thresh)
    if len(xs) == 0:
        return img_L  # brak obiektu -> bez zmian

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    crop = img_L.crop((x0, y0, x1 + 1, y1 + 1))

    out = Image.new("L", img_L.size, 0)  # czarne tło
    w, h = crop.size
    ox = (out.size[0] - w) // 2
    oy = (out.size[1] - h) // 2
    out.paste(crop, (ox, oy))
    return out
def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    # grayscale, odwrócenie (jeśli rysujesz czarne linie na białym tle, możesz usunąć invert)
    img = img_pil.convert("L")
    img = ImageOps.invert(img)   # usuń jeśli niepotrzebne
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5      # prosta normalizacja
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return t

def pil_to_tensor2_debug(img_pil, prefix="debug"):
    img = img_pil.convert("L")
    img = img.point(lambda x: 255 - x)
    # 1️⃣ ZAPIS: po konwersji do grayscale
    img.save(f"{prefix}_01_gray.png")

    # 2) NOWY KROK: centrowanie bbox (przed resize)
    img_centered = center_pil_by_bbox(img, thresh=10)

    resize = T.Resize((64, 64))
    img_resized = resize(img_centered)

    # 2️⃣ ZAPIS: po resize (TO JEST KLUCZOWE)
    img_resized.save(f"{prefix}_02_resized.png")

    to_tensor = T.ToTensor()
    tensor = to_tensor(img_resized)  # [1,64,64]

    # 3️⃣ ZAPIS: tensor → obraz (PRZED normalizacją)
    img_from_tensor = T.ToPILImage()(tensor)
    img_from_tensor.save(f"{prefix}_03_tensor_before_norm.png")

    normalize = T.Normalize(mean=[0.5], std=[0.5])
    tensor_norm = normalize(tensor)

    # 4️⃣ ZAPIS: tensor → obraz (PO normalizacji)
    # cofamy normalizację, żeby dało się zapisać
    tensor_denorm = tensor_norm * 0.5 + 0.5
    img_after_norm = T.ToPILImage()(tensor_denorm.clamp(0, 1))
    img_after_norm.save(f"{prefix}_04_tensor_after_norm.png")

    # batch dim
    return tensor_norm.unsqueeze(0)  # [1,1,64,64]

def predict(model, img_pil):
    kanji_list = []
    probs_list = []
    with torch.no_grad():
        x = pil_to_tensor2_debug(img_pil)
        logits = model(x) #surowe wyjścia sieci neuronowej
        probs = F.softmax(logits, dim=1).cpu().numpy()[0] #Softmax to funkcja, która przekształca liczby na rozkład prawdopodobieństwa (suma = 1)
        # idx = int(np.argmax(probs))
        idx_array = np.argsort(probs)
        idx_array = idx_array[-3:]
        idx_array = idx_array[::-1]
        for idx in idx_array:
            kanji = CLASS_TO_KANJI[idx] if idx < len(CLASS_TO_KANJI) else f"cls_{idx}"
            kanji_list.append(kanji)
            probs_list.append(float(probs[idx]))
        # kanji = CLASS_TO_KANJI[idx] if idx < len(CLASS_TO_KANJI) else f"cls_{idx}"
        return kanji_list, probs_list