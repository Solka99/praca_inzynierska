# kanji_evaluation.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import distance_transform_edt
from PIL import Image
import matplotlib.pyplot as plt


# ---------- 1. Prosty preprocessing ----------

def preprocess_image(img: np.ndarray, size=(128, 128)) -> np.ndarray:
    """
    Normalizuje obraz:
    - konwersja do szarości
    - adaptacyjny wybór BINARY vs BINARY_INV
    - resize
    - wynik: binarka 0/1 (0 = tło, 1 = kreska)
    """
    # 1. Szarość
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # 2. Threshold w dwóch wersjach
    _, bin_normal = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, bin_inv = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    bin_normal01 = (bin_normal > 0).astype(np.uint8)
    bin_inv01 = (bin_inv > 0).astype(np.uint8)

    # 3. Wybierz tę binarkę, która ma rozsądny udział "kresek"
    FG_TARGET = 0.15  # 15% pikseli jako kreski – możesz potem zmienić
    cand = [
        (bin_normal01, abs(bin_normal01.mean() - FG_TARGET)),
        (bin_inv01, abs(bin_inv01.mean() - FG_TARGET)),
    ]
    best_bin = min(cand, key=lambda x: x[1])[0]

    # 4. Resize do wspólnego rozmiaru
    best_resized = cv2.resize(best_bin, size, interpolation=cv2.INTER_NEAREST)

    return best_resized



# ---------- 2. Metryki podobieństwa ----------

def compute_ssim(user_bin: np.ndarray, template_bin: np.ndarray) -> float:
    # SSIM potrzebuje wartości 0–255
    u = (user_bin * 255).astype(np.uint8)
    t = (template_bin * 255).astype(np.uint8)
    score = ssim(u, t, data_range=255)
    return float(score)  # ~[0,1]


def compute_iou(user_bin: np.ndarray, template_bin: np.ndarray) -> float:
    intersection = np.logical_and(user_bin == 1, template_bin == 1).sum()
    union = np.logical_or(user_bin == 1, template_bin == 1).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)  # [0,1]


def compute_chamfer_score(user_bin: np.ndarray, template_bin: np.ndarray) -> float:
    """
    Chamfer distance – im mniejsza tym lepiej.
    Zwracamy przeskalowany „score” w [0,1], gdzie 1 = idealne dopasowanie.
    Prosta wersja: średnia z odległości user→template i template→user.
    """
    # Tło = 0, kreski = 1 → chcemy odległość od kresek
    # distance_transform_edt liczy odległość od zer, więc odwracamy
    dt_template = distance_transform_edt(1 - template_bin)
    dt_user = distance_transform_edt(1 - user_bin)

    # punkty kresek
    user_points = user_bin == 1
    template_points = template_bin == 1

    if not user_points.any() or not template_points.any():
        return 0.0

    # user -> template
    d1 = dt_template[user_points].mean()
    # template -> user
    d2 = dt_user[template_points].mean()

    chamfer_dist = (d1 + d2) / 2.0  # im mniejsza tym lepiej

    # Zamiana na „score” w (0,1].
    # Prosta funkcja monotoniczna (możesz później dostroić współczynnik 5.0).
    score = np.exp(-0.5 * chamfer_dist)
    return float(score)


def compute_hu_score(user_bin: np.ndarray, template_bin: np.ndarray) -> float:
    """
    Hu moments – porównujemy kształt globalny.
    Liczymy log10(|Hu|), potem różnicę i zamieniamy na score w [0,1].
    """
    # OpenCV oczekuje 0–255
    u = (user_bin * 255).astype(np.uint8)
    t = (template_bin * 255).astype(np.uint8)

    moments_u = cv2.moments(u)
    moments_t = cv2.moments(t)

    hu_u = cv2.HuMoments(moments_u).flatten()
    hu_t = cv2.HuMoments(moments_t).flatten()

    # Skala log – standardowy trik przy Hu
    hu_u_log = -np.sign(hu_u) * np.log10(np.abs(hu_u) + 1e-15)
    hu_t_log = -np.sign(hu_t) * np.log10(np.abs(hu_t) + 1e-15)

    diff = np.linalg.norm(hu_u_log - hu_t_log)

    # Zamiana „im mniejsza różnica tym lepiej” na [0,1]
    # Znowu prosta funkcja wygaszająca
    score = np.exp(-0.5 * diff)
    return float(score)


# ---------- 3. Łączymy metryki w wynik 0–100 ----------

def aggregate_score(ssim_score, iou_score, chamfer_score, hu_score) -> float:
    """
    Prosta ważona średnia.
    Wagi możesz później zmienić eksperymentalnie.
    """
    # w_ssim = 0.35
    w_ssim = 1
    # w_iou = 0.35
    w_iou = 0.0
    w_chamfer = 0.0
    # w_hu = 0.15
    w_hu = 0.0

    total = w_ssim + w_iou + w_chamfer + w_hu
    score_01 = (w_ssim * ssim_score +
                w_iou * iou_score +
                w_chamfer * chamfer_score +
                w_hu * hu_score) / total

    return float(max(0.0, min(1.0, score_01)) * 100.0)

def aggregate_score_2(ssim_score, iou_score, chamfer_score, hu_score) -> float:
    """
    Wersja uproszczona:
    - ignorujemy Chamfer w końcowej ocenie (na razie),
    - większa waga dla SSIM i Hu,
    - IoU jako niżej ważony dodatek,
    - na końcu nieliniowe rozciągnięcie skali.
    """
    w_ssim = 0.6
    w_hu = 0.25
    w_iou = 0.15
    # chamfer na razie nie używany
    # w_chamfer = 0.0

    s_lin = (
        w_ssim * ssim_score +
        w_hu * hu_score +
        w_iou * iou_score
    ) / (w_ssim + w_hu + w_iou)

    # zabezpieczenie
    s_lin = max(0.0, min(1.0, s_lin))

    # lekkie "podciągnięcie" wyniku
    gamma = 0.7
    s_final = (s_lin ** gamma) * 100.0

    return float(s_final)


def evaluate_kanji(user_img: np.ndarray, template_img: np.ndarray) -> dict:
    """
    Główna funkcja:
    - przyjmuje obrazy (np. z pliku albo z canvasu),
    - robi preprocessing,
    - liczy metryki i końcowy wynik.
    """
    user_bin = preprocess_image(user_img)
    plt.imshow(user_bin)
    plt.axis('off')  # Turn off axis labels
    plt.show()

    plt.savefig("bbbb.png")  # save instead of showing

    template_bin = preprocess_image(template_img)

    s = compute_ssim(user_bin, template_bin)
    i = compute_iou(user_bin, template_bin)
    c = compute_chamfer_score(user_bin, template_bin)
    h = compute_hu_score(user_bin, template_bin)

    final_score = aggregate_score(s, i, c, h)

    print({
        "ssim": s,
        "iou": i,
        "chamfer_score": c,
        "hu_score": h,
        "final_score": final_score
    }
)
    return {
        "ssim": s,
        "iou": i,
        "chamfer_score": c,
        "hu_score": h,
        "final_score": final_score
    }


# ---------- 4. Pomocnicza funkcja do wczytania obrazu z pliku ----------

def load_image_as_np(path: str) -> np.ndarray:
    """
    Wczytuje obraz z pliku (PNG/JPG) jako numpy array (H, W, C).
    Użyteczne np. dla wzorców.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)
