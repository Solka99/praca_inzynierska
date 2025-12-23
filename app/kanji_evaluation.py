# kanji_evaluation.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import distance_transform_edt
from PIL import Image
import matplotlib.pyplot as plt



def preprocess_min(img: np.ndarray, size=(128, 127)) -> np.ndarray:
    """
    Minimalny preprocessing:
    1) jeśli RGBA -> spłaszczenie na białe tło
    2) grayscale
    3) resize do (width=128, height=127)
    4) Otsu + BINARY_INV (czarne -> 1, białe -> 0)
    Wynik: uint8 0/1, shape (127, 128)
    """

    # 1) Jeśli obraz ma alfę (RGBA), zrób z niego normalny obraz na białym tle
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = img[:, :, :3].astype(np.float32)          # kolory
        alpha = img[:, :, 3].astype(np.float32) / 255.0 # 0..1
        white = np.ones_like(rgb) * 255.0               # białe tło
        img = (rgb * alpha[..., None] + white * (1 - alpha[..., None])).astype(np.uint8)

    # 2) Skala szarości
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 3) Resize (OpenCV: size = (width, height))
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4) Binarizacja: czarne kreski -> 1, białe tło -> 0
    _, bin255 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5) 0/255 -> 0/1
    bin01 = (bin255 > 0).astype(np.uint8)

    return bin01




# ---------- 2. Metryki podobieństwa ----------

def compute_ssim(user_bin: np.ndarray, template_bin: np.ndarray) -> float:
    # SSIM potrzebuje wartości 0–255
    u = (user_bin * 255).astype(np.uint8)
    t = (template_bin * 255).astype(np.uint8)
    score = ssim(u, t, data_range=255)
    return float(score)  # ~[0,1]



def compute_chamfer_score(user_bin: np.ndarray,
                          template_bin: np.ndarray,
                          sigma_px: float = 10.0) -> float:
    """
    Chamfer score w [0,1], gdzie 1 = idealnie.
    - Liczymy symetryczny Chamfer distance w pikselach (im mniejszy, tym lepiej).
    - Zamieniamy distance -> score funkcją exp(-(d/sigma)^2).
      sigma_px dobierz tak, żeby "OK" rysunki dawały sensowny score (np. 0.6-0.9).
    """

    # 1) Ujednolicenie: True = kreska
    u = (user_bin > 0)
    t = (template_bin > 0)

    if not u.any() or not t.any():
        return 0.0

    # 2) Distance transform: odległość do najbliższej kreski w drugim obrazie
    # DT liczy odległość do zer, więc robimy: kreski -> 0, tło -> 1
    dt_t = distance_transform_edt(~t)  # odległość do kresek template
    dt_u = distance_transform_edt(~u)  # odległość do kresek user

    # 3) Symetryczny Chamfer (w px)
    d1 = dt_t[u].mean()  # user -> template
    d2 = dt_u[t].mean()  # template -> user
    d = 0.5 * (d1 + d2)

    # print("chamfer score", d)

    # 4) Distance -> score (0..1]
    score = np.exp(-(d / sigma_px) ** 2)
    return float(score)



# ---------- 3. Łączymy metryki w wynik 0–100 ----------

def aggregate_score(ssim_score, chamfer_score) -> float:
    """
    Prosta ważona średnia.
    Wagi możesz później zmienić eksperymentalnie.
    """
    w_ssim = 0.5
    w_chamfer = 0.5

    total = w_ssim + w_chamfer
    score_01 = (w_ssim * ssim_score +
                w_chamfer * chamfer_score) / total

    return float(max(0.0, min(1.0, score_01)) * 100.0)


def center_by_bbox(bin01: np.ndarray) -> np.ndarray:
    ys, xs = np.where(bin01 > 0)
    if len(xs) == 0:
        return bin01

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = bin01[y0:y1+1, x0:x1+1]

    out = np.zeros_like(bin01)
    h, w = crop.shape
    oy = (out.shape[0] - h) // 2
    ox = (out.shape[1] - w) // 2
    out[oy:oy+h, ox:ox+w] = crop
    return out

def ensure_black_strokes(bin01: np.ndarray) -> np.ndarray:
    # chcemy: mało jedynek (kreski), dużo zer (tło)
    # jeśli jest odwrotnie, odwracamy
    if bin01.mean() > 0.5:
        bin01 = 1 - bin01
    return bin01

def has_ximgproc() -> bool:
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning")

def skeleton01(bin01: np.ndarray) -> np.ndarray:
    """
    Zwraca szkielet 0/1 (1-pikselowe kreski).
    Wymaga opencv-contrib-python (cv2.ximgproc.thinning).
    """
    sk = cv2.ximgproc.thinning((bin01 * 255).astype(np.uint8))
    return (sk > 0).astype(np.uint8)

def count_components(bin01: np.ndarray) -> int:
    """Liczba spójnych składowych (bez tła)."""
    n, _ = cv2.connectedComponents(bin01.astype(np.uint8))
    return int(n - 1)

def count_endpoints(skel01: np.ndarray) -> int:
    """
    Endpoint = piksel szkieletu mający dokładnie 1 sąsiada (8-sąsiedztwo).
    Daje sensowny sygnał: więcej bazgrołów -> więcej endpointów,
    brak kresek -> mniej endpointów.
    """
    s = skel01.astype(np.uint8)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    neigh = cv2.filter2D(s, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return int(np.logical_and(s == 1, neigh == 1).sum())

def stroke_penalty(user_bin01: np.ndarray, tmpl_bin01: np.ndarray) -> float:
    """
    Kara za brak/dodatkowe kreski: wynik w [0,1], gdzie 1 = brak kary.
    Jak interpretować penalty

    ~0.8 – 1.0 → liczba/struktura kresek podobna

    ~0.5 – 0.8 → coś nie gra (braki albo dodatkowe kreski)

    <0.5 → mocno różni się „kreskowo”
    """
    if not has_ximgproc():
        # jeśli nie masz opencv-contrib, nie da się prosto zrobić szkieletyzacji
        return 1.0

    u_sk = skeleton01(user_bin01)
    t_sk = skeleton01(tmpl_bin01)

    u_cc = count_components(u_sk)
    t_cc = count_components(t_sk)

    u_end = count_endpoints(u_sk)
    t_end = count_endpoints(t_sk)

    # różnice względne (0=idealnie)
    cc_diff = abs(u_cc - t_cc) / max(t_cc, 1)
    end_diff = abs(u_end - t_end) / max(t_end, 1)

    # prosta kombinacja: endpoints ważniejsze
    penalty = 1.0 - (0.3 * cc_diff + 0.7 * end_diff)
    return float(np.clip(penalty, 0.0, 1.0))


def evaluate_kanji(user_img: np.ndarray, template_img: np.ndarray) -> dict:



    user_bin = preprocess_min(user_img, (128, 127))
    user_bin = ensure_black_strokes(user_bin)

    template_bin = preprocess_min(template_img, (128, 127))
    template_bin = ensure_black_strokes(template_bin)

    # print("user fg%", user_bin.mean())
    # print("tmpl fg%", template_bin.mean())

    user_bin = center_by_bbox(user_bin)
    template_bin = center_by_bbox(template_bin)

    pen = stroke_penalty(user_bin, template_bin)

    # print("stroke penalty:", pen)


    s = compute_ssim(user_bin, template_bin)
    c = compute_chamfer_score(user_bin, template_bin)

    final_score = aggregate_score(s, c)
    final_score=final_score*pen

    # print("user:", user_bin.dtype, user_bin.min(), user_bin.max(), "mean", user_bin.mean())
    # print("tmpl:", template_bin.dtype, template_bin.min(), template_bin.max(), "mean", template_bin.mean())

    # print({
    #     "ssim": s,
    #     "iou": i,
    #     "chamfer_score": c,
    #     "hu_score": h,
    #     "final_score": final_score
    # }
# )
    return {
        "ssim": s,
        "chamfer_score": c,
        "stroke_penalty": pen,
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
