import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np, cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity as ssim

def show_image(otsu_thresh):
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("Otsu’s Thresholding")
    plt.axis('off')
    plt.savefig("otsu_threshold_result.png")  # save instead of showing
    print("Saved Otsu result to otsu_threshold_result.png")


def preprocess_kanji(pil_img):
    """Binarizacja
    
    Każdy piksel zamieniamy na 0 lub 255:
    
    0 → tło (czarne),
    
    255 → kreska (biała).
    
    Robimy to automatycznie przez tzw. progowanie Otsu (z cv2.threshold),
    które samodzielnie wybiera próg jasności między tłem a rysunkiem."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale for Otsu

    ret, otsu_thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Calculated Otsu threshold value:", ret)
    show_image(otsu_thresh)

    # 3️⃣ Znalezienie obszaru, gdzie faktycznie jest znak (bounding box)
    ys, xs = np.where(otsu_thresh > 0)  # współrzędne wszystkich "białych" pikseli
    if len(xs) == 0 or len(ys) == 0:
        print("Brak rysunku na obrazie!")
        # nic nie narysowano → zwróć pusty obraz
        return Image.fromarray(np.zeros((64, 64), np.uint8)) # 64 może być do zmiany w zależności od wielokości rysunku

    # współrzędne narożników znaku
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # wycinamy tylko obszar z rysunkiem
    cropped = otsu_thresh[y1:y2 + 1, x1:x2 + 1]

    # 5️⃣ Dodanie marginesu wokół znaku (centrowanie)
    h, w = cropped.shape
    pad = int(0.2 * max(h, w))  # 20% marginesu
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    canvas[pad:pad + h, pad:pad + w] = cropped

    # 6️⃣ Zmiana rozmiaru do np. 64x64
    resized = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_NEAREST)

    # 7️⃣ Zwracamy gotowy obraz
    return Image.fromarray(resized)



def _iou(a,b):
    inter = np.logical_and(a>0, b>0).sum()
    union = np.logical_or(a>0, b>0).sum()
    return 0.0 if union==0 else inter/union

def _chamfer(a,b):
    # odległość Chamfera na szkielecie (niżej lepiej)
    sk_a = skeletonize(a>0).astype(np.uint8)
    sk_b = skeletonize(b>0).astype(np.uint8)
    # mapa odległości do najbliższej kreski
    da = cv2.distanceTransform(255 - sk_a*255, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(255 - sk_b*255, cv2.DIST_L2, 3)
    # średnia odległość w obie strony
    ca = (da[sk_b>0]).mean() if (sk_b>0).any() else 999
    cb = (db[sk_a>0]).mean() if (sk_a>0).any() else 999
    return float((ca+cb)/2)

def score_quality(user_pil: Image.Image, template_pil: Image.Image):
    U = preprocess_kanji(user_pil); T = preprocess_kanji(template_pil)
    # SSIM (0..1, wyżej lepiej)
    ssim_val = ssim(U, T, data_range=255)
    # IoU (0..1, wyżej lepiej)
    iou_val = _iou(U, T)
    # Chamfer (px, niżej lepiej) -> przeskaluj do 0..1
    ch = _chamfer(U, T)
    ch_norm = np.clip(1.0 - ch/10.0, 0.0, 1.0)  # 0 od ~10px średniej odchyłki
    # Hu moments (niżej lepiej) -> przeskaluj
    huU = cv2.HuMoments(cv2.moments(U>0)).ravel()
    huT = cv2.HuMoments(cv2.moments(T>0)).ravel()
    hu_dist = float(np.linalg.norm(huU - huT))
    hu_norm = np.clip(1.0 - hu_dist/5.0, 0.0, 1.0)

    # Złożony wynik 0..100 (dostosuj wagi pod swój zbiór)
    final01 = 0.4*ssim_val + 0.3*ch_norm + 0.2*iou_val + 0.1*hu_norm
    score = float(final01*100.0)

    # Proste wskazówki
    tips = []
    if ch > 4: tips.append("Linie są w innych miejscach (spróbuj wycentrować znak).")
    if iou_val < 0.6: tips.append("Za małe pokrycie wzorca – popraw proporcje/długości kresek.")
    if ssim_val < 0.7: tips.append("Kształt różni się lokalnie – popracuj nad kątem kresek.")
    return score, {"ssim": ssim_val, "iou": iou_val, "chamfer_px": ch, "hu_norm": hu_norm}, tips




from PIL import Image
template_img = Image.open("C:\\Users\\alicj\\OneDrive\\Desktop\\kanji moje.png")
user_img = Image.open("C:\\Users\\alicj\\OneDrive\\Desktop\\user_kanji.png")

# processed = preprocess_kanji(img)
# processed.show()

# przykład

# user_processed = preprocess_kanji(user_img)
# template_processed = preprocess_kanji(img)

score, metrics, tips = score_quality(user_img, template_img)

print(f"Jakość wykonania: {score:.1f}/100")
print("Szczegóły:", metrics)
print("Wskazówki:", tips)

