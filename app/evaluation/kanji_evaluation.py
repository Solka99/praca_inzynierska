import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image



def preprocess_min(img: np.ndarray, size=(128, 127)) -> np.ndarray:
    print("shape:", img.shape)
    if img.ndim == 3 and img.shape[2] == 4:
        print("alpha min/max:", img[:, :, 3].min(), img[:, :, 3].max())


    if img.ndim == 3 and img.shape[2] == 4:
        print('alfa')
        rgb = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        white = np.ones_like(rgb) * 255.0
        img = (rgb * alpha[..., None] + white * (1 - alpha[..., None])).astype(np.uint8)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bin255 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bin01 = (bin255 > 0).astype(np.uint8)

    return bin01





def compute_chamfer_score(user_bin: np.ndarray,
                          template_bin: np.ndarray,
                          sigma_px: float = 5.0) -> float:

    u = (user_bin > 0)
    t = (template_bin > 0)

    if not u.any() or not t.any():
        return 0.0


    dt_t = distance_transform_edt(~t)
    dt_u = distance_transform_edt(~u)

    d1 = dt_t[u].mean()
    d2 = dt_u[t].mean()  #
    d = 0.5 * (d1 + d2)

    # print("chamfer score", d)

    score = np.exp(-(d / sigma_px) ** 2)
    return float(score)




def aggregate_score(ssim_score, chamfer_score) -> float:
    w_ssim = 0
    w_chamfer = 1

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

    if bin01.mean() > 0.5:
        bin01 = 1 - bin01
    return bin01


def skeleton01(bin01: np.ndarray) -> np.ndarray:
    sk = cv2.ximgproc.thinning((bin01 * 255).astype(np.uint8))
    return (sk > 0).astype(np.uint8)

def count_components(bin01: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(bin01.astype(np.uint8))
    return int(n - 1)

def count_endpoints(skel01: np.ndarray) -> int:
    s = skel01.astype(np.uint8)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    neigh = cv2.filter2D(s, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return int(np.logical_and(s == 1, neigh == 1).sum())

def bridge_gaps(bin01: np.ndarray, k: int = 3, it: int = 1) -> np.ndarray:

    b = (bin01.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=it)
    return (b > 0).astype(np.uint8)


def stroke_penalty(user_bin01: np.ndarray, tmpl_bin01: np.ndarray) -> float:

    u = bridge_gaps(user_bin01, k=4, it=1)
    t = bridge_gaps(tmpl_bin01, k=4, it=1)
    u_sk = skeleton01(u)
    t_sk = skeleton01(t)

    u_cc = count_components(u_sk)
    t_cc = count_components(t_sk)

    u_end = count_endpoints(u_sk)
    t_end = count_endpoints(t_sk)

    # różnice względne (0=idealnie)
    cc_diff = abs(u_cc - t_cc) / max(t_cc, 1)
    end_diff = abs(u_end - t_end) / max(t_end, 1)

    penalty = 1.0 - (0.5 * cc_diff + 0.5 * end_diff)
    return float(np.clip(penalty, 0.0, 1.0))


def evaluate_kanji(user_img: np.ndarray, template_img: np.ndarray) -> dict:



    user_bin = preprocess_min(user_img, (128, 127))
    user_bin = ensure_black_strokes(user_bin)

    template_bin = preprocess_min(template_img, (128, 127))
    template_bin = ensure_black_strokes(template_bin)


    # cv2.imwrite("before.png", user_bin * 255)
    user_bin = center_by_bbox(user_bin)

    # cv2.imwrite("after_centered.png", user_bin * 255)
    template_bin = center_by_bbox(template_bin)

    pen = stroke_penalty(user_bin, template_bin)

    # print("stroke penalty:", pen)

    s=1
    c = compute_chamfer_score(user_bin, template_bin)

    final_score = aggregate_score(s, c)
    final_score=final_score*pen


    # print({
    #     "chamfer_score": c,
    #     "stroke_penalty": pen,
    #     "final_score": final_score
    # }
# )
    return {
        "chamfer_score": c,
        "stroke_penalty": pen,
        "final_score": final_score
    }



def load_image_as_np(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def preprocess_whole(img: np.ndarray) -> np.ndarray:
    img= preprocess_min(img,(128, 127))
    img=ensure_black_strokes(img)
    img=center_by_bbox(img)
    return img

