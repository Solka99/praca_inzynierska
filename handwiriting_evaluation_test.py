import cv2

from app.kanji_evaluation import evaluate_kanji, load_image_as_np
import os
from pathlib import Path


def take_all_images_from_folder(folder_name: str):
    fpaths_in_folder=[]
    etl_dir = Path().resolve() / "data" / "ETL8G" / folder_name
    for fname in os.listdir(etl_dir):
        if fname.endswith(".png"):
            fpath = os.path.join(etl_dir, fname)
            fpaths_in_folder.append(fpath)

    return fpaths_in_folder[:956]

path_list_01 = take_all_images_from_folder("ETL8G_12_unpack")
path_list_templates = take_all_images_from_folder("ETL8G_33_unpack")
scores=[]
for path_img, path_template in zip(path_list_01, path_list_templates):
    img = load_image_as_np(path_img)
    template = load_image_as_np(path_template)
    score=evaluate_kanji(img,template)
    scores.append(score)

print(scores)