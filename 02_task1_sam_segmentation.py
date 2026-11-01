import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

torch.set_num_threads(os.cpu_count() or 4)

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=12,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=1000
)

def select_egg_mask(masks, img_shape):
    h, w = img_shape[:2]
    img_area = h * w
    best_mask = None
    best_score = -1.0

    for m in masks:
        seg = m["segmentation"]
        area = m["area"]
        if area < 0.02 * img_area or area > 0.85 * img_area:
            continue

        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (cv2.contourArea(cnt) / (perimeter * perimeter))
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0

        score = (circularity * 0.5) + (solidity * 0.5) + (m["predicted_iou"] * 0.3)
        if score > best_score:
            best_score = score
            best_mask = seg

    if best_mask is None and len(masks) > 0:
        sorted_by_area = sorted(masks, key=lambda x: x["area"], reverse=True)
        best_mask = sorted_by_area[0]["segmentation"]

    if best_mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    return (best_mask.astype(np.uint8)) * 255

def process_directory(img_dir, out_mask_dir):
    os.makedirs(out_mask_dir, exist_ok=True)
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))

    for path in tqdm(img_paths, desc=f"Processing {img_dir}"):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        
        orig_h, orig_w = bgr.shape[:2]
        max_dim = 1024
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)
            bgr_resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            bgr_resized = bgr

        rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
        generated_masks = mask_generator.generate(rgb)
        egg_mask = select_egg_mask(generated_masks, rgb.shape)
        
        if egg_mask.shape[:2] != (orig_h, orig_w):
            egg_mask = cv2.resize(egg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        file_name = os.path.splitext(os.path.basename(path))[0] + ".png"
        cv2.imwrite(os.path.join(out_mask_dir, file_name), egg_mask)

process_directory("data/dev/images", "data/dev/masks_task1")
process_directory("data/eval/images", "data/eval/masks_task1")