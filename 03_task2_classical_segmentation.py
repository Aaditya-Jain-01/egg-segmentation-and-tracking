import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def segment_egg_classical(bgr_image):
    h, w = bgr_image.shape[:2]
    total_area = h * w

    denoised = cv2.bilateralFilter(bgr_image, 9, 75, 75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)

    l_channel = lab[:, :, 0]
    b_channel = lab[:, :, 2]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    cr_channel = ycrcb[:, :, 1]

    _, otsu_l = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    skin_rule = (cr_channel > 135) & (cr_channel < 175) & (s_channel > 35) & (s_channel < 190)
    skin_mask = (skin_rule.astype(np.uint8)) * 255

    candidate = cv2.bitwise_and(otsu_l, cv2.bitwise_not(skin_mask))
    candidate[v_channel < 70] = 0

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    opened = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel_clean)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_fill)

    im_floodfill = closed.copy()
    mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask_floodfill, (0, 0), 255)
    filled = closed | cv2.bitwise_not(im_floodfill)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    best_cnt = None
    best_score = -1.0
    center_x, center_y = w / 2.0, h / 2.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.02 * total_area or area > 0.80 * total_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_from_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            norm_dist = 1.0 - (dist_from_center / np.sqrt(center_x**2 + center_y**2))
        else:
            norm_dist = 0.5

        score = (area / total_area) * 0.30 + solidity * 0.35 + circularity * 0.20 + norm_dist * 0.15
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is not None:
        cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_fill)

    return final_mask

def process_classical(img_dir, out_mask_dir):
    os.makedirs(out_mask_dir, exist_ok=True)
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))

    for path in tqdm(img_paths, desc=f"Classical Processing {img_dir}"):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        mask = segment_egg_classical(bgr)
        file_name = os.path.splitext(os.path.basename(path))[0] + ".png"
        cv2.imwrite(os.path.join(out_mask_dir, file_name), mask)

process_classical("data/dev/images", "data/dev/masks_task2")
process_classical("data/eval/images", "data/eval/masks_task2")