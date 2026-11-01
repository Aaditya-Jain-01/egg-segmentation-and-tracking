import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_metrics(gt_mask, pred_mask):
    gt = (gt_mask > 127).astype(np.uint8)
    pred = (pred_mask > 127).astype(np.uint8)

    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tn = np.sum((gt == 0) & (pred == 0))

    intersection = tp
    union = tp + fp + fn

    iou = intersection / union if union > 0 else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    dice = (2.0 * intersection) / (2.0 * intersection + fp + fn) if (2 * intersection + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }

def evaluate_split(split_name, base_dir):
    img_dir = os.path.join(base_dir, split_name, "images")
    gt_dir = os.path.join(base_dir, split_name, "masks_task1")
    pred_dir = os.path.join(base_dir, split_name, "masks_task2")

    records = []
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    for gt_path in gt_files:
        base_name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base_name)
        img_candidates = glob.glob(os.path.join(img_dir, os.path.splitext(base_name)[0] + ".*"))

        if not os.path.exists(pred_path) or not img_candidates:
            continue

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        metrics = compute_metrics(gt_mask, pred_mask)
        metrics["filename"] = base_name
        metrics["split"] = split_name
        metrics["img_path"] = img_candidates[0]
        metrics["gt_path"] = gt_path
        metrics["pred_path"] = pred_path
        records.append(metrics)

    return pd.DataFrame(records)

df_dev = evaluate_split("dev", "data")
df_eval = evaluate_split("eval", "data")
df_all = pd.concat([df_dev, df_eval], ignore_index=True)

os.makedirs("outputs/qualitative_plots", exist_ok=True)
df_all.to_csv("outputs/evaluation_metrics.csv", index=False)

summary_table = df_all.groupby("split")[["iou", "dice", "precision", "recall", "accuracy"]].mean()
print("=== Quantitative Evaluation Results ===")
print(summary_table.to_string())

worst_samples = df_all.sort_values(by="iou", ascending=True).head(6)

for idx, row in worst_samples.iterrows():
    bgr = cv2.imread(row["img_path"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(row["gt_path"], cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(row["pred_path"], cv2.IMREAD_GRAYSCALE)

    error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
    error_map[(gt > 127) & (pred > 127)] = [255, 255, 0]
    error_map[(gt == 0) & (pred > 127)] = [255, 0, 0]
    error_map[(gt > 127) & (pred == 0)] = [0, 255, 0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth (Task 1 SAM)")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title(f"Prediction (Task 2 Classical)\nIoU: {row['iou']:.3f}")
    axes[2].axis("off")

    axes[3].imshow(error_map)
    axes[3].set_title("Error Map (Yellow:TP, Red:FP, Green:FN)")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = f"outputs/qualitative_plots/worst_case_{row['split']}_{os.path.splitext(row['filename'])[0]}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()