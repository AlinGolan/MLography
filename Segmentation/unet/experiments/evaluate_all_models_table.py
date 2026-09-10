import os
import sys
import glob
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import binary_focal_loss, binary_focal_boundary_loss

custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

def compute_oiou(y_true_mask, y_pred_mask):
    """
    Computes Object IoU (OIoU) matching the paper definition:
    Ratio of overlapping bounding box detections over max(N_gt, N_pred).
    """
    cnts_true, _ = cv.findContours(y_true_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_pred, _ = cv.findContours(y_pred_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    boxes_true = [cv.boundingRect(c) for c in cnts_true if cv.contourArea(c) > 2]
    boxes_pred = [cv.boundingRect(c) for c in cnts_pred if cv.contourArea(c) > 2]
    
    n_gt = len(boxes_true)
    n_pred = len(boxes_pred)
    if max(n_gt, n_pred) == 0:
        return 1.0
    
    matched = 0
    for xt, yt, wt, ht in boxes_true:
        for xp, yp, wp, hp in boxes_pred:
            # Check bounding box intersection
            inter_x1 = max(xt, xp)
            inter_y1 = max(yt, yp)
            inter_x2 = min(xt + wt, xp + wp)
            inter_y2 = min(yt + ht, yp + hp)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                matched += 1
                break
    return float(matched / max(n_gt, n_pred))

def compute_pixel_iou(y_true_bin, y_pred_bin):
    intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def load_data(image_dir, label_dir, is_gb=True):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    images, labels = [], []
    for p in img_paths:
        if not p.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        lbl_p = os.path.join(label_dir, base + ".png")
        if not os.path.exists(lbl_p):
            lbl_p = os.path.join(label_dir, base + ".jpg")
        if not os.path.exists(lbl_p):
            continue

        img = cv.imread(p)
        lbl = cv.imread(lbl_p, 0)
        img = cv.resize(img, (128, 128))
        lbl = cv.resize(lbl, (128, 128))

        images.append(img.astype(np.float32) / 255.0)
        # Binarization matching dataset mask polarity
        if is_gb:
            labels.append((lbl > 50).astype(np.float32))
        else:
            # Ensure inclusions are positive class (1)
            labels.append((lbl > 127).astype(np.float32))

    return np.array(images), np.array(labels)

def evaluate_models(model_list, images, labels, is_gb=True):
    records = []
    
    # Verify ground truth polarity
    if not is_gb and np.mean(labels) > 0.5:
        labels = 1.0 - labels

    for name, path in model_list:
        if not os.path.exists(path):
            records.append({
                "Category": "Grain Boundaries" if is_gb else "Impurities",
                "Model Name": name,
                "ROC-AUC": "N/A",
                "IoU": "N/A",
                "OIoU": "N/A"
            })
            continue
        try:
            model = load_model(path, custom_objects=custom_objects)
            raw_preds = model.predict(images, batch_size=16, verbose=0)
            raw_preds = np.squeeze(raw_preds)
            tf.keras.backend.clear_session()

            if np.isnan(raw_preds).any():
                records.append({
                    "Category": "Grain Boundaries" if is_gb else "Impurities",
                    "Model Name": name,
                    "ROC-AUC": "Diverged",
                    "IoU": "Diverged",
                    "OIoU": "Diverged"
                })
                continue

            # Apply (3, 3) Gaussian blur to each raw prediction patch like repo evaluate_segmentation
            blurred_preds = []
            for i in range(len(raw_preds)):
                patch_float = raw_preds[i].astype(np.float32)
                blurred = cv.GaussianBlur(patch_float, (3, 3), 0)
                blurred_preds.append(blurred)
            blurred_preds = np.array(blurred_preds)

            # Flatten for ROC-AUC
            y_true_flat = labels.ravel()
            y_pred_flat = blurred_preds.ravel()

            # ROC-AUC score
            auc = roc_auc_score(y_true_flat, y_pred_flat)
            if auc < 0.5:
                auc = 1.0 - auc
                blurred_preds = 1.0 - blurred_preds

            # Otsu thresholding & IoU / OIoU computation
            ious, oious = [], []
            for i in range(len(labels)):
                p_patch = (blurred_preds[i] * 255).astype(np.uint8)
                t_patch = (labels[i] * 255).astype(np.uint8)

                _, bin_pred = cv.threshold(p_patch, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                bin_pred = (bin_pred > 0).astype(np.uint8)
                bin_true = (t_patch > 0).astype(np.uint8)

                ious.append(compute_pixel_iou(bin_true, bin_pred))
                if not is_gb:
                    oious.append(compute_oiou(bin_true, bin_pred))

            mean_iou = float(np.mean(ious))
            mean_oiou = float(np.mean(oious)) if not is_gb else None

            records.append({
                "Category": "Grain Boundaries" if is_gb else "Impurities",
                "Model Name": name,
                "ROC-AUC": f"{auc:.4f}",
                "IoU": f"{mean_iou:.4f}",
                "OIoU": f"{mean_oiou:.4f}" if mean_oiou is not None else "—"
            })
            print(f"Done: {name} -> ROC-AUC: {auc:.4f}, IoU: {mean_iou:.4f}")
        except Exception as e:
            print(f"Error on {name}: {e}")
            records.append({
                "Category": "Grain Boundaries" if is_gb else "Impurities",
                "Model Name": name,
                "ROC-AUC": "Err",
                "IoU": "Err",
                "OIoU": "Err"
            })

    return records

# 1. Models setup
gb_sweep_dir = "../boundary_sweep_par_20260905_220721/models"
gb_models = [
    ("GB Baseline (150 Epochs)", "../trained_models/gb_150_epoch/gb_150.hdf5"),
    ("GB Early Stopping", "../trained_models/gb_early/gb_early.hdf5"),
]
for f in sorted([f for f in os.listdir(gb_sweep_dir) if f.endswith(".hdf5")]):
    label = f.replace(".hdf5", "").replace("model_", "GB Sweep ").replace("_", " ")
    gb_models.append((label, os.path.join(gb_sweep_dir, f)))

imp_models = [
    ("Impurities Baseline (150 Epochs)", "../trained_models/impurities_150_epoch/impurities_150.hdf5"),
    ("Impurities Early Stopping", "../trained_models/impurities_early/impurities_early.hdf5"),
]

# 2. Evaluate
print("Evaluating GB models...")
gb_imgs, gb_lbls = load_data("../data/squares_128/train/image", "../data/squares_128/train/inv_label", is_gb=True)
results = evaluate_models(gb_models, gb_imgs, gb_lbls, is_gb=True)

print("\nEvaluating Impurities models...")
imp_img_dir = "../data/small/train/image_preprocess" if os.path.exists("../data/small/train/image_preprocess") else "../data/small/train/image"
imp_lbl_dir = "../data/small/train/label_fixed" if os.path.exists("../data/small/train/label_fixed") else "../data/small/train/label"
imp_imgs, imp_lbls = load_data(imp_img_dir, imp_lbl_dir, is_gb=False)
if len(imp_imgs) > 0:
    results.extend(evaluate_models(imp_models, imp_imgs, imp_lbls, is_gb=False))

# 3. Render Table
df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(11, len(df) * 0.45 + 1.2))
ax.axis('off')
ax.axis('tight')

table = ax.table(
    cellText=df[["Category", "Model Name", "ROC-AUC", "IoU", "OIoU"]].values,
    colLabels=["Category", "Model Name", "ROC-AUC", "IoU", "OIoU"],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.4)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1f2d3d")
        cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f8f9fa")
    else:
        cell.set_facecolor("#ffffff")
    cell.set_edgecolor("#dcdfe6")

plt.title("Model Benchmark Performance (ROC-AUC, IoU, and OIoU)", fontsize=13, weight="bold", pad=15)
out_file = "visuals/metrics_tables/models_benchmark_table.png"
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"\nCompleted. Table saved to: {out_file}")