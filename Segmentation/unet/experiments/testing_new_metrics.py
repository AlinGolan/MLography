import os
import glob
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import binary_focal_loss, binary_focal_boundary_loss

custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

def extract_bboxes(mask):
    """Helper function to extract bounding boxes from a binary mask."""
    cnts, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return [cv.boundingRect(c) for c in cnts]

def compute_oiou(y_true_mask, y_pred_mask):
    """Computes Object IoU (OIoU)."""
    boxes_true = extract_bboxes(y_true_mask)
    boxes_pred = extract_bboxes(y_pred_mask)
    
    n_gt = len(boxes_true)
    n_pred = len(boxes_pred)
    
    if max(n_gt, n_pred) == 0:
        return 1.0
        
    matched_count = 0
    used_preds = set()
    
    for xt, yt, wt, ht in boxes_true:
        for i, (xp, yp, wp, hp) in enumerate(boxes_pred):
            if i in used_preds:
                continue
            
            inter_x1 = max(xt, xp)
            inter_y1 = max(yt, yp)
            inter_x2 = min(xt + wt, xp + wp)
            inter_y2 = min(yt + ht, yp + hp)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                matched_count += 1
                used_preds.add(i)
                break
                
    return float(matched_count / max(n_gt, n_pred))

def compute_bbox_iou(y_true_bin, y_pred_bin):
    """Computes pixel-wise IoU over the BOUNDING BOXES of the masks."""
    boxes_true = extract_bboxes(y_true_bin)
    boxes_pred = extract_bboxes(y_pred_bin)
    
    canvas_true = np.zeros_like(y_true_bin)
    canvas_pred = np.zeros_like(y_pred_bin)
    
    for x, y, w, h in boxes_true:
        canvas_true[y:y+h, x:x+w] = 1
        
    for x, y, w, h in boxes_pred:
        canvas_pred[y:y+h, x:x+w] = 1
        
    intersection = np.logical_and(canvas_true, canvas_pred).sum()
    union = np.logical_or(canvas_true, canvas_pred).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def compute_miou(y_true_bin, y_pred_bin):
    """Computes the Mean IoU (mIoU) across both classes."""
    inter_1 = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
    union_1 = np.logical_or(y_true_bin == 1, y_pred_bin == 1).sum()
    iou_1 = inter_1 / union_1 if union_1 > 0 else (1.0 if inter_1 == 0 else 0.0)
    
    inter_0 = np.logical_and(y_true_bin == 0, y_pred_bin == 0).sum()
    union_0 = np.logical_or(y_true_bin == 0, y_pred_bin == 0).sum()
    iou_0 = inter_0 / union_0 if union_0 > 0 else (1.0 if inter_0 == 0 else 0.0)
    
    return (iou_1 + iou_0) / 2.0

def compute_ari(y_true_bin, y_pred_bin):
    """
    Computes Adjusted Rand Index (ARI) by extracting connected grain components.
    Inverts the boundaries (0) to grains (1) and applies connected components.
    """
    _, true_labels = cv.connectedComponents((1 - y_true_bin).astype(np.uint8))
    _, pred_labels = cv.connectedComponents((1 - y_pred_bin).astype(np.uint8))
    
    return adjusted_rand_score(true_labels.ravel(), pred_labels.ravel())

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
        if is_gb:
            labels.append((lbl > 50).astype(np.float32))
        else:
            labels.append((lbl > 127).astype(np.float32))

    return np.array(images), np.array(labels)

def evaluate_models(model_list, images, labels, is_gb=True):
    records = []
    
    if not is_gb and np.mean(labels) > 0.5:
        labels = 1.0 - labels

    for name, path in model_list:
        if not os.path.exists(path):
            records.append({
                "Category": "Grain Boundaries" if is_gb else "Impurities",
                "Model Name": name,
                "ROC-AUC": "N/A",
                "mAP": "N/A",
                "IoU": "N/A",
                "ARI": "N/A",
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
                    "mAP": "Diverged",
                    "IoU": "Diverged",
                    "ARI": "Diverged",
                    "OIoU": "Diverged"
                })
                continue

            y_true_flat = labels.ravel()
            y_pred_flat = raw_preds.ravel()

            auc = roc_auc_score(y_true_flat, y_pred_flat)
            if auc < 0.5:
                auc = 1.0 - auc
                raw_preds = 1.0 - raw_preds
                y_pred_flat = raw_preds.ravel()
                
            ap = average_precision_score(y_true_flat, y_pred_flat)

            ious, oious, aris = [], [], []
            for i in range(len(labels)):
                t_patch = (labels[i] * 255).astype(np.uint8)
                bin_true = (t_patch > 0).astype(np.uint8)

                if is_gb:
                    # 1. Normalized Hard Binarization
                    p_patch = cv.normalize(raw_preds[i], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                    _, bin_pred = cv.threshold(p_patch, 127, 255, cv.THRESH_BINARY)
                    
                    # 2. Morphological Close
                    kernel = np.ones((3,3), np.uint8)
                    bin_pred = cv.morphologyEx(bin_pred, cv.MORPH_CLOSE, kernel)
                    
                    # 3. Guo-Hall Thinning
                    try:
                        thinned_mask = cv.ximgproc.thinning(bin_pred, thinningType=cv.ximgproc.THINNING_GUOHALL)
                    except AttributeError:
                        thinned_mask = bin_pred
                        
                    # 4. Watershed Algorithm
                    thinned_inv = cv.bitwise_not(thinned_mask)
                    dist_transform = cv.distanceTransform(thinned_inv, cv.DIST_L2, 3)
                    
                    # Find the sure foreground (centers of the grains)
                    _, sure_fg = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
                    sure_fg = np.uint8(sure_fg)
                    
                    sure_bg = cv.dilate(thinned_mask, np.ones((3, 3), np.uint8), iterations=1)
                    unknown = cv.subtract(sure_bg, sure_fg)
                    
                    _, markers = cv.connectedComponents(sure_fg)
                    markers = markers + 1
                    markers[unknown == 255] = 0
                    
                    # Prepare original image for watershed
                    img_8u = (images[i] * 255).astype(np.uint8)
                    if len(img_8u.shape) == 2 or img_8u.shape[2] == 1:
                        img_3c = cv.cvtColor(img_8u, cv.COLOR_GRAY2BGR)
                    else:
                        img_3c = img_8u
                        
                    markers = cv.watershed(img_3c, markers)
                    
                    water_mask = np.zeros_like(bin_pred)
                    water_mask[markers == -1] = 1 # Set boundaries to 1 for mIoU/ARI
                    
                    ious.append(compute_miou(bin_true, water_mask))
                    aris.append(compute_ari(bin_true, water_mask))
                else:
                    # Impurities logic (keep Gaussian Blur + Otsu)
                    patch_float = raw_preds[i].astype(np.float32)
                    blurred = cv.GaussianBlur(patch_float, (3, 3), 0)
                    p_patch = (blurred * 255).astype(np.uint8)
                    _, bin_pred = cv.threshold(p_patch, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                    bin_pred = (bin_pred > 0).astype(np.uint8)
                    
                    ious.append(compute_bbox_iou(bin_true, bin_pred))
                    oious.append(compute_oiou(bin_true, bin_pred))

            mean_iou = float(np.mean(ious))
            mean_ari = float(np.mean(aris)) if is_gb else None
            mean_oiou = float(np.mean(oious)) if not is_gb else None

            records.append({
                "Category": "Grain Boundaries" if is_gb else "Impurities",
                "Model Name": name,
                "ROC-AUC": f"{auc:.4f}",
                "mAP": f"{ap:.4f}",
                "IoU": f"{mean_iou:.4f}",
                "ARI": f"{mean_ari:.4f}" if mean_ari is not None else "—",
                "OIoU": f"{mean_oiou:.4f}" if mean_oiou is not None else "—"
            })
            print(f"Done: {name} -> ROC-AUC: {auc:.4f}, mAP: {ap:.4f}, IoU: {mean_iou:.4f}")
        except Exception as e:
            print(f"Error on {name}: {e}")
            records.append({
                "Category": "Grain Boundaries" if is_gb else "Impurities",
                "Model Name": name,
                "ROC-AUC": "Err",
                "mAP": "Err",
                "IoU": "Err",
                "ARI": "Err",
                "OIoU": "Err"
            })

    return records

# 1. Models setup
gb_sweep_dir = "../boundary_sweep_par_20260905_220721/models"
gb_models = [
    ("GB Baseline (150 Epochs)", "../trained_models/gb_150_epoch/gb_150.hdf5"),
    ("GB Early Stopping", "../trained_models/gb_early/gb_early.hdf5"),
    ("GB UNet", "../trained_models/gb_unet/model.weights.h5"),
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
fig, ax = plt.subplots(figsize=(13, len(df) * 0.45 + 1.2))
ax.axis('off')
ax.axis('tight')

cols = ["Category", "Model Name", "ROC-AUC", "mAP", "IoU", "ARI", "OIoU"]
table = ax.table(
    cellText=df[cols].values,
    colLabels=cols,
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

plt.title("Model Benchmark Performance (Post-Processed)", fontsize=13, weight="bold", pad=15)
out_file = "visuals/metrics_tables/models_benchmark_table_postprocessed.png"
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"\nCompleted. Table saved to: {out_file}")