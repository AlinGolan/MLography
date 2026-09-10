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

# Import all postprocessing methods
from postprocess import postprocess_boundaries, paper_postprocess, blur_watershed_postprocess

custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

def extract_bboxes(mask, min_area=5):
    """
    Extracts straight bounding rectangles from connected components,
    filtering out tiny noise speckles below min_area pixels.
    """
    cnts, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if cv.contourArea(c) >= min_area:
            boxes.append(cv.boundingRect(c))
    return boxes

def compute_box_iou(boxA, boxB):
    """Computes IoU between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area

    if union_area == 0:
        return 1.0 if inter_area == 0 else 0.0
    return inter_area / float(union_area)

def compute_oiou(y_true_mask, y_pred_mask, iou_threshold=0.3):
    """
    Computes Object IoU (OIoU):
    Matches bounding boxes with IoU >= iou_threshold 1-to-1,
    dividing by max(N_gt, N_pred).
    """
    boxes_true = extract_bboxes(y_true_mask, min_area=5)
    boxes_pred = extract_bboxes(y_pred_mask, min_area=5)
    
    n_gt = len(boxes_true)
    n_pred = len(boxes_pred)
    
    if max(n_gt, n_pred) == 0:
        return 1.0
        
    matched_count = 0
    used_preds = set()
    
    for bt in boxes_true:
        best_iou = 0.0
        best_pred_idx = -1
        for i, bp in enumerate(boxes_pred):
            if i in used_preds:
                continue
            iou = compute_box_iou(bt, bp)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = i
        
        if best_iou >= iou_threshold and best_pred_idx != -1:
            matched_count += 1
            used_preds.add(best_pred_idx)
                
    return float(matched_count / max(n_gt, n_pred))

def compute_bbox_iou(y_true_bin, y_pred_bin):
    """
    Computes IoU on the straight bounding boxes of impurities,
    matching the paper's specification for the impurities task.
    """
    boxes_true = extract_bboxes(y_true_bin, min_area=5)
    boxes_pred = extract_bboxes(y_pred_bin, min_area=5)
    
    canvas_true = np.zeros_like(y_true_bin, dtype=np.uint8)
    canvas_pred = np.zeros_like(y_pred_bin, dtype=np.uint8)
    
    for x, y, w, h in boxes_true:
        canvas_true[y:y+h, x:x+w] = 1
        
    for x, y, w, h in boxes_pred:
        canvas_pred[y:y+h, x:x+w] = 1
        
    intersection = np.logical_and(canvas_true, canvas_pred).sum()
    union = np.logical_or(canvas_true, canvas_pred).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def compute_gb_iou(bin_pred, bin_true):
    """
    Computes Grain Boundary semantic segmentation IoU using a pre-calculated binary prediction mask.
    """
    # Ensure bin_pred is strictly binary (0 and 1)
    bin_pred = (bin_pred > 0).astype(np.uint8)

    # Class 1 (Boundary) IoU
    inter_1 = np.logical_and(bin_true == 1, bin_pred == 1).sum()
    union_1 = np.logical_or(bin_true == 1, bin_pred == 1).sum()
    iou_boundary = float(inter_1 / union_1) if union_1 > 0 else (1.0 if inter_1 == 0 else 0.0)

    # Class 0 (Grain Interior) IoU
    inter_0 = np.logical_and(bin_true == 0, bin_pred == 0).sum()
    union_0 = np.logical_or(bin_true == 0, bin_pred == 0).sum()
    iou_grains = float(inter_0 / union_0) if union_0 > 0 else (1.0 if inter_0 == 0 else 0.0)

    # Mean IoU across boundary contours and grains
    m_iou = (iou_boundary + iou_grains) / 2.0
    return m_iou, iou_boundary, iou_grains

def load_data(image_dir, label_dir, is_gb=True):
    """
    Loads images and ground truth binary masks.
    Ensures labels are binary (0 for background, 1 for target feature).
    """
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
        lbl = cv.imread(lbl_p, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (128, 128))
        lbl = cv.resize(lbl, (128, 128), interpolation=cv.INTER_NEAREST)

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
            if is_gb:
                records.append({"Category": "Grain Boundaries", "Model Name": name, "ROC-AUC": "N/A", "IoU (Paper)": "N/A", "IoU (Blur)": "N/A", "OIoU": "N/A"})
            else:
                records.append({"Category": "Impurities", "Model Name": name, "ROC-AUC": "N/A", "IoU (Paper)": "N/A", "IoU (Blur)": "—", "OIoU": "N/A"})
            continue
            
        try:
            model = load_model(path, custom_objects=custom_objects)
            raw_preds = model.predict(images, batch_size=16, verbose=0)
            raw_preds = np.squeeze(raw_preds).astype(np.float32)
            tf.keras.backend.clear_session()

            if np.isnan(raw_preds).any():
                if is_gb:
                    records.append({"Category": "Grain Boundaries", "Model Name": name, "ROC-AUC": "Div", "IoU (Paper)": "Div", "IoU (Blur)": "Div", "OIoU": "Div"})
                else:
                    records.append({"Category": "Impurities", "Model Name": name, "ROC-AUC": "Div", "IoU (Paper)": "Div", "IoU (Blur)": "—", "OIoU": "Div"})
                continue

            # 1. ROC-AUC: Raw sigmoid probabilities directly against ground truth
            y_true_flat = labels.ravel()
            y_pred_flat = raw_preds.ravel()
            auc = roc_auc_score(y_true_flat, y_pred_flat)

            if is_gb:
                ious_paper, ious_blur = [], []
                
                for i in range(len(labels)):
                    bin_true = (labels[i] > 0.5).astype(np.uint8)
                    pred_patch = raw_preds[i]

                    # Paper Postprocessing 
                    pred_paper = paper_postprocess(pred_patch)
                    miou_paper, _, _ = compute_gb_iou(pred_paper, bin_true)
                    ious_paper.append(miou_paper)

                    # Blur Watershed Postprocessing
                    pred_blur = blur_watershed_postprocess(
                        pred_patch, 
                        blur_ksize=3, 
                        min_distance=5, 
                        peak_threshold_abs=0.3
                    )
                    miou_blur, _, _ = compute_gb_iou(pred_blur, bin_true)
                    ious_blur.append(miou_blur)

                mean_iou_paper = float(np.mean(ious_paper))
                mean_iou_blur = float(np.mean(ious_blur))

                records.append({
                    "Category": "Grain Boundaries",
                    "Model Name": name,
                    "ROC-AUC": f"{auc:.4f}",
                    "IoU (Paper)": f"{mean_iou_paper:.4f}",
                    "IoU (Blur)": f"{mean_iou_blur:.4f}",
                    "OIoU": "—"
                })
                print(f"Done: {name} -> ROC-AUC: {auc:.4f} | IoU (Paper): {mean_iou_paper:.4f} | IoU (Blur): {mean_iou_blur:.4f}")

            else:
                ious, oious = [], []
                for i in range(len(labels)):
                    bin_true = (labels[i] > 0.5).astype(np.uint8)
                    pred_patch = raw_preds[i]

                    # 3. Impurities: Probability binarization (threshold > 0.5) with small noise suppression
                    bin_pred = (pred_patch > 0.5).astype(np.uint8)
                    cnts, _ = cv.findContours(bin_pred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    cleaned_pred = np.zeros_like(bin_pred)
                    for c in cnts:
                        if cv.contourArea(c) >= 5:
                            cv.drawContours(cleaned_pred, [c], -1, 1, -1)
                    
                    ious.append(compute_bbox_iou(bin_true, cleaned_pred))
                    oious.append(compute_oiou(bin_true, cleaned_pred, iou_threshold=0.3))

                mean_iou = float(np.mean(ious))
                mean_oiou = float(np.mean(oious))

                records.append({
                    "Category": "Impurities",
                    "Model Name": name,
                    "ROC-AUC": f"{auc:.4f}",
                    "IoU (Paper)": f"{mean_iou:.4f}",
                    "IoU (Blur)": "—",
                    "OIoU": f"{mean_oiou:.4f}"
                })
                print(f"Done: {name} -> ROC-AUC: {auc:.4f}, IoU: {mean_iou:.4f}")

        except Exception as e:
            print(f"Error on {name}: {e}")
            if is_gb:
                records.append({"Category": "Grain Boundaries", "Model Name": name, "ROC-AUC": "Err", "IoU (Paper)": "Err", "IoU (Blur)": "Err", "OIoU": "Err"})
            else:
                records.append({"Category": "Impurities", "Model Name": name, "ROC-AUC": "Err", "IoU (Paper)": "Err", "IoU (Blur)": "—", "OIoU": "Err"})

    return records

# 1. Models setup (Explicit list from trained_models/new_models)
MODELS_DIR = "../trained_models/new_models"

gb_models = [
    ("GB Early Stopping", f"{MODELS_DIR}/gb_early.hdf5"),
    ("GB model_01 (bw=0.1)", f"{MODELS_DIR}/model_02_bw0.1.hdf5"),
    ("GB model_02 (bw=0.75)", f"{MODELS_DIR}/model_05_bw0.75.hdf5"),
    ("GB model_03 (bw=5.0)", f"{MODELS_DIR}/model_10_bw5.0.hdf5"),
]

imp_models = [
    ("Impurities Early Stopping", f"{MODELS_DIR}/impurities_early.hdf5"),
]

# 2. Evaluation
print("Evaluating GB models...")
gb_imgs, gb_lbls = load_data("../data/squares_128_split/test/image", "../data/squares_128_split/test/inv_label", is_gb=True)
if len(gb_imgs) == 0:
    print("Warning: Test set missing or empty, falling back to train directory.")
    gb_imgs, gb_lbls = load_data("../data/squares_128_split/train/image", "../data/squares_128_split/train/inv_label", is_gb=True)

results = evaluate_models(gb_models, gb_imgs, gb_lbls, is_gb=True)

print("\nEvaluating Impurities models...")
imp_img_dir = "../data/small_split/test/image_preprocess_cons"
imp_lbl_dir = "../data/small_split/test/label_fixed_cons"
imp_imgs, imp_lbls = load_data(imp_img_dir, imp_lbl_dir, is_gb=False)
if len(imp_imgs) == 0:
    print("Warning: Test set missing or empty, falling back to train directory.")
    imp_imgs, imp_lbls = load_data("../data/small_split/train/image_preprocess_cons", "../data/small_split/train/label_fixed_cons", is_gb=False)

if len(imp_imgs) > 0:
    results.extend(evaluate_models(imp_models, imp_imgs, imp_lbls, is_gb=False))

# 3. Render Table
df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(13, len(df) * 0.45 + 1.2))  # Widened figure slightly for the extra column
ax.axis('off')
ax.axis('tight')

table = ax.table(
    cellText=df[["Category", "Model Name", "ROC-AUC", "IoU (Paper)", "IoU (Blur)", "OIoU"]].values,
    colLabels=["Category", "Model Name", "ROC-AUC", "IoU (Paper)", "IoU (Blur)", "OIoU"],
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