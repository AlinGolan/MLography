import os
import sys
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import binary_focal_loss, binary_focal_boundary_loss
from postprocess import postprocess_boundaries

# 1. Directories and model paths
IMP_PRE_INPUT_PATH = "../data/small/train/image"
IMP_INPUT_PATH = "../data/small_split/train/image_preprocess_cons"
GB_INPUT_PATH = "../data/squares_128_split/train/image"

IMP_GT_PATH = "../data/small_split/train/label_fixed_cons"
GB_GT_PATH = "../data/squares_128_split/train/inv_label"

IMP_MODEL_PATH = "../trained_models/new_models/impurities_early.hdf5"
GB_MODEL_PATH = "../trained_models/new_models/gb_early.hdf5"

# 2. Match stem pairs within each domain separately
def get_image_dict(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: '{directory}'")
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return {os.path.splitext(f)[0]: f for f in files}

# Impurities pool: match across pre-input, input, and ground truth
imp_pre_dict = get_image_dict(IMP_PRE_INPUT_PATH)
imp_in_dict = get_image_dict(IMP_INPUT_PATH)
imp_gt_dict = get_image_dict(IMP_GT_PATH)
imp_stems = sorted(list(set(imp_pre_dict.keys()) & set(imp_in_dict.keys()) & set(imp_gt_dict.keys())))
if not imp_stems:
    raise FileNotFoundError("No matching file stems found across all three IMP directories.")

# Grain boundary pool: match across input and ground truth
gb_in_dict = get_image_dict(GB_INPUT_PATH)
gb_gt_dict = get_image_dict(GB_GT_PATH)
gb_stems = sorted(list(set(gb_in_dict.keys()) & set(gb_gt_dict.keys())))
if not gb_stems:
    raise FileNotFoundError("No matching files found between GB input and GB GT directories.")

# Pick one random patch per domain independently
chosen_imp_stem = random.choice(imp_stems)
chosen_gb_stem = random.choice(gb_stems)
print(f"Selected IMP patch: '{chosen_imp_stem}'")
print(f"Selected GB patch:  '{chosen_gb_stem}'")

# Resolve actual filepaths
imp_pre_file = os.path.join(IMP_PRE_INPUT_PATH, imp_pre_dict[chosen_imp_stem])
imp_input_file = os.path.join(IMP_INPUT_PATH, imp_in_dict[chosen_imp_stem])
imp_gt_file = os.path.join(IMP_GT_PATH, imp_gt_dict[chosen_imp_stem])

gb_input_file = os.path.join(GB_INPUT_PATH, gb_in_dict[chosen_gb_stem])
gb_gt_file = os.path.join(GB_GT_PATH, gb_gt_dict[chosen_gb_stem])

# 3. Helper to load and preprocess images
def load_and_preprocess(filepath, is_mask=False):
    if is_mask:
        img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load mask at: {filepath}")
        return cv.resize(img, (128, 128))
    
    img = cv.imread(filepath)
    if img is None:
        raise ValueError(f"Could not load image at: {filepath}")
    img_rgb = cv.cvtColor(cv.resize(img, (128, 128)), cv.COLOR_BGR2RGB)
    img_tensor = img_rgb.astype("float32") / 255.0
    return img_rgb, np.expand_dims(img_tensor, axis=0)

imp_pre_vis, _ = load_and_preprocess(imp_pre_file)
imp_input_vis, imp_tensor = load_and_preprocess(imp_input_file)
imp_gt = load_and_preprocess(imp_gt_file, is_mask=True)

gb_input_vis, gb_tensor = load_and_preprocess(gb_input_file)
gb_gt = load_and_preprocess(gb_gt_file, is_mask=True)

# 4. Model inference
custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

print("Running impurities model inference...")
imp_model = load_model(IMP_MODEL_PATH, custom_objects=custom_objects)
imp_pred = np.squeeze(imp_model.predict(imp_tensor, verbose=0)[0])

# Postprocessing Impurities: Binarization (threshold > 0.5) + small speckle removal
imp_bin = (imp_pred > 0.5).astype(np.uint8)
cnts, _ = cv.findContours(imp_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
imp_postprocessed = np.zeros_like(imp_bin)
for c in cnts:
    if cv.contourArea(c) >= 5:
        cv.drawContours(imp_postprocessed, [c], -1, 255, -1)

print("Running grain boundary model inference...")
gb_model = load_model(GB_MODEL_PATH, custom_objects=custom_objects)
gb_pred = np.squeeze(gb_model.predict(gb_tensor, verbose=0)[0])

# Postprocessing Grain Boundary output
gb_postprocessed = postprocess_boundaries(gb_pred)

# 5. Continuous 2x5 Grid aligned vertically:
# Col 0: Pre-Input
# Col 1: Input (Preprocessed)
# Col 2: Output (Sigmoid Probabilities)
# Col 3: Postprocessed / Binarization
# Col 4: Ground Truth
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# --- Row 0: Impurities ---
axes[0, 0].imshow(imp_pre_vis)
axes[0, 0].set_title(f"IMP Pre-Input\n({chosen_imp_stem})", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(imp_input_vis)
axes[0, 1].set_title(f"IMP Input\n(Preprocessed)", fontsize=11, fontweight="bold")
axes[0, 1].axis("off")

axes[0, 2].imshow(imp_pred, cmap="gray", vmin=0, vmax=1)
axes[0, 2].set_title("IMP Output\n(Raw Probabilities)", fontsize=11, fontweight="bold")
axes[0, 2].axis("off")

axes[0, 3].imshow(imp_postprocessed, cmap="gray")
axes[0, 3].set_title("IMP Postprocess\n(Binarization)", fontsize=11, fontweight="bold")
axes[0, 3].axis("off")

axes[0, 4].imshow(imp_gt, cmap="gray")
axes[0, 4].set_title("IMP Ground Truth", fontsize=11, fontweight="bold")
axes[0, 4].axis("off")

# --- Row 1: Grain Boundaries ---
# Col 0 is left blank/neutral since GB task has no separate pre-processing raw stage
axes[1, 0].text(0.5, 0.5, "N/A\n(No Pre-Input)", ha="center", va="center", fontsize=11, color="gray")
axes[1, 0].set_title("GB Pre-Input", fontsize=11, fontweight="bold")
axes[1, 0].axis("off")

axes[1, 1].imshow(gb_input_vis)
axes[1, 1].set_title(f"GB Input\n({chosen_gb_stem})", fontsize=11, fontweight="bold")
axes[1, 1].axis("off")

axes[1, 2].imshow(gb_pred, cmap="gray", vmin=0, vmax=1)
axes[1, 2].set_title("GB Output\n(Raw Probabilities)", fontsize=11, fontweight="bold")
axes[1, 2].axis("off")

axes[1, 3].imshow(gb_postprocessed, cmap="gray")
axes[1, 3].set_title("GB Postprocess\n(Watershed/Thinned)", fontsize=11, fontweight="bold")
axes[1, 3].axis("off")

axes[1, 4].imshow(gb_gt, cmap="gray")
axes[1, 4].set_title("GB Ground Truth", fontsize=11, fontweight="bold")
axes[1, 4].axis("off")

plt.tight_layout()
output_img = f"visuals/single_patch_inference/patch_comparison_imp_{chosen_imp_stem}_gb_{chosen_gb_stem}.png"
plt.savefig(output_img, dpi=200, bbox_inches="tight")
plt.show()
print(f"Comparison saved to: {output_img}")