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

# 1. Paths
GB_INPUT_PATH = "../data/squares_128_split/train/image"
GB_GT_PATH = "../data/squares_128_split/train/inv_label"

# Directory containing the sweep models (adjust folder timestamp if needed)
SWEEP_MODELS_DIR = "../trained_models/new_models"

# Modular list: add or remove models here seamlessly
MODELS_TO_EVALUATE = [
    ("GB Early Stopping", os.path.join(SWEEP_MODELS_DIR, "gb_early.hdf5")),
    ("Model 01 (bw0.1)", os.path.join(SWEEP_MODELS_DIR, "model_02_bw0.1.hdf5")),
    ("Model 05 (bw0.75)", os.path.join(SWEEP_MODELS_DIR, "model_05_bw0.75.hdf5")),
    ("Model 10 (bw5.0)",  os.path.join(SWEEP_MODELS_DIR, "model_10_bw5.0.hdf5")),
]

# 2. Match stem pairs and randomly select a patch
def get_image_dict(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: '{directory}'")
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return {os.path.splitext(f)[0]: f for f in files}

gb_in_dict = get_image_dict(GB_INPUT_PATH)
gb_gt_dict = get_image_dict(GB_GT_PATH)
gb_stems = sorted(list(set(gb_in_dict.keys()) & set(gb_gt_dict.keys())))

if not gb_stems:
    raise FileNotFoundError("No matching files found between GB input and GB GT directories.")

chosen_gb_stem = random.choice(gb_stems)
print(f"Selected GB patch: '{chosen_gb_stem}'")

# 3. Load and preprocess image and GT
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

gb_input_file = os.path.join(GB_INPUT_PATH, gb_in_dict[chosen_gb_stem])
gb_gt_file = os.path.join(GB_GT_PATH, gb_gt_dict[chosen_gb_stem])

gb_input_vis, gb_tensor = load_and_preprocess(gb_input_file)
gb_gt = load_and_preprocess(gb_gt_file, is_mask=True)

# 4. Custom objects dictionary for model loading
custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

# 5. Modular Inference & Post-processing Loop
model_results = []
for label, model_path in MODELS_TO_EVALUATE:
    print(f"Running inference for {label}...")
    if not os.path.exists(model_path):
        print(f"[Warning] Model file not found: {model_path}")
        model_results.append((label, None, None))
        continue

    model = load_model(model_path, custom_objects=custom_objects)
    pred = np.squeeze(model.predict(gb_tensor, verbose=0)[0])
    post = postprocess_boundaries(pred)
    model_results.append((label, pred, post))

# 6. Dynamic Grid Plotting based on the sketch
# Row 0: Centered GB Input and GB GT
# Row 1: Model Raw Outputs (N columns)
# Row 2: Model Post-processed Outputs (N columns)
num_models = len(MODELS_TO_EVALUATE)
fig = plt.figure(figsize=(4 * num_models, 11))
gs = fig.add_gridspec(3, num_models, height_ratios=[1.1, 1.0, 1.0])

# Top row: Centering Input and GT
if num_models >= 2:
    col_input = (num_models // 2) - 1
    col_gt = num_models // 2
else:
    col_input, col_gt = 0, 0

ax_input = fig.add_subplot(gs[0, col_input])
ax_input.imshow(gb_input_vis)
ax_input.set_title(f"GB Input\n({chosen_gb_stem})", fontsize=11, fontweight="bold")
ax_input.axis("off")

ax_gt = fig.add_subplot(gs[0, col_gt])
ax_gt.imshow(gb_gt, cmap="gray")
ax_gt.set_title("GB Ground Truth", fontsize=11, fontweight="bold")
ax_gt.axis("off")

# Hide any remaining unused cells in the top row
for c in range(num_models):
    if c not in (col_input, col_gt):
        ax_empty = fig.add_subplot(gs[0, c])
        ax_empty.axis("off")

# Middle row (Raw model outputs) & Bottom row (Post-processed results)
for col_idx, (label, pred, post) in enumerate(model_results):
    ax_raw = fig.add_subplot(gs[1, col_idx])
    ax_post = fig.add_subplot(gs[2, col_idx])

    if pred is not None:
        ax_raw.imshow(pred, cmap="gray")
        ax_raw.set_title(f"{label}\nRaw Output", fontsize=10)
        
        ax_post.imshow(post, cmap="gray")
        ax_post.set_title(f"{label}\nPostprocess", fontsize=10)
    else:
        ax_raw.text(0.5, 0.5, "Model not found", ha="center", va="center")
        ax_post.text(0.5, 0.5, "N/A", ha="center", va="center")

    ax_raw.axis("off")
    ax_post.axis("off")

plt.tight_layout()
output_img = f"visuals/gb_bw_sweep_inference/gb_multi_model_comparison_{chosen_gb_stem}.png"
plt.savefig(output_img, dpi=200, bbox_inches="tight")
plt.show()
print(f"Visual saved to: {output_img}")