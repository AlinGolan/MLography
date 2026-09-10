import os
import sys
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import binary_focal_loss, binary_focal_boundary_loss

# 1. Source inpainted image, full label, and model paths
INPAINTED_IMG_PATH = "../data/without_impurities/10.png"
# UPDATE THIS PATH to wherever your full-size ground truth masks are stored
FULL_LABEL_PATH = "../data/post_segmented_edges_binary/masked/10.png" 

BASELINE_PATH = "../trained_models/gb_150_epoch/gb_150.hdf5"
EARLY_PATH = "../trained_models/gb_early/gb_early.hdf5"
SWEEP_DIR = "../boundary_sweep_par_20260905_220721/models"

# 2. Extract a random 128x128 crop from the inpainted image and its label
if not os.path.exists(INPAINTED_IMG_PATH):
    raise FileNotFoundError(f"Inpainted image not found at {INPAINTED_IMG_PATH}.")
if not os.path.exists(FULL_LABEL_PATH):
    raise FileNotFoundError(f"Ground truth label not found at {FULL_LABEL_PATH}.")

full_img = cv.imread(INPAINTED_IMG_PATH)
full_lbl = cv.imread(FULL_LABEL_PATH, cv.IMREAD_GRAYSCALE)

h, w, _ = full_img.shape

# Pick random top-left corner ensuring a full 128x128 window
y = random.randint(0, h - 128)
x = random.randint(0, w - 128)
patch_img = full_img[y:y+128, x:x+128]
patch_lbl = full_lbl[y:y+128, x:x+128]

# If the label is inverted (white grains, black boundaries), flip it for visual consistency
if np.mean(patch_lbl) > 127:
    patch_lbl = 255 - patch_lbl

# Normalize to [0, 1] with shape (1, 128, 128, 3)
patch_tensor = (patch_img.astype("float32") / 255.0)[np.newaxis, ...]

# 3. Register custom losses for model loading
custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

# 4. Gather models (Baseline + Early + all 10 sweep models sorted)
model_entries = []
if os.path.exists(BASELINE_PATH):
    model_entries.append(("Baseline (gb_150)", BASELINE_PATH))
if os.path.exists(EARLY_PATH):
    model_entries.append(("Early Stopping", EARLY_PATH))

sweep_files = sorted([f for f in os.listdir(SWEEP_DIR) if f.endswith(".hdf5")])
for f in sweep_files:
    label = f.replace(".hdf5", "").replace("model_", "m").replace("_", " ")
    model_entries.append((label, os.path.join(SWEEP_DIR, f)))

# Total panels: Input + GT + 12 models = 14 panels. A 4x4 grid (16 panels) fits this well.
rows, cols = 4, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
axes = axes.flatten()

# Display the Input Patch
axes[0].imshow(cv.cvtColor(patch_img, cv.COLOR_BGR2RGB))
axes[0].set_title(f"Input Crop (x:{x}, y:{y})", fontsize=11, fontweight="bold")
axes[0].axis("off")

# Display the Ground Truth Patch
axes[1].imshow(patch_lbl, cmap="gray")
axes[1].set_title("Ground Truth", fontsize=11, fontweight="bold", color="green")
axes[1].axis("off")

# 5. Run inference and plot each model
for idx, (title, model_path) in enumerate(model_entries, start=2):
    print(f"[{idx-1}/{len(model_entries)}] Running inference: {title}...")
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        pred = model.predict(patch_tensor, verbose=0)[0]
        pred_map = np.squeeze(pred)

        axes[idx].imshow(pred_map, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=10)
    except Exception as e:
        axes[idx].text(0.5, 0.5, f"Error loading\n{title}", ha="center", va="center", fontsize=9)
        print(f"Failed loading {model_path}: {e}")
    
    axes[idx].axis("off")
    tf.keras.backend.clear_session()

# Hide any remaining unused subplots
for idx in range(2 + len(model_entries), len(axes)):
    axes[idx].axis("off")

plt.suptitle(f"GB Models Comparison on Inpainted Patch ({INPAINTED_IMG_PATH})", fontsize=15, y=0.95, fontweight="bold")
plt.subplots_adjust(wspace=0.1, hspace=0.2)

output_filename = "visuals/gb_bw_sweep_inference/all_gb_models_comparison_with_gt.png"
plt.savefig(output_filename, dpi=200, bbox_inches="tight")
print(f"\nAll models evaluated. Visual saved to: {output_filename}")