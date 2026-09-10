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

# 1. Directories and Paths
GB_INPUT_PATH = "../data/squares_128_split/train/image"
GB_GT_PATH = "../data/squares_128_split/train/inv_label"

# Directory containing the sweep models
SWEEP_MODELS_DIR = "../trained_models/new_models"

# Modular list: configure models to compare (Label, Model Path)
MODELS_TO_EVALUATE = [
    ("GB Early Stopping", os.path.join(SWEEP_MODELS_DIR, "gb_early.hdf5")),
    ("Model 01 (bw0.1)", os.path.join(SWEEP_MODELS_DIR, "model_02_bw0.1.hdf5")),
    ("Model 05 (bw0.75)", os.path.join(SWEEP_MODELS_DIR, "model_05_bw0.75.hdf5")),
    #("Model 07 (bw1.5)",  os.path.join(SWEEP_MODELS_DIR, "model_07_bw1.5.hdf5")),
    ("Model 10 (bw5.0)",  os.path.join(SWEEP_MODELS_DIR, "model_10_bw5.0.hdf5")),
]

# 2. Select a Random Patch Matching Between Input and Ground Truth
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

chosen_stem = random.choice(gb_stems)
print(f"Selected GB patch: '{chosen_stem}'")

# 3. Load and Preprocess Patch and Ground Truth
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

gb_input_vis, gb_tensor = load_and_preprocess(os.path.join(GB_INPUT_PATH, gb_in_dict[chosen_stem]))
gb_gt = load_and_preprocess(os.path.join(GB_GT_PATH, gb_gt_dict[chosen_stem]), is_mask=True)

# 4. Custom Loss Registrations for Model Loading
custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

# 5. Evaluate Models and Collect Post-processing Steps
pipeline_results = []
for label, model_path in MODELS_TO_EVALUATE:
    print(f"Processing model: {label}...")
    if not os.path.exists(model_path):
        print(f"Warning: File not found: {model_path}")
        pipeline_results.append((label, None, None))
        continue

    model = load_model(model_path, custom_objects=custom_objects)
    pred_map = np.squeeze(model.predict(gb_tensor, verbose=0)[0])
    
    # Run modular postprocessing with intermediate steps dictionary
    _, steps = postprocess_boundaries(
        pred_map,
        high_thresh=0.45,
        low_thresh=0.20,
        seed_thresh_factor=0.25,
        min_grain_area=15,
        return_intermediate=True
    )
    pipeline_results.append((label, pred_map, steps))

# 6. Plotting Grid: 
# Header Row: Centered Input and Ground Truth (across 4 columns)
# Modular Rows: One row per model with 4 steps:
# [GB Output] -> [Hysteresis Thresholding] -> [Guo-Hall Thinning] -> [Watershedding]
num_models = len(MODELS_TO_EVALUATE)
num_cols = 4
total_rows = 1 + num_models

fig = plt.figure(figsize=(16, 3.8 * total_rows))
gs = fig.add_gridspec(total_rows, num_cols, height_ratios=[1.1] + [1.0] * num_models)

# Top row: Center Input and Ground Truth (columns 1 and 2)
ax_top_input = fig.add_subplot(gs[0, 1])
ax_top_input.imshow(gb_input_vis)
ax_top_input.set_title(f"GB Input\n({chosen_stem})", fontsize=11, fontweight="bold")
ax_top_input.axis("off")

ax_top_gt = fig.add_subplot(gs[0, 2])
ax_top_gt.imshow(gb_gt, cmap="gray")
ax_top_gt.set_title("GB Ground Truth", fontsize=11, fontweight="bold")
ax_top_gt.axis("off")

# Hide outer cells of header row
fig.add_subplot(gs[0, 0]).axis("off")
fig.add_subplot(gs[0, 3]).axis("off")

# Model rows
for row_idx, (label, pred_map, steps) in enumerate(pipeline_results, start=1):
    ax_out = fig.add_subplot(gs[row_idx, 0])
    ax_bin = fig.add_subplot(gs[row_idx, 1])
    ax_guo = fig.add_subplot(gs[row_idx, 2])
    ax_wat = fig.add_subplot(gs[row_idx, 3])

    if pred_map is not None and steps is not None:
        ax_out.imshow(pred_map, cmap="gray", vmin=0, vmax=1)
        ax_out.set_title(f"{label}\nGB Output", fontsize=10)

        # Updated title to reflect the hysteresis binarization step
        ax_bin.imshow(steps["binarized"], cmap="gray")
        ax_bin.set_title("Hysteresis Thresholding", fontsize=10)

        ax_guo.imshow(steps["thinned"], cmap="gray")
        ax_guo.set_title("Guo-Hall Thinning", fontsize=10)

        ax_wat.imshow(steps["watershed"], cmap="gray")
        ax_wat.set_title("Watershedding", fontsize=10)
    else:
        for ax in (ax_out, ax_bin, ax_guo, ax_wat):
            ax.text(0.5, 0.5, "Model not found", ha="center", va="center")

    for ax in (ax_out, ax_bin, ax_guo, ax_wat):
        ax.axis("off")

plt.tight_layout()
output_img = f"visuals/gb_postprocess/gb_postprocess_multi_model_{chosen_stem}.png"
plt.savefig(output_img, dpi=200, bbox_inches="tight")
plt.show()
print(f"Visual saved to: {output_img}")