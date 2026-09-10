import os
import sys
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import binary_focal_loss, binary_focal_boundary_loss

# Import all three available postprocessing functions
from postprocess import postprocess_boundaries, paper_postprocess, blur_watershed_postprocess

# =============================================================================
# 1. POST-PROCESSING CONFIGURATION (THE SWITCHER)
# =============================================================================
# Change this variable to switch between methods: "direct", "paper", or "blur_watershed"
ACTIVE_METHOD = "blur_watershed"

POSTPROCESS_CONFIGS = {
    "direct": {
        "func": postprocess_boundaries,
        "kwargs": {"bin_thresh": 0.20, "min_grain_area": 15}
    },
    "paper": {
        "func": paper_postprocess,
        "kwargs": {"bin_thresh": 55.0/255.0, "min_distance": 20}
    },
    "blur_watershed": {
        "func": blur_watershed_postprocess,
        "kwargs": {"blur_ksize": 5, "min_distance": 20, "peak_threshold_abs": 0.5}
    }
}

# =============================================================================
# 2. DIRECTORIES AND MODEL PATHS
# =============================================================================
GB_INPUT_PATH = "../data/squares_128_split/train/image"
GB_GT_PATH = "../data/squares_128_split/train/inv_label"
SWEEP_MODELS_DIR = "../trained_models/new_models"

MODELS_TO_EVALUATE = [
    ("GB Early Stopping", os.path.join(SWEEP_MODELS_DIR, "gb_early.hdf5")),
    ("Model 01 (bw0.1)", os.path.join(SWEEP_MODELS_DIR, "model_02_bw0.1.hdf5")),
    ("Model 05 (bw0.75)", os.path.join(SWEEP_MODELS_DIR, "model_05_bw0.75.hdf5")),
    ("Model 10 (bw5.0)",  os.path.join(SWEEP_MODELS_DIR, "model_10_bw5.0.hdf5")),
]

# =============================================================================
# 3. LOAD PATCH & GROUND TRUTH
# =============================================================================
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
print(f"Using Post-Processing Method: {ACTIVE_METHOD}")

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

# Custom Loss Registrations
custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

# =============================================================================
# 4. EVALUATE MODELS & RUN MODULAR POST-PROCESSING
# =============================================================================
config = POSTPROCESS_CONFIGS[ACTIVE_METHOD]
post_func = config["func"]
post_kwargs = config["kwargs"]

pipeline_results = []
step_keys = []  # Will dynamically store the names of the intermediate steps

for label, model_path in MODELS_TO_EVALUATE:
    print(f"Processing model: {label}...")
    if not os.path.exists(model_path):
        print(f"Warning: File not found: {model_path}")
        pipeline_results.append((label, None, None))
        continue

    model = load_model(model_path, custom_objects=custom_objects)
    pred_map = np.squeeze(model.predict(gb_tensor, verbose=0)[0])
    
    # Execute the selected post-processing function
    _, steps = post_func(
        pred_map, 
        return_intermediate=True, 
        **post_kwargs
    )
    
    # Capture the step keys from the first successful run to build the grid
    if not step_keys and steps is not None:
        step_keys = list(steps.keys())
        
    pipeline_results.append((label, pred_map, steps))

# =============================================================================
# 5. DYNAMIC PLOTTING GRID
# =============================================================================
num_models = len(MODELS_TO_EVALUATE)
num_cols = 1 + len(step_keys)  # 1 for the Raw Output + N intermediate steps
total_rows = 1 + num_models

fig = plt.figure(figsize=(4 * num_cols, 3.8 * total_rows))
gs = fig.add_gridspec(total_rows, num_cols, height_ratios=[1.1] + [1.0] * num_models)

# -- Header Row: Center the Input and Ground Truth --
center_idx = max(0, (num_cols - 2) // 2)

for i in range(num_cols):
    ax = fig.add_subplot(gs[0, i])
    if i == center_idx:
        ax.imshow(gb_input_vis)
        ax.set_title(f"GB Input\n({chosen_stem})", fontsize=11, fontweight="bold")
    elif i == center_idx + 1:
        ax.imshow(gb_gt, cmap="gray")
        ax.set_title("GB Ground Truth", fontsize=11, fontweight="bold")
    ax.axis("off")

# -- Model Rows --
for row_idx, (label, pred_map, steps) in enumerate(pipeline_results, start=1):
    
    # Column 0: Raw Model Output
    ax_out = fig.add_subplot(gs[row_idx, 0])
    if pred_map is not None:
        ax_out.imshow(pred_map, cmap="gray", vmin=0, vmax=1)
        ax_out.set_title(f"{label}\nGB Output", fontsize=10)
    else:
        ax_out.text(0.5, 0.5, "Model not found", ha="center", va="center")
    ax_out.axis("off")

    # Columns 1 to N: Dynamic Intermediate Steps
    for col_idx, key in enumerate(step_keys, start=1):
        ax_step = fig.add_subplot(gs[row_idx, col_idx])
        
        if steps is not None:
            # Use a colorful map for integer labels/markers so regions are visible
            cmap = "nipy_spectral" if ("label" in key or "marker" in key) else "gray"
            interpolation = "nearest" if cmap == "nipy_spectral" else None
            
            ax_step.imshow(steps[key], cmap=cmap, interpolation=interpolation)
            
            # Format the dictionary key into a clean title (e.g., 'watershed_contours' -> 'Watershed Contours')
            formatted_title = key.replace("_", " ").title()
            ax_step.set_title(formatted_title, fontsize=10)
        else:
            ax_step.text(0.5, 0.5, "N/A", ha="center", va="center")
            
        ax_step.axis("off")

plt.tight_layout()
output_img = f"visuals/gb_postprocess/gb_postprocess_{ACTIVE_METHOD}_{chosen_stem}.png"
plt.savefig(output_img, dpi=200, bbox_inches="tight")
plt.show()
print(f"Visual saved to: {output_img}")