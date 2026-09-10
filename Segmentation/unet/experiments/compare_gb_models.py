import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import binary_focal_loss, binary_focal_boundary_loss

# 1. Direct file paths
INPUT_PATCH = "../data/squares_128/train/image"  # Folder with 128x128 crops
MODEL_BASELINE = "../trained_models/gb_150_epoch/gb_150.hdf5"
MODEL_IMPROVED = "../boundary_sweep_20260905_220223/models/model_01_bw0.0.hdf5"

# 2. Pick the first image from the folder
sample_images = sorted([f for f in os.listdir(INPUT_PATCH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
img_path = os.path.join(INPUT_PATCH, sample_images[0])
print(f"Testing on: {sample_images[0]}")

# 3. Read and normalize image (128x128x3)
img = cv.imread(img_path)
img_resized = cv.resize(img, (128, 128))
img_tensor = (img_resized.astype("float32") / 255.0)[np.newaxis, ...]

# 4. Custom loss functions needed to load models
custom_objects = {
    'binary_focal_loss_fixed': binary_focal_loss(alpha=0.2),
    'binary_focal_boundary_loss_fixed': binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
}

# 5. Evaluate Baseline (gb_150)
print("Loading Baseline model...")
baseline_model = load_model(MODEL_BASELINE, custom_objects=custom_objects)
baseline_pred = np.squeeze(baseline_model.predict(img_tensor)[0])

# 6. Evaluate Improved (boundary sweep)
print("Loading Improved model...")
improved_model = load_model(MODEL_IMPROVED, custom_objects=custom_objects)
improved_pred = np.squeeze(improved_model.predict(img_tensor)[0])

# 7. Plot side-by-side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(cv.cvtColor(img_resized, cv.COLOR_BGR2RGB))
axes[0].set_title(f"Input: {sample_images[0]}")
axes[0].axis("off")

axes[1].imshow(baseline_pred, cmap="gray")
axes[1].set_title("Baseline (gb_150)")
axes[1].axis("off")

axes[2].imshow(improved_pred, cmap="gray")
axes[2].set_title("Improved (model_01_bw0.0)")
axes[2].axis("off")

plt.tight_layout()
output_path = "visuals/gb_bw_sweep_inference/gb_comparison.png"
plt.savefig(output_path, dpi=200)
print(f"Saved comparison to {output_path}")