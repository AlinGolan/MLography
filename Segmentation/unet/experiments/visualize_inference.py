import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def create_visualization():
    # Define paths based on your directory structure
    results_dir = "../data/inference_results"
    
    img_name = "20"
    
    # File paths for the 6 components
    input_path = os.path.join(results_dir, f"{img_name}_input.jpg")
    imp_mask_path = os.path.join(results_dir, f"{img_name}_imp_mask.jpg")
    inpainted_path = os.path.join(results_dir, f"{img_name}_inpainted.png")
    gb_output_path = os.path.join(results_dir, f"{img_name}_gb_output.png")
    gb_postproc_path = os.path.join(results_dir, f"{img_name}_gb_postprocess.png")
    
    # Load images (using OpenCV, converting BGR to RGB for matplotlib)
    input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    imp_mask = cv2.imread(imp_mask_path, cv2.IMREAD_GRAYSCALE)
    inpainted_img = cv2.cvtColor(cv2.imread(inpainted_path), cv2.COLOR_BGR2RGB)
    gb_output = cv2.imread(gb_output_path, cv2.IMREAD_GRAYSCALE)
    gb_postproc = cv2.imread(gb_postproc_path, cv2.IMREAD_GRAYSCALE)
    
    # Create the 6th image: GB Postprocess + Inpainted Image Overlay
    # Thin the post-processed binary mask to white lines, then overlay onto the inpainted image
    kernel = np.ones((3,3), np.uint8)
    gb_lines = cv2.threshold(gb_postproc, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Blend: white boundaries over the inpainted image
    overlay_img = inpainted_img.copy()
    mask_lines = gb_lines == 255
    overlay_img[mask_lines] = [255, 255, 255] # White boundaries
    
    # Setup 2x3 matplotlib grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    titles = [
        "Input", 
        "Impurities Mask", 
        "Inpainting", 
        "GB Output", 
        "GB Post Process", 
        "GB Post Process + Inpainted"
    ]
    
    images = [
        input_img, 
        imp_mask, 
        inpainted_img, 
        gb_output, 
        gb_postproc, 
        overlay_img
    ]
    
    for ax, img, title in zip(axes.flat, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    
    # Save visualization
    output_vis_path = f"visuals/single_patch_inference/{img_name}_inference_visualization.png"
    plt.savefig(output_vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualization successfully saved to: {output_vis_path}")
    plt.show()

if __name__ == "__main__":
    create_visualization()