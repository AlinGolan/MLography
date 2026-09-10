import cv2 as cv
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries

def paper_postprocess(
    pred_map: np.ndarray,
    bin_thresh: float = 55.0 / 255.0,  # ~0.215 to match the original '55'
    min_distance: int = 20,
    return_intermediate: bool = False
):
    """
    Implementation of the paper's original post-processing pipeline.
    (Binarization -> Guo-Hall -> Distance Transform/Watershed -> Contours -> Guo-Hall)
    """
    # 1. Format & Normalization
    pred_map = np.squeeze(pred_map).astype(np.float32)
    if pred_map.ndim != 2:
        raise ValueError(f"Expected a 2D prediction map, got shape {pred_map.shape}")

    if pred_map.max() > 1.0:
        pred_map = pred_map / 255.0
    pred_map = np.clip(pred_map, 0.0, 1.0)
    
    # Scale to 0-255 uint8 for OpenCV operations
    img_uint8 = (pred_map * 255).astype(np.uint8)

    # ---------------------------------------------------------
    # STEP 1: edges_binarization
    # ---------------------------------------------------------
    # Original threshold was 55 on a 0-255 scale
    _, threshed = cv.threshold(img_uint8, int(bin_thresh * 255), 255, cv.THRESH_BINARY)
    
    # First Guo-Hall Thinning
    try:
        thinned = cv.ximgproc.thinning(threshed, thinningType=cv.ximgproc.THINNING_GUOHALL)
    except AttributeError:
        thinned = threshed  # Fallback if ximgproc is missing

    # ---------------------------------------------------------
    # STEP 2: edges_postprocess
    # ---------------------------------------------------------
    # Invert the thinned boundaries to get the grain interiors
    image_inv = cv.bitwise_not(thinned)
    image_norm = image_inv / 255.0  # Normalize to 0 and 1 for distance transform
    
    # Compute Distance Transform
    distance_map = ndimage.distance_transform_edt(image_norm)
    
    # Find local maxima coordinates
    coords = peak_local_max(
        distance_map, 
        min_distance=min_distance, 
        labels=(image_norm > 0.5).astype(int),  # <-- Explicitly create an integer mask
        exclude_border=0
    )
    
    # Manually create the boolean mask from the coordinates
    local_max = np.zeros(distance_map.shape, dtype=bool)
    if len(coords) > 0:
        local_max[tuple(coords.T)] = True
    
    # Create markers from the local maxima 
    markers = ndimage.label(local_max)[0]
    
    # Watershed to expand the grain centers back out to the boundaries
    labels = watershed(-distance_map, markers, mask=image_norm)
    
    cnts_mask = np.zeros_like(img_uint8)
    
    # Iterate through unique labels to draw thick contours
    for label_idx in np.unique(labels):
        if label_idx == 0:
            continue
            
        # Isolate the current grain
        mask = np.zeros_like(img_uint8)
        mask[labels == label_idx] = 255
        
        # Find its contour
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if len(cnts) == 0:
            continue
            
        # Keep only the largest contour to avoid artifacts
        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)
        if area == 0:
            continue
            
        # Draw the contour with thickness 2 (as in the original code)
        cv.drawContours(cnts_mask, [c], -1, 255, 2)
        
    # Second Guo-Hall Thinning on the drawn contours
    try:
        final_boundaries = cv.ximgproc.thinning(cnts_mask, thinningType=cv.ximgproc.THINNING_GUOHALL)
    except AttributeError:
        final_boundaries = cnts_mask
        
    if return_intermediate:
        return final_boundaries, {
            "binarized": threshed,
            "thinned_initial": thinned,
            "watershed_contours": cnts_mask,
            "watershed_labels": labels
        }

    return final_boundaries

def postprocess_boundaries(
    pred_map: np.ndarray,
    bin_thresh: float = 0.20,
    min_grain_area: int = 15,
    return_intermediate: bool = False,
    **kwargs
):
    """
    Direct and robust (Binarization -> Thinning -> Topological Watershed) pipeline.
    Uses the skeleton to define basins, naturally erasing dead-end spurs and dot noise
    without destroying valid small grains.
    """
    # 1. Format & Normalization
    pred_map = np.squeeze(pred_map).astype(np.float32)
    if pred_map.ndim != 2:
        raise ValueError(f"Expected a 2D prediction map, got shape {pred_map.shape}")

    if pred_map.max() > 1.0:
        pred_map = pred_map / 255.0
    pred_map = np.clip(pred_map, 0.0, 1.0)

    # ---------------------------------------------------------
    # STEP 1: BINARIZATION
    # ---------------------------------------------------------
    smoothed = cv.GaussianBlur(pred_map, (3, 3), 0)
    _, bin_mask = cv.threshold((smoothed * 255).astype(np.uint8), int(bin_thresh * 255), 255, cv.THRESH_BINARY)
    
    # Close small gaps in the binary mask to ensure continuous lines before thinning
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    bin_mask = cv.morphologyEx(bin_mask, cv.MORPH_CLOSE, kernel)

    # ---------------------------------------------------------
    # STEP 2: GUO-HALL THINNING
    # ---------------------------------------------------------
    try:
        thinned = cv.ximgproc.thinning(bin_mask, thinningType=cv.ximgproc.THINNING_GUOHALL)
    except AttributeError:
        thinned = bin_mask

    # ---------------------------------------------------------
    # STEP 3: WATERSHEDDING (Topological Spur Eraser)
    # ---------------------------------------------------------
    # Find the empty spaces (grain basins) between the skeleton lines
    grain_interiors = cv.bitwise_not(thinned)
    
    # CRITICAL: Use connectivity=4 so basins don't leak through diagonal skeleton junctions
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(grain_interiors, connectivity=4)
    
    markers = np.zeros_like(labels, dtype=np.int32)
    marker_id = 1
    
    for i in range(1, num_labels):
        # Only keep regions large enough to be actual grains.
        # This erases tiny trapped pockets (dot noise).
        if stats[i, cv.CC_STAT_AREA] >= min_grain_area:
            markers[labels == i] = marker_id
            marker_id += 1
            
    # Run Watershed
    # Use the smoothed prediction as the topographic map so ridges align with model probability
    topo = cv.cvtColor((smoothed * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
    markers = cv.watershed(topo, markers)
    
    # Extract Watershed ridges (-1)
    final_boundaries = np.zeros_like(bin_mask)
    final_boundaries[markers == -1] = 255
    
    # Clean borders (watershed always marks the image border)
    final_boundaries[0, :] = 0
    final_boundaries[-1, :] = 0
    final_boundaries[:, 0] = 0
    final_boundaries[:, -1] = 0

    if return_intermediate:
        return final_boundaries, {
            "binarized": bin_mask,
            "thinned": thinned,
            "watershed": final_boundaries,
        }

    return final_boundaries

def blur_watershed_postprocess(
    pred_map: np.ndarray,
    blur_ksize: int = 3,
    min_distance: int = 5,
    peak_threshold_abs: float = 0.3,
    return_intermediate: bool = False
):
    """
    Continuous Topography Pipeline (Gaussian Blur -> Peak Detection -> Watershed).
    Outputs thick boundaries mimicking the ground truth reference style.
    """
    # 1. Format & Normalization
    pred_map = np.squeeze(pred_map).astype(np.float32)
    if pred_map.ndim != 2:
        raise ValueError(f"Expected a 2D prediction map, got shape {pred_map.shape}")

    if pred_map.max() > 1.0:
        pred_map = pred_map / 255.0
    pred_map = np.clip(pred_map, 0.0, 1.0)

    # ---------------------------------------------------------
    # STEP 1: GAUSSIAN BLUR
    # ---------------------------------------------------------
    # Smooth the probability map to remove isolated noisy pixels.
    blurred_map = cv.GaussianBlur(pred_map, (blur_ksize, blur_ksize), 0)

    # ---------------------------------------------------------
    # STEP 2: FIND MARKERS (Grain Centers)
    # ---------------------------------------------------------
    grain_map = 1.0 - blurred_map

    coords = peak_local_max(
        grain_map,
        min_distance=min_distance,
        threshold_abs=peak_threshold_abs,
        exclude_border=0
    )
    
    local_max = np.zeros(grain_map.shape, dtype=bool)
    if len(coords) > 0:
        local_max[tuple(coords.T)] = True
        
    markers = ndimage.label(local_max)[0]

    # ---------------------------------------------------------
    # STEP 3: WATERSHED & BOUNDARY EXTRACTION
    # ---------------------------------------------------------
    labels = watershed(blurred_map, markers)
    
    # Extract 1-pixel thin boundaries
    boundaries = find_boundaries(labels, mode='inner')
    final_boundaries = (boundaries * 255).astype(np.uint8)
    
    # ---------------------------------------------------------
    # STEP 4: DILATION (Thicken to match reference image)
    # ---------------------------------------------------------
    # Expand the 1-pixel boundary to be thicker and more visible
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    final_boundaries = cv.dilate(final_boundaries, kernel, iterations=1)

    if return_intermediate:
        return final_boundaries, {
            "blurred": blurred_map,
            "boundaries": final_boundaries,
            "watershed_labels": labels,
        }

    return final_boundaries