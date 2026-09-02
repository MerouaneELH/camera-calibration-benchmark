import sys
import torch
import math
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import DATASET_B_UNDISTORTED_DIR, OUTPUT_DIR, VISUALIZATION_DIR, ensure_output_directories
from benchmark.io import image_paths, read_image
from perspective2d import PerspectiveFields
from perspective2d.utils import draw_from_r_p_f_cx_cy

# Create a folder to save the cool visual outputs
VIS_DIR = VISUALIZATION_DIR / "perspective_fields"
ensure_output_directories()
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Load the AI model
print("Loading Perspective Fields model...")
model = PerspectiveFields('Paramnet-360Cities-edina-uncentered').eval().cpu()

predictions = {}
input_paths = image_paths(DATASET_B_UNDISTORTED_DIR)

# Process each image
for img_path in input_paths:
    filename = img_path.name
    try:
        img_bgr = read_image(img_path)
    except ValueError as error:
        print(f"Skipping {error}")
        continue
        continue
    h, w = img_bgr.shape[:2]
    
    # 1. AI Inference
    with torch.no_grad():
        pred = model.inference(img_bgr=img_bgr)
    
    # 2. Extract Math (Focal Length & Center)
    vfov_rad = math.radians(pred['pred_general_vfov'].item())
    focal_length = (h / 2.0) / math.tan(vfov_rad / 2.0)
    
    # The model outputs relative center coordinates (e.g. 0.5 for the middle), 
    # so we multiply by width (w) and height (h) to get actual pixels.
    cx = pred['pred_rel_cx'].item() * w 
    cy = pred['pred_rel_cy'].item() * h
    K_pred = np.array([[focal_length, 0, cx],
                       [0, focal_length, cy],
                       [0, 0, 1]])
    predictions[filename] = K_pred
    
    # 3. DRAW THE VECTORS! 
    # Convert image to RGB for the drawing tool
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # The new function mathematically projects the grid using the 5 camera parameters
    blend_rgb = draw_from_r_p_f_cx_cy(
        img_rgb,
        pred['pred_roll'].item(),
        pred['pred_pitch'].item(),
        pred['pred_general_vfov'].item(),
        pred['pred_rel_cx'].item(),
        pred['pred_rel_cy'].item(),
        "deg"
    ).astype(np.uint8)

    # Convert back to BGR so OpenCV can save it correctly
    blend_bgr = cv2.cvtColor(blend_rgb, cv2.COLOR_RGB2BGR)
    
    # Save the visualization
    vis_path = VIS_DIR / f"vis_{filename}"
    cv2.imwrite(vis_path, blend_bgr)
    
    print(f"Processed & Visualized: {filename}")

# Save the mathematical matrices for the evaluator script
if not predictions:
    raise SystemExit("No Perspective Fields predictions were produced.")

output_path = OUTPUT_DIR / "preds_perspective.npz"
np.savez(output_path, **predictions)
print(f"Done! Check the '{VIS_DIR}' folder for your images.")