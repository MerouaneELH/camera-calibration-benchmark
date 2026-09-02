import os, glob, torch, math
import numpy as np
import cv2
from perspective2d import PerspectiveFields
from perspective2d.utils import draw_perspective_fields

# Create a folder to save the cool visual outputs
vis_folder = 'visualizations/perspective_fields'
os.makedirs(vis_folder, exist_ok=True)

# Load the AI model
print("Loading Perspective Fields model...")
model = PerspectiveFields('Paramnet-360Cities-edina-uncentered').eval().cpu()

predictions = {}

# Process each image
for img_path in glob.glob('dataset_B_evaluation/*.jpg'):
    filename = os.path.basename(img_path)
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    
    # 1. AI Inference
    with torch.no_grad():
        pred = model.inference(img_bgr=img_bgr)
    
    # 2. Extract Math (Focal Length & Center)
    vfov_rad = math.radians(pred['vfov'].item())
    focal_length = (h / 2.0) / math.tan(vfov_rad / 2.0)
    
    cx = pred['cx'].item() * w if pred['cx'].item() < 2 else pred['cx'].item()
    cy = pred['cy'].item() * h if pred['cy'].item() < 2 else pred['cy'].item()
    
    K_pred = np.array([[focal_length, 0, cx],
                       [0, focal_length, cy],
                       [0, 0, 1]])
    predictions[filename] = K_pred
    
    # 3. DRAW THE VECTORS! 
    # Convert image to RGB for the drawing tool
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Extract the raw vector maps the AI predicted
    up_vectors = pred['up']
    latitude = pred['lati']
    
    # Blend the vectors onto the image
    blend_rgb = draw_perspective_fields(img_rgb, up_vectors, latitude)
    
    # Convert back to BGR so OpenCV can save it correctly
    blend_bgr = cv2.cvtColor(blend_rgb, cv2.COLOR_RGB2BGR)
    
    # Save the visualization
    vis_path = os.path.join(vis_folder, f"vis_{filename}")
    cv2.imwrite(vis_path, blend_bgr)
    
    print(f"Processed & Visualized: {filename}")

# Save the mathematical matrices for the evaluator script
np.savez("preds_perspective.npz", **predictions)
print(f"Done! Check the '{vis_folder}' folder for your images.")