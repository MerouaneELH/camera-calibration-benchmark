import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import DATASET_B_UNDISTORTED_DIR, OUTPUT_DIR, VISUALIZATION_DIR, ensure_output_directories
from benchmark.io import image_paths
from geocalib import GeoCalib
from geocalib import viz2d
from geocalib.perspective_fields import get_perspective_field

# Create folders for outputs
VIS_DIR = VISUALIZATION_DIR / "geocalib"
ensure_output_directories()
VIS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)

predictions = {}
input_paths = image_paths(DATASET_B_UNDISTORTED_DIR)

for img_path in input_paths:
    filename = img_path.name
    
    # GeoCalib loads the image as a tensor of shape [C, H, W]
    img_tensor = model.load_image(img_path).to(device)
    
    with torch.no_grad():
        result = model.calibrate(img_tensor, camera_model="pinhole")
    
    camera = result["camera"]
    gravity = result["gravity"]
    
    # 1. Save the Predicted Intrinsic Matrix
    K_pred = camera.K.cpu().numpy()[0] if hasattr(camera, 'K') else camera["K"]
    predictions[filename] = K_pred
    
    # 2. Extract vectors for visualization
    up, lat = get_perspective_field(camera, gravity)
    
    # 3. Draw using GeoCalib's built-in Matplotlib tools
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy() # Convert [C, H, W] to [H, W, C]
    fig = viz2d.plot_images([img_np], pad=0)
    ax = fig.get_axes()
    
    # Overlay the gravity up-vectors and latitude curves
    viz2d.plot_vector_fields([up[0].cpu()], axes=[ax[0]])
    viz2d.plot_latitudes([lat[0, 0].cpu()], axes=[ax[0]])
    
    # 4. Convert the Matplotlib figure to an OpenCV image and save it
    fig.canvas.draw()
    vis_img = np.array(fig.canvas.renderer.buffer_rgba())
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2BGR)
    
    vis_path = VIS_DIR / f"vis_{filename}"
    cv2.imwrite(vis_path, vis_img)
    
    # CRITICAL: Close the figure to prevent RAM memory leaks
    plt.close(fig) 
    
    print(f"GeoCalib processed & visualized: {filename}")

if not predictions:
    raise SystemExit("No GeoCalib predictions were produced.")

output_path = OUTPUT_DIR / "preds_geocalib.npz"
np.savez(output_path, **predictions)
print(f"Done! Visualizations saved to '{VIS_DIR}'.")