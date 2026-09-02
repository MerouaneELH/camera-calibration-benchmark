import os, glob, torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from geocalib import GeoCalib
from geocalib import viz2d
from geocalib.perspective_fields import get_perspective_field

# Create folders for outputs
vis_folder = 'visualizations/geocalib'
os.makedirs(vis_folder, exist_ok=True)
os.makedirs('Outputs', exist_ok=True) # Ensures the save folder exists

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)

predictions = {}
for img_path in glob.glob('Data/dataset_B_evaluation/*.jpg'):
    filename = os.path.basename(img_path)
    
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
    
    vis_path = os.path.join(vis_folder, f"vis_{filename}")
    cv2.imwrite(vis_path, vis_img)
    
    # CRITICAL: Close the figure to prevent RAM memory leaks
    plt.close(fig) 
    
    print(f"GeoCalib processed & visualized: {filename}")

np.savez("Outputs/preds_geocalib.npz", **predictions)
print(f"Done! Visualizations saved to '{vis_folder}'.")