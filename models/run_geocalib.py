import os, glob, torch
import numpy as np
import cv2
from geocalib import GeoCalib

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)

predictions = {}
for img_path in glob.glob('Data/dataset_B_evaluation/*.jpg'):
    filename = os.path.basename(img_path)
    img_tensor = model.load_image(img_path).to(device)
    
    with torch.no_grad():
        result = model.calibrate(img_tensor, camera_model="pinhole")
    
    K_pred = result["camera"].K.cpu().numpy()[0] if hasattr(result["camera"], 'K') else result["camera"]["K"]
    predictions[filename] = K_pred
    print(f"GeoCalib processed: {filename}")

np.savez("Outputs/preds_geocalib.npz", **predictions)