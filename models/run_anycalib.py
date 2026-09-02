import os, glob, torch
import numpy as np
import torchvision.transforms.functional as TF
import cv2
from anycalib import AnyCalib

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AnyCalib(model_id='anycalib_pinhole').to(device)

predictions = {}
for img_path in glob.glob('Data/dataset_B_evaluation/*.jpg'):
    filename = os.path.basename(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.predict(img_tensor, cam_id="pinhole")
    
    intrinsics = output["intrinsics"][0].cpu().numpy()
    K_pred = np.array([[intrinsics[0], 0, intrinsics[2]],
                       [0, intrinsics[1], intrinsics[3]],
                       [0, 0, 1]])
    predictions[filename] = K_pred
    print(f"AnyCalib processed: {filename}")

np.savez("preds_anycalib.npz", **predictions)