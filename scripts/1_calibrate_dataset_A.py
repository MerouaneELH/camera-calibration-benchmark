import cv2
import numpy as np
import glob

# ==========================================
# 1. PHYSICAL BOARD SETTINGS (38mm)
# ==========================================
SQUARES_X = 5          
SQUARES_Y = 7          
SQUARE_LENGTH = 0.038  
MARKER_LENGTH = 0.029  

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
detector = cv2.aruco.CharucoDetector(board)

# ==========================================
# 2. PROCESS IMAGES
# ==========================================
image_files = glob.glob('Data/dataset_A_reference/*.jpg')
image_files.sort()

all_obj_points = []
all_img_points = []
image_size = None

print(f"Processing {len(image_files)} images from Dataset A...")

for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = (gray.shape[1], gray.shape[0])

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 5:
        # NEW OPEN CV METHOD: Match the 2D corners to the 3D physical board coordinates
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        
        if obj_points is not None and img_points is not None and len(obj_points) > 5:
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
    else:
        print(f"Skipped {image_file}: Not enough corners detected.")

# ==========================================
# 3. CALCULATE CALIBRATION
# ==========================================
if len(all_obj_points) == 0:
    print("Error: No corners were successfully matched. Check your images.")
    exit()

print("\nCalculating K Matrix...")
# Standard OpenCV camera calibration
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_points, 
    all_img_points, 
    image_size, 
    None, 
    None
)

print(f"\n✅ CALIBRATION SUCCESSFUL ✅")
print(f"Reprojection Error: {ret:.3f} pixels (Lower is better, aim for < 1.0)")
print("\nReference K Matrix:")
print(np.round(K, 2))

# Save for Dataset B
np.savez("reference_calibration.npz", K=K, D=D)
print("\nSaved reference parameters to 'reference_calibration.npz'.")