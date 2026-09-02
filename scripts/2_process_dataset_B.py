import cv2
import numpy as np
import glob
import os

# Load the calibration from Dataset A
try:
    calib_data = np.load("reference_calibration.npz")
    K = calib_data['K']
    D = calib_data['D']
except:
    print("Error: Could not find reference_calibration.npz. Run Script 1 first!")
    exit()

# Same board dimensions
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.038
MARKER_LENGTH = 0.028

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
detector = cv2.aruco.CharucoDetector(board)

image_files = glob.glob('Data/dataset_B_evaluation/*.jpg')
image_files.sort()

print(f"Calculating 3D Physical Ground Truth for {len(image_files)} Dataset B images...\n")
print("-" * 50)
print(f"{'Image Name':<30} | {'Camera Distance (mm)':<20}")
print("-" * 50)

for image_file in image_files:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    
    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 4:
        # NEW OPEN CV METHOD: Match image points to physical board points
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        
        # Solve PnP gets the 3D position (Translation Vector) and rotation
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
        
        if success:
            # Calculate distance in meters, then convert to millimeters
            distance_meters = np.linalg.norm(tvec)
            distance_mm = distance_meters * 1000
            
            # Print the file name and the calculated physical distance
            filename = os.path.basename(image_file)
            print(f"{filename:<30} | {distance_mm:.1f} mm")
        else:
            print(f"{os.path.basename(image_file):<30} | Pose estimation failed")
    else:
        print(f"{os.path.basename(image_file):<30} | Board not detected")

print("-" * 50)
print("Dataset B processing complete! These are your Ground Truth measurements.")