from pathlib import Path

import cv2
import numpy as np

from .config import BOARD


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def image_paths(directory: Path) -> list[Path]:
    paths = [path for pattern in IMAGE_EXTENSIONS for path in directory.glob(pattern)]
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No supported images found in {directory}")
    return paths


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def create_charuco_detector() -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
    dictionary = cv2.aruco.getPredefinedDictionary(BOARD.dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (BOARD.squares_x, BOARD.squares_y),
        BOARD.square_length,
        BOARD.marker_length,
        dictionary,
    )
    return board, cv2.aruco.CharucoDetector(board)


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with np.load(path) as data:
        if "K" not in data or "D" not in data:
            raise ValueError(f"Calibration file is missing K or D: {path}")
        return data["K"], data["D"]
