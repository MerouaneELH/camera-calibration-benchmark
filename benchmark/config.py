from dataclasses import dataclass
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUTPUT_DIR = ROOT / "Outputs"
VISUALIZATION_DIR = ROOT / "visualizations"


@dataclass(frozen=True)
class BoardConfig:
    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.038
    marker_length: float = 0.029
    dictionary_id: int = cv2.aruco.DICT_5X5_100


BOARD = BoardConfig()
CALIBRATION_PATH = OUTPUT_DIR / "reference_calibration.npz"
DATASET_A_DIR = DATA_DIR / "dataset_A_reference"
DATASET_B_DIR = DATA_DIR / "dataset_B_evaluation"
DATASET_B_UNDISTORTED_DIR = DATA_DIR / "dataset_B_undistorted"


def ensure_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
