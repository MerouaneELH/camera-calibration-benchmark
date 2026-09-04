"""Validated image and calibration I/O plus ChArUco construction helpers."""

from pathlib import Path
import csv
import hashlib

import cv2
import numpy as np

from .config import BOARD


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def image_paths(directory: Path) -> list[Path]:
    """Return sorted supported image paths, failing when a dataset is empty.

    Args:
        directory: Directory containing input images.

    Raises:
        FileNotFoundError: If no JPG, JPEG, or PNG files are present.
    """

    paths = [path for pattern in IMAGE_EXTENSIONS for path in directory.glob(pattern)]
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No supported images found in {directory}")
    return paths


def load_dataset_metadata(path: Path, images: list[Path]) -> dict[str, dict[str, str]]:
    """Load and validate physical metadata for every Dataset B image.

    The CSV is experimental ground truth supplied by the experimenter. Values
    are never inferred from filenames or image geometry.
    """
    required = {"image", "physical_distance_mm", "viewpoint_angle_deg", "horizontal_position"}
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Dataset metadata is empty: {path}")
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {', '.join(sorted(missing_columns))}")

    image_names = {image.name for image in images}
    metadata: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        name = (row.get("image") or "").strip()
        if not name:
            raise ValueError(f"Metadata row {row_number} has an empty image value")
        if name in metadata:
            raise ValueError(f"Duplicate metadata row for image '{name}'")
        try:
            distance = float(row["physical_distance_mm"])
            angle = float(row["viewpoint_angle_deg"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Metadata row {row_number} has non-numeric distance or angle") from error
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError(f"Metadata row {row_number} has invalid physical_distance_mm: {distance}")
        if not np.isfinite(angle):
            raise ValueError(f"Metadata row {row_number} has invalid viewpoint_angle_deg: {angle}")
        if not (row.get("horizontal_position") or "").strip():
            raise ValueError(f"Metadata row {row_number} has an empty horizontal_position")
        metadata[name] = row

    extra = sorted(set(metadata) - image_names)
    missing = sorted(image_names - set(metadata))
    if extra:
        raise ValueError(f"Metadata references nonexistent images: {', '.join(extra)}")
    if missing:
        raise ValueError(f"Images missing metadata rows: {', '.join(missing)}")
    return metadata


def read_image(path: Path) -> np.ndarray:
    """Read a BGR image and convert OpenCV's ``None`` failure into an exception."""

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def file_digest(path: Path) -> str:
    """Return a SHA-256 digest used to detect stale generated artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_charuco_detector() -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
    """Build the board detector from the single shared board configuration."""

    dictionary = cv2.aruco.getPredefinedDictionary(BOARD.dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (BOARD.squares_x, BOARD.squares_y),
        BOARD.square_length,
        BOARD.marker_length,
        dictionary,
    )
    return board, cv2.aruco.CharucoDetector(board)


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and minimally validate the intrinsic matrix and distortion vector."""

    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with np.load(path) as data:
        if "K" not in data or "D" not in data:
            raise ValueError(f"Calibration file is missing K or D: {path}")
        return data["K"], data["D"]
