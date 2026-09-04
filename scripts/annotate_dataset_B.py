"""Manually annotate Dataset B object corners and experimental setup height.

Each image is displayed for four clicks in the order P1 top-left, P2 top-right,
P3 bottom-right, P4 bottom-left. Coordinates are converted back to original
image pixels when the display is resized. Press ``r`` to reset, Enter to save,
``h`` to change the current setup height, and ``q`` to quit. Existing rows are
updated by filename instead of duplicated.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import (
    DATASET_B_DIR,
    DATASET_B_METADATA_PATH,
    OBJECT_LENGTH_MM,
    OBJECT_WIDTH_MM,
)
from benchmark.io import image_paths

WINDOW = "Dataset B annotation"


def load_rows() -> dict[str, dict[str, str]]:
    """Load existing annotations so rerunning updates rows by image name."""
    if not DATASET_B_METADATA_PATH.exists():
        return {}
    required = {"experimental_setup_height_mm", "true_length_mm", "true_width_mm"}
    with DATASET_B_METADATA_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            if row.get("image") and required.issubset(row):
                rows[row["image"]] = row
        return rows


def save_rows(rows: dict[str, dict[str, str]]) -> None:
    """Write annotations with a stable, validator-compatible column order."""
    fields = ["image", "experimental_setup_height_mm", "true_length_mm", "true_width_mm"]
    fields += [f"p{index}_{axis}" for index in range(1, 5) for axis in ("x", "y")]
    DATASET_B_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_B_METADATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[name] for name in sorted(rows))


def annotate_image(path: Path, height_mm: float, true_length_mm: float, true_width_mm: float) -> dict[str, str] | None:
    """Collect four original-resolution clicks for one image."""
    image = cv2.imread(str(path))
    if image is None:
        print(f"Skipping unreadable image: {path}")
        return None
    original_height, original_width = image.shape[:2]
    max_width, max_height = 1400, 900
    scale = min(1.0, max_width / original_width, max_height / original_height)
    display = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    points: list[tuple[int, int]] = []

    def click(_event, x, y, _flags, _param):
        if _event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((round(x / scale), round(y / scale)))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, click)
    while True:
        canvas = display.copy()
        display_points = [(round(x * scale), round(y * scale)) for x, y in points]
        for index, point in enumerate(display_points):
            cv2.circle(canvas, point, 6, (0, 255, 0), -1)
            cv2.putText(canvas, f"P{index + 1}", (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(display_points) > 1:
            cv2.polylines(canvas, [np.asarray(display_points)], False, (0, 255, 255), 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("h"):
            value = input(f"Setup height for {path.name} in mm [{height_mm}]: ").strip()
            if value:
                height_mm = float(value)
        elif key == 13 and len(points) == 4:
            cv2.destroyWindow(WINDOW)
            row = {"image": path.name, "experimental_setup_height_mm": str(height_mm), "true_length_mm": str(true_length_mm), "true_width_mm": str(true_width_mm)}
            row.update({f"p{index}_{axis}": str(point[axis_index]) for index, point in enumerate(points, 1) for axis, axis_index in (("x", 0), ("y", 1))})
            return row
        elif key == ord("q"):
            cv2.destroyWindow(WINDOW)
            return None
    

def main() -> None:
    """Annotate all Dataset B images using the configured rectangle dimensions."""
    rows = load_rows()
    images = image_paths(DATASET_B_DIR)
    height_mm = float(input("Experimental setup height in mm: "))
    true_length_mm = OBJECT_LENGTH_MM
    true_width_mm = OBJECT_WIDTH_MM
    print(f"Using measured rectangle dimensions: {true_length_mm:.1f} mm x {true_width_mm:.1f} mm")
    for path in images:
        print(f"Annotating {path.name}; click P1, P2, P3, P4.")
        row = annotate_image(path, height_mm, true_length_mm, true_width_mm)
        if row is None:
            break
        rows[path.name] = row
        save_rows(rows)
        print(f"Saved {path.name} to {DATASET_B_METADATA_PATH}")


if __name__ == "__main__":
    main()