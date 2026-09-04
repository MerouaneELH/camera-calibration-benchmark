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
    DATASET_B_SETUP_HEIGHTS_MM,
    OBJECT_LENGTH_MM,
    OBJECT_WIDTH_MM,
)
from benchmark.io import image_paths

WINDOW = "Dataset B annotation"
CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 900


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
    """Collect four original-resolution clicks with zoom and pan controls."""
    image = cv2.imread(str(path))
    if image is None:
        print(f"Skipping unreadable image: {path}")
        return None
    original_height, original_width = image.shape[:2]
    base_scale = min(1.0, CANVAS_WIDTH / original_width, CANVAS_HEIGHT / original_height)
    zoom = 1.0
    pan_x = (CANVAS_WIDTH - original_width * base_scale) / 2
    pan_y = (CANVAS_HEIGHT - original_height * base_scale) / 2
    points: list[tuple[int, int]] = []

    def image_to_screen(point: tuple[int, int]) -> tuple[int, int]:
        display_scale = base_scale * zoom
        return round(point[0] * display_scale + pan_x), round(point[1] * display_scale + pan_y)

    def screen_to_image(x: int, y: int) -> tuple[int, int]:
        display_scale = base_scale * zoom
        return round((x - pan_x) / display_scale), round((y - pan_y) / display_scale)

    def click(event, x, y, _flags, _param):
        nonlocal zoom, pan_x, pan_y
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            point = screen_to_image(x, y)
            if 0 <= point[0] < original_width and 0 <= point[1] < original_height:
                points.append(point)
        elif event == cv2.EVENT_MOUSEWHEEL:
            old_scale = base_scale * zoom
            old_image_x = (x - pan_x) / old_scale
            old_image_y = (y - pan_y) / old_scale
            zoom *= 1.2 if _flags > 0 else 1 / 1.2
            zoom = min(8.0, max(1.0, zoom))
            new_scale = base_scale * zoom
            pan_x = x - old_image_x * new_scale
            pan_y = y - old_image_y * new_scale

    def render() -> np.ndarray:
        display_scale = base_scale * zoom
        resized = cv2.resize(image, None, fx=display_scale, fy=display_scale)
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        x0, y0 = round(pan_x), round(pan_y)
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(CANVAS_WIDTH, x0 + resized.shape[1]), min(CANVAS_HEIGHT, y0 + resized.shape[0])
        if x1 < x2 and y1 < y2:
            canvas[y1:y2, x1:x2] = resized[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        return canvas

    def move_pan(dx: int, dy: int) -> None:
        nonlocal pan_x, pan_y
        pan_x += dx
        pan_y += dy

    def reset_view() -> None:
        nonlocal zoom, pan_x, pan_y
        zoom = 1.0
        pan_x = (CANVAS_WIDTH - original_width * base_scale) / 2
        pan_y = (CANVAS_HEIGHT - original_height * base_scale) / 2

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CANVAS_WIDTH, CANVAS_HEIGHT)
    cv2.setMouseCallback(WINDOW, click)
    while True:
        canvas = render()
        display_points = [image_to_screen(point) for point in points]
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
        elif key in (ord("z"), ord("0")):
            reset_view()
        elif key in (81, ord("a")):
            move_pan(-50, 0)
        elif key in (83, ord("d")):
            move_pan(50, 0)
        elif key in (82, ord("w")):
            move_pan(0, -50)
        elif key in (84, ord("s")):
            move_pan(0, 50)
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
    true_length_mm = OBJECT_LENGTH_MM
    true_width_mm = OBJECT_WIDTH_MM
    print(f"Using measured rectangle dimensions: {true_length_mm:.1f} mm x {true_width_mm:.1f} mm")
    for path in images:
        group = path.stem.split(".", 1)[0]
        if group not in DATASET_B_SETUP_HEIGHTS_MM:
            raise ValueError(f"Cannot determine setup-height group from filename: {path.name}")
        height_mm = DATASET_B_SETUP_HEIGHTS_MM[group]
        print(f"Annotating {path.name}; setup height={height_mm:.0f} mm; click P1, P2, P3, P4.")
        row = annotate_image(path, height_mm, true_length_mm, true_width_mm)
        if row is None:
            break
        rows[path.name] = row
        save_rows(rows)
        print(f"Saved {path.name} to {DATASET_B_METADATA_PATH}")


if __name__ == "__main__":
    main()