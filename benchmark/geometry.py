"""Geometry helpers shared by benchmark evaluation stages."""

import cv2
import numpy as np


def perpendicular_distance_mm(rvec: np.ndarray, tvec: np.ndarray) -> float:
    """Return camera-center distance perpendicular to a board plane at Z=0."""
    rotation, _ = cv2.Rodrigues(rvec)
    camera_center = -rotation.T @ tvec.reshape(3, 1)
    return float(abs(camera_center[2, 0]) * 1000.0)


def board_plane_points_mm(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Intersect image rays with the ChArUco board plane and return mm coordinates.

    ChArUco board coordinates use Z=0. This uses calibrated ray-plane
    geometry, rather than a pixels-to-millimeters scale approximation.
    """
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(points, camera_matrix, distortion).reshape(-1, 2)
    rays = np.column_stack((rays, np.ones(len(rays))))
    rotation, _ = cv2.Rodrigues(rvec)
    ray_board = (rotation.T @ rays.T).T
    origin_board = rotation.T @ tvec.reshape(3)
    scale = -origin_board[2] / ray_board[:, 2]
    return ((scale[:, None] * ray_board) - origin_board)[:, :2] * 1000.0


def object_dimensions_mm(object_points: np.ndarray) -> tuple[float, float]:
    """Return length P1-P2 and width P1-P4 from ordered 2D plane points."""
    points = np.asarray(object_points, dtype=np.float64).reshape(4, 2)
    return float(np.linalg.norm(points[1] - points[0])), float(np.linalg.norm(points[3] - points[0]))
