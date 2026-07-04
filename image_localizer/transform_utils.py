"""Transform estimation and decomposition utilities.
Provides helpers to:
- compute polygon from transform
- extract translation, rotation, scale from 2x3 affine
- choose affine vs homography
- compute center, percentages
"""
from typing import Tuple, Dict
import numpy as np
import cv2


def apply_transform_to_corners(transform: np.ndarray, w: int, h: int) -> np.ndarray:
    """Apply 2x3 affine or 3x3 homography to template corners.
    Returns 4x2 array of transformed corners in target image coords: [[x,y],...]
    Corners order: (0,0),(w,0),(w,h),(0,h)
    """
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    if transform.shape == (2,3):
        pts = cv2.transform(corners.reshape(-1,1,2), transform).reshape(-1,2)
    elif transform.shape == (3,3):
        pts = cv2.perspectiveTransform(corners.reshape(-1,1,2), transform).reshape(-1,2)
    else:
        raise ValueError('Unsupported transform shape')
    return pts


def affine_params_from_matrix(A: np.ndarray) -> Dict[str,float]:
    """Decompose 2x3 affine matrix A = [a b tx; c d ty]
    Returns translation tx,ty; rotation (deg); scale_x, scale_y; shear.

    Math summary:
      [a b] = R * S where R is rotation and S is scale/shear matrix
      scale_x = sqrt(a^2 + c^2)
      scale_y = sqrt(b^2 + d^2)
      rotation = atan2(c, a) (radians)
    """
    if A.shape != (2,3):
        raise ValueError('Affine matrix must be 2x3')
    a, b, tx = A[0]
    c, d, ty = A[1]
    scale_x = np.sqrt(a*a + c*c)
    scale_y = np.sqrt(b*b + d*d)
    # rotation (radians) — using a,c which are first column of linear part
    rotation = np.degrees(np.arctan2(c, a))
    # Shear can be approximated
    shear = (a*b + c*d) / (scale_x*scale_x) if scale_x != 0 else 0.0
    return {
        'tx': float(tx), 'ty': float(ty),
        'rotation_deg': float(rotation),
        'scale_x': float(scale_x), 'scale_y': float(scale_y),
        'shear': float(shear)
    }


def bounding_rect_from_polygon(pts: np.ndarray) -> Tuple[int,int,int,int]:
    xs = pts[:,0]
    ys = pts[:,1]
    x = int(np.min(xs))
    y = int(np.min(ys))
    w = int(np.max(xs) - x)
    h = int(np.max(ys) - y)
    return x, y, w, h


def center_and_percent(pts: np.ndarray, target_w: int, target_h: int) -> Dict[str,float]:
    cx = float(np.mean(pts[:,0]))
    cy = float(np.mean(pts[:,1]))
    return {
        'center_x': cx,
        'center_y': cy,
        'center_x_pct': cx / float(target_w),
        'center_y_pct': cy / float(target_h)
    }


def is_perspective_significant(homography: np.ndarray, affine: np.ndarray, tol: float = 1e-2) -> bool:
    """Compare homography and affine (promoted to 3x3) — return True if perspective terms matter.
    Uses relative difference on the last row/column elements.
    """
    if homography is None or affine is None:
        return True
    H = homography / homography[2,2]
    A3 = np.vstack([affine, np.array([0.0,0.0,1.0])])
    diff = np.abs(H - A3)
    # focus on perspective components H[2,0:2]
    pers = diff[2,0:2].max()
    return pers > tol
