"""Masked multi-scale template matching utilities."""
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .transform_utils import center_and_percent


def _crop_to_mask(template: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(mask)
    if points is None:
        return template, mask
    x, y, w, h = cv2.boundingRect(points)
    return template[y:y + h, x:x + w], mask[y:y + h, x:x + w]


def find_template_masked(
    template_path: str,
    target_path: str,
    min_scale: float = 0.08,
    max_scale: float = 1.25,
    steps: int = 80,
    white_threshold: int = 245,
    min_match_width: int = 32,
    min_match_height: int = 32,
    confidence_threshold: float = 0.6,
) -> Dict[str, Any]:
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    target = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if template is None or target is None:
        raise FileNotFoundError('Template or target image not found')

    target_h, target_w = target.shape[:2]
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(template_gray, 0, white_threshold - 1)
    template, mask = _crop_to_mask(template, mask)

    best = {
        'score': -1.0,
        'color_score': -1.0,
        'gray_score': -1.0,
        'edge_score': -1.0,
        'scale': None,
        'location': None,
        'shape': None,
    }

    for scale in np.linspace(min_scale, max_scale, steps):
        width = max(2, int(round(template.shape[1] * scale)))
        height = max(2, int(round(template.shape[0] * scale)))
        if width < min_match_width or height < min_match_height or width >= target_w or height >= target_h:
            continue

        resized_template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        resized_gray = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if cv2.countNonZero(resized_mask) == 0:
            continue

        color_response = cv2.matchTemplate(target, resized_template, cv2.TM_CCORR_NORMED, mask=resized_mask)
        color_response = np.nan_to_num(color_response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        gray_response = cv2.matchTemplate(target_gray, resized_gray, cv2.TM_CCOEFF_NORMED, mask=resized_mask)
        gray_response = np.nan_to_num(gray_response, nan=-1.0, posinf=-1.0, neginf=-1.0)

        candidate_response = (0.55 * color_response) + (0.45 * np.maximum(gray_response, 0.0))
        _, _, _, max_loc = cv2.minMaxLoc(candidate_response)
        x, y = max_loc
        color_score = float(color_response[y, x])
        gray_score = float(gray_response[y, x])
        edge_score = _edge_agreement(resized_gray, target_gray[y:y + height, x:x + width], resized_mask)
        score = (0.45 * color_score) + (0.35 * max(0.0, gray_score)) + (0.20 * edge_score)

        if score > best['score']:
            best.update({
                'score': float(score),
                'color_score': color_score,
                'gray_score': gray_score,
                'edge_score': float(edge_score),
                'scale': float(scale),
                'location': max_loc,
                'shape': (width, height),
            })

    if best['location'] is None or best['shape'] is None:
        return {
            'method': 'masked_template',
            'template_path': template_path,
            'target_path': target_path,
            'confidence': 0.0,
            'confidence_threshold': confidence_threshold,
            'match_found': False,
            'error': 'template_match_failed',
        }

    x, y = best['location']
    width, height = best['shape']
    polygon = np.array([
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ], dtype=np.float32)
    center = center_and_percent(polygon, target_w, target_h)

    match_found = best['score'] >= confidence_threshold
    result = {
        'method': 'masked_template',
        'template_path': template_path,
        'target_path': target_path,
        'transform_type': 'axis_aligned_scale',
        'match_score': best['score'],
        'color_score': best['color_score'],
        'gray_score': best['gray_score'],
        'edge_score': best['edge_score'],
        'confidence': best['score'],
        'confidence_threshold': confidence_threshold,
        'match_found': match_found,
        'scale': best['scale'],
        'best_candidate_polygon': polygon.tolist(),
        'best_candidate_bbox': (x, y, width, height),
        'center_x': center['center_x'],
        'center_y': center['center_y'],
        'center_x_pct': center['center_x_pct'],
        'center_y_pct': center['center_y_pct'],
        'width_px': width,
        'height_px': height,
        'width_pct': width / float(target_w),
        'height_pct': height / float(target_h),
        'scale_x': best['scale'],
        'scale_y': best['scale'],
        'rotation_deg': 0.0,
    }
    if match_found:
        result.update({
            'polygon': polygon.tolist(),
            'bbox': (x, y, width, height),
        })
    return result


def _edge_agreement(template_gray: np.ndarray, target_patch_gray: np.ndarray, mask: np.ndarray) -> float:
    if target_patch_gray.shape[:2] != template_gray.shape[:2]:
        return 0.0

    template_edges = cv2.Canny(template_gray, 60, 160) > 0
    patch_edges = cv2.Canny(target_patch_gray, 60, 160) > 0
    active_mask = mask > 0
    template_edges = np.logical_and(template_edges, active_mask)
    patch_edges = np.logical_and(patch_edges, active_mask)

    denominator = int(template_edges.sum() + patch_edges.sum())
    if denominator == 0:
        return 0.0
    overlap = int(np.logical_and(template_edges, patch_edges).sum())
    return float((2.0 * overlap) / denominator)
