"""Masked multi-scale template matching utilities."""
from typing import Any, Dict, List, Tuple

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
    max_detections: int = 20,
    nms_iou_threshold: float = 0.3,
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

    candidates: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {}

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
        candidate_locations = _find_candidate_locations(
            candidate_response,
            min_response=max(0.05, confidence_threshold * 0.5),
            max_locations=max_detections * 4,
        )

        for x, y in candidate_locations:
            color_score = float(color_response[y, x])
            gray_score = float(gray_response[y, x])
            edge_score = _edge_agreement(resized_gray, target_gray[y:y + height, x:x + width], resized_mask)
            score = float((0.45 * color_score) + (0.35 * max(0.0, gray_score)) + (0.20 * edge_score))
            candidate = _build_match(
                template_path=template_path,
                target_path=target_path,
                x=x,
                y=y,
                width=width,
                height=height,
                target_w=target_w,
                target_h=target_h,
                scale=float(scale),
                score=score,
                color_score=color_score,
                gray_score=gray_score,
                edge_score=float(edge_score),
            )
            if not best or score > best['confidence']:
                best = candidate
            if score >= confidence_threshold:
                candidates.append(candidate)

    if not best:
        return {
            'method': 'masked_template',
            'template_path': template_path,
            'target_path': target_path,
            'confidence': 0.0,
            'confidence_threshold': confidence_threshold,
            'match_found': False,
            'match_count': 0,
            'matches': [],
            'error': 'template_match_failed',
        }

    matches = _non_max_suppression(candidates, nms_iou_threshold)[:max_detections]
    match_found = len(matches) > 0
    primary = matches[0] if match_found else best
    result = {
        'method': 'masked_template',
        'template_path': template_path,
        'target_path': target_path,
        'transform_type': 'axis_aligned_scale',
        'match_score': primary['confidence'],
        'color_score': primary['color_score'],
        'gray_score': primary['gray_score'],
        'edge_score': primary['edge_score'],
        'confidence': primary['confidence'],
        'confidence_threshold': confidence_threshold,
        'match_found': match_found,
        'match_count': len(matches),
        'matches': matches,
        'scale': primary['scale'],
        'best_candidate_polygon': best['polygon'],
        'best_candidate_bbox': best['bbox'],
        'center_x': primary['center_x'],
        'center_y': primary['center_y'],
        'center_x_pct': primary['center_x_pct'],
        'center_y_pct': primary['center_y_pct'],
        'width_px': primary['width_px'],
        'height_px': primary['height_px'],
        'width_pct': primary['width_pct'],
        'height_pct': primary['height_pct'],
        'scale_x': primary['scale_x'],
        'scale_y': primary['scale_y'],
        'rotation_deg': 0.0,
    }
    if match_found:
        result.update({
            'polygon': primary['polygon'],
            'bbox': primary['bbox'],
        })
    return result


def _build_match(
    template_path: str,
    target_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
    target_w: int,
    target_h: int,
    scale: float,
    score: float,
    color_score: float,
    gray_score: float,
    edge_score: float,
) -> Dict[str, Any]:
    polygon = np.array([
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ], dtype=np.float32)
    center = center_and_percent(polygon, target_w, target_h)
    return {
        'method': 'masked_template',
        'template_path': template_path,
        'target_path': target_path,
        'transform_type': 'axis_aligned_scale',
        'confidence': score,
        'match_score': score,
        'color_score': color_score,
        'gray_score': gray_score,
        'edge_score': edge_score,
        'scale': scale,
        'polygon': polygon.tolist(),
        'bbox': [int(x), int(y), int(width), int(height)],
        'center_x': center['center_x'],
        'center_y': center['center_y'],
        'center_x_pct': center['center_x_pct'],
        'center_y_pct': center['center_y_pct'],
        'width_px': int(width),
        'height_px': int(height),
        'width_pct': width / float(target_w),
        'height_pct': height / float(target_h),
        'scale_x': scale,
        'scale_y': scale,
        'rotation_deg': 0.0,
    }


def _find_candidate_locations(response: np.ndarray, min_response: float, max_locations: int) -> List[Tuple[int, int]]:
    response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
    local_max = response == cv2.dilate(response, np.ones((3, 3), dtype=np.uint8))
    ys, xs = np.where(np.logical_and(local_max, response >= min_response))
    if len(xs) == 0:
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        return [max_loc] if max_val >= min_response else []

    scores = response[ys, xs]
    order = np.argsort(scores)[::-1][:max_locations]
    return [(int(xs[i]), int(ys[i])) for i in order]


def _non_max_suppression(matches: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for match in sorted(matches, key=lambda item: item['confidence'], reverse=True):
        if all(_bbox_iou(match['bbox'], kept['bbox']) <= iou_threshold for kept in selected):
            selected.append(match)
    return selected


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = (aw * ah) + (bw * bh) - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


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
