"""Visualization utilities: draw polygon, rectangle, center, and overlay.
"""
from typing import Dict
import cv2
import numpy as np


def draw_result(target_path: str, result: Dict, out_path: str = 'detection_visual.png') -> None:
    img = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(target_path)
    matches = result.get('matches') or [result]
    for idx, match in enumerate(matches, start=1):
        if 'polygon' not in match or 'bbox' not in match:
            continue
        pts = np.array(match['polygon'], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        x, y, w, h = match['bbox']
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cx = int(match['center_x'])
        cy = int(match['center_y'])
        cv2.drawMarker(img, (cx, cy), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)
        label = f"{idx}: {match.get('confidence', 0):.3f}"
        cv2.putText(img, label, (x, max(y - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    txt = f"matches={len(result.get('matches', []))} conf={result.get('confidence',0):.3f}"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(out_path, img)
