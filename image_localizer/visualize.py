"""Visualization utilities: draw polygon, rectangle, center, and overlay.
"""
from typing import Dict
import cv2
import numpy as np


def draw_result(target_path: str, result: Dict, out_path: str = 'detection_visual.png') -> None:
    img = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(target_path)
    pts = np.array(result['polygon'], dtype=np.int32).reshape(-1,2)
    # draw polygon
    cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=3)
    # bounding rect
    x,y,w,h = result['bbox']
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    # center
    cx = int(result['center_x'])
    cy = int(result['center_y'])
    cv2.drawMarker(img, (cx,cy), color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)
    # put text summary
    txt = f"rot={result.get('rotation_deg',0):.1f}deg scale=({result.get('scale_x',0):.3f},{result.get('scale_y',0):.3f}) conf={result.get('confidence',0):.3f}"
    cv2.putText(img, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imwrite(out_path, img)
