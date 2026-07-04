"""YOLO object detection and Tesseract OCR integration utilities."""
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


def _import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise ImportError('ultralytics is required for YOLO detection. Install with pip install ultralytics') from exc


def _import_pytesseract():
    try:
        import pytesseract
        _configure_tesseract_cmd(pytesseract)
        return pytesseract
    except ImportError as exc:
        raise ImportError('pytesseract is required for OCR. Install with pip install pytesseract') from exc


def _configure_tesseract_cmd(pytesseract) -> None:
    if shutil.which('tesseract'):
        return

    configured_cmd = os.environ.get('TESSERACT_CMD')
    candidate_paths = [
        configured_cmd,
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        str(Path.home() / 'AppData' / 'Local' / 'Programs' / 'Tesseract-OCR' / 'tesseract.exe'),
    ]
    for candidate in candidate_paths:
        if candidate and Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


def _normalize_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')
    return path


def load_yolo_model(model_name: str = 'yolov8n.pt'):  # pragma: no cover
    YOLO = _import_ultralytics()
    return YOLO(model_name)


def detect_objects(image_path: str, model_name: str = 'yolov8n.pt', conf: float = 0.25, iou: float = 0.45, device: str = 'cpu', imgsz: int = 640) -> Dict[str, Any]:
    """Detect objects in an image using Ultralytics YOLO.
    Returns: {image_path, shape, detections}
    """
    path = _normalize_image_path(image_path)
    model = load_yolo_model(model_name)
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f'Could not read image: {path}')

    results = model.predict(source=image, conf=conf, iou=iou, device=device, imgsz=imgsz, verbose=False)
    if len(results) == 0:
        return {'image_path': str(path), 'shape': image.shape, 'detections': []}

    result = results[0]
    names = result.names if hasattr(result, 'names') else {}
    detections: List[Dict[str, Any]] = []
    for box in getattr(result, 'boxes', []):
        xyxy = box.xyxy.cpu().numpy().reshape(-1).tolist()
        conf_score = float(box.conf.cpu().numpy().reshape(-1)[0]) if hasattr(box, 'conf') else 0.0
        cls_idx = int(box.cls.cpu().numpy().reshape(-1)[0]) if hasattr(box, 'cls') else -1
        detections.append({
            'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
            'confidence': conf_score,
            'class_id': cls_idx,
            'class_name': names.get(cls_idx, str(cls_idx)),
        })

    return {'image_path': str(path), 'shape': image.shape, 'detections': detections}


def extract_text(image_path: str, lang: str = 'eng', config: str = '--psm 3') -> Dict[str, Any]:
    """Extract text from an image with pytesseract.
    Returns full text plus individual word boxes.
    If Tesseract is unavailable, returns an error payload instead of raising.
    """
    path = _normalize_image_path(image_path)
    pytesseract = _import_pytesseract()
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f'Could not read image: {path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        data = pytesseract.image_to_data(gray, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    except (pytesseract.pytesseract.TesseractNotFoundError, FileNotFoundError) as exc:
        return {
            'image_path': str(path),
            'text': '',
            'words': [],
            'error': 'tesseract_not_found',
            'error_message': str(exc),
            'ocr_available': False,
        }

    words: List[Dict[str, Any]] = []
    text_chunks = []
    for i, word in enumerate(data.get('text', [])):
        if not word or word.strip() == '':
            continue
        text_chunks.append(word)
        words.append({
            'text': word,
            'confidence': float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
        })

    return {
        'image_path': str(path),
        'text': ' '.join(text_chunks).strip(),
        'words': words,
        'ocr_available': True,
    }


def detect_image(image_path: str, model_name: str = 'yolov8n.pt', conf: float = 0.25, iou: float = 0.45, device: str = 'cpu', imgsz: int = 640, ocr_lang: str = 'eng', ocr_config: str = '--psm 3', enable_ocr: bool = True) -> Dict[str, Any]:
    """Run YOLO object detection and Tesseract OCR on the same image."""
    detection = detect_objects(image_path, model_name=model_name, conf=conf, iou=iou, device=device, imgsz=imgsz)
    ocr = {'image_path': image_path, 'text': '', 'words': [], 'ocr_available': False}
    if enable_ocr:
        try:
            ocr = extract_text(image_path, lang=ocr_lang, config=ocr_config)
        except ImportError as exc:
            ocr = {
                'image_path': image_path,
                'text': '',
                'words': [],
                'error': 'pytesseract_not_installed',
                'error_message': str(exc),
                'ocr_available': False,
            }
    return {'detection': detection, 'ocr': ocr}


def draw_yolo_tesseract_output(image_path: str, output: Dict[str, Any], out_path: str) -> None:
    path = _normalize_image_path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f'Could not load image for drawing: {path}')

    dets = output.get('detection', {}).get('detections', []) if 'detection' in output else output.get('detections', [])
    for obj in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in obj['bbox']]
        label = f"{obj['class_name']} {obj['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    ocr = output.get('ocr', {}) if 'ocr' in output else output
    for word in ocr.get('words', []):
        x, y, w, h = word['left'], word['top'], word['width'], word['height']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imwrite(str(out_path), image)
