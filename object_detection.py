"""Run YOLO object detection and Tesseract OCR on a single image."""
import argparse
import json
from pathlib import Path
from image_localizer.template_match import find_template_masked
from image_localizer.visualize import draw_result
from image_localizer.yolo_tesseract import detect_image, draw_yolo_tesseract_output
from image_localizer.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description='Run YOLO/OCR and find a template image inside a target image')
    parser.add_argument('--image', default='mockup.jpeg', help='Path to target/mockup image')
    parser.add_argument('--template', default='logo.jpeg', help='Path to template image to find inside the target')
    parser.add_argument('--template-min-scale', type=float, default=0.08, help='Smallest masked template scale to search')
    parser.add_argument('--template-max-scale', type=float, default=1.25, help='Largest masked template scale to search')
    parser.add_argument('--template-steps', type=int, default=80, help='Number of masked template scales to search')
    parser.add_argument('--template-min-width', type=int, default=32, help='Smallest matched template width to consider')
    parser.add_argument('--template-min-height', type=int, default=32, help='Smallest matched template height to consider')
    parser.add_argument('--template-confidence-threshold', type=float, default=0.6, help='Minimum masked template confidence to report match_found=true')
    parser.add_argument('--template-max-detections', type=int, default=20, help='Maximum number of template matches to return')
    parser.add_argument('--template-nms-iou', type=float, default=0.3, help='IoU threshold for merging duplicate template matches')
    parser.add_argument('--no-template', action='store_true', help='Skip template localization')
    parser.add_argument('--yolo-model', default='yolov8n.pt', help='YOLO model name or path')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='YOLO NMS IoU threshold')
    parser.add_argument('--device', default='cpu', help='Device for YOLO inference (cpu or cuda)')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO input image size')
    parser.add_argument('--ocr-lang', default='eng', help='Tesseract OCR language code')
    parser.add_argument('--ocr-config', default='--psm 3', help='Additional Tesseract config options')
    parser.add_argument('--no-ocr', action='store_true', help='Skip OCR and run only YOLO detection')
    parser.add_argument('--out', default='image_output/object_detection_output.json', help='JSON output path')
    parser.add_argument('--template-viz', default='image_output/template_match_vis.png', help='Template match visualization output path')
    args = parser.parse_args()
    ensure_dir(Path(args.out).parent)
    result = detect_image(
        image_path=args.image,
        model_name=args.yolo_model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        ocr_lang=args.ocr_lang,
        ocr_config=args.ocr_config,
        enable_ocr=not args.no_ocr,
    )

    if not args.no_template:
        template_result = find_template_masked(
            args.template,
            args.image,
            min_scale=args.template_min_scale,
            max_scale=args.template_max_scale,
            steps=args.template_steps,
            min_match_width=args.template_min_width,
            min_match_height=args.template_min_height,
            confidence_threshold=args.template_confidence_threshold,
            max_detections=args.template_max_detections,
            nms_iou_threshold=args.template_nms_iou,
        )
        result['template_match'] = template_result

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f'Wrote JSON output to {out_path}')

    template_match = result.get('template_match', {})
    if 'polygon' in template_match:
        draw_result(args.image, template_match, out_path=args.template_viz)
        print(f'Wrote template match visualization to {args.template_viz}')
        print(f"Template bbox: {template_match.get('bbox')}")
        print(f"Template confidence: {template_match.get('confidence')}")
        print(f"Template match_found: {template_match.get('match_found')}")
        print(f"Template match_count: {template_match.get('match_count')}")
    elif template_match:
        print(f"Template match rejected: confidence {template_match.get('confidence')} below threshold {template_match.get('confidence_threshold')}")
        print(f"Best candidate bbox: {template_match.get('best_candidate_bbox')}")


if __name__ == '__main__':
    main()
