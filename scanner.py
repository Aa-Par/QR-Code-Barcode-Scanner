import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


def locate_barcode_regions_classical(gray: np.ndarray) -> list:

    grad_x = cv2.Scharr(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    grad_y = cv2.Scharr(gray, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(cv2.convertScaleAbs(grad_x), cv2.convertScaleAbs(grad_y))
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = gray.shape[0] * gray.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.002 * img_area:  # discard tiny noise regions
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes


def detect_and_decode_qr(image: np.ndarray) -> list:
    detector = cv2.QRCodeDetector()
    results = []
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    except cv2.error:
        ok, decoded_info, points = False, [], None

    if ok and points is not None:
        for info, pts in zip(decoded_info, points):
            if info:  
                results.append({"type": "QRCODE", "data": info, "points": pts})

    if not results:
        data, pts, _ = detector.detectAndDecode(image)
        if data and pts is not None:
            results.append({"type": "QRCODE", "data": data, "points": pts})

    return results


def detect_and_decode_barcode(image: np.ndarray) -> list:
    results = []
    if not hasattr(cv2, "barcode"):
        return results  

    detector = cv2.barcode.BarcodeDetector()
    smoothed = cv2.GaussianBlur(image, (7, 7), 0)
    ok, decoded_info, decoded_type, points = detector.detectAndDecodeWithType(smoothed)
    if not ok:
        ok, decoded_info, decoded_type, points = detector.detectAndDecodeWithType(image)

    if ok and points is not None:
        for info, btype, pts in zip(decoded_info, decoded_type, points):
            if info:
                type_name = btype if btype else "BARCODE"
                results.append({"type": type_name, "data": info, "points": pts})

    return results


def annotate_image(image: np.ndarray, qr_results: list, barcode_results: list,
                    classical_boxes: list) -> np.ndarray:
    annotated = image.copy()

    for (x, y, w, h) in classical_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (180, 180, 180), 1)

    for r in qr_results:
        pts = r["points"].reshape(-1, 2).astype(int)
        cv2.polylines(annotated, [pts], True, (0, 200, 0), 3)
        x, y = pts[0]
        cv2.putText(annotated, f'QR: {r["data"][:30]}', (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    for r in barcode_results:
        pts = r["points"].reshape(-1, 2).astype(int)
        cv2.polylines(annotated, [pts], True, (255, 100, 0), 3)
        x, y = pts[0]
        cv2.putText(annotated, f'{r["type"]}: {r["data"][:30]}', (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    return annotated


def run_scan(image_path: str, output_dir: str = None) -> None:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image (unsupported format or corrupt file): {image_path}")
        sys.exit(1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"\n[INFO] Loaded image: {image_path}  (shape: {image.shape[1]}x{image.shape[0]})")
    print("[INFO] Running classical gradient-based barcode localization ...")
    classical_boxes = locate_barcode_regions_classical(gray)
    print(f"[INFO]   -> {len(classical_boxes)} candidate region(s) found.")

    print("[INFO] Running QR code detection & decoding ...")
    qr_results = detect_and_decode_qr(image)

    print("[INFO] Running barcode detection & decoding ...")
    if not hasattr(cv2, "barcode"):
        print("[WARN]   cv2.barcode module not found. Install 'opencv-contrib-python' "
              "to enable 1D barcode decoding. Skipping barcode decode step.")
        barcode_results = []
    else:
        barcode_results = detect_and_decode_barcode(image)

    print("\n" + "=" * 60)
    print("SCAN RESULTS")
    print("=" * 60)

    if not qr_results and not barcode_results:
        print("No QR codes or barcodes could be decoded in this image.")
        print("Tips: ensure the code is in focus, well-lit, and not too small.")
    else:
        for i, r in enumerate(qr_results, 1):
            print(f"[QR #{i}]  Type: QRCODE  |  Data: {r['data']}")
        for i, r in enumerate(barcode_results, 1):
            print(f"[Barcode #{i}]  Type: {r['type']}  |  Data: {r['data']}")

    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    annotated = annotate_image(image, qr_results, barcode_results, classical_boxes)
    base = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{base}_annotated_{timestamp}.png")
    cv2.imwrite(out_path, annotated)
    print(f"\n[INFO] Annotated result image saved to: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan a single image for QR codes and barcodes (CLI only, no camera)."
    )
    parser.add_argument(
        "-i", "--image", type=str, default=None,
        help="Path to the input image. If omitted, you will be prompted interactively."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the annotated output image "
             "(default: an 'outputs' folder next to scanner.py)."
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        image_path = input("Enter path to the QR/Barcode image: ").strip().strip('"').strip("'")

    run_scan(image_path, args.output_dir)


if __name__ == "__main__":
    main()
