import argparse, cv2, sys
from .detectors import PlateDetectorYOLO, FaceDetectorHaar, FaceDetectorYOLO
from .blur_ops import gaussian_inplace, pixelate_inplace
from .utils import choose_device, expand_box, nms_merge

def main():
    ap = argparse.ArgumentParser("Blur license plates and/or faces in video")
    ap.add_argument("--input", required=True, help="video path or camera index, e.g. 0")
    ap.add_argument("--output", default="out_blurred.mp4", help="output video path")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="inference device")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--scale", type=float, default=1.35, help="bbox expansion factor")
    ap.add_argument("--show", action="store_true", help="preview window")
    ap.add_argument("--blur-plates", action="store_true", default=True, help="blur license plates")
    ap.add_argument("--no-blur-plates", dest="blur_plates", action="store_false")
    ap.add_argument("--blur-faces", action="store_true", default=True, help="blur faces")
    ap.add_argument("--no-blur-faces", dest="blur_faces", action="store_false")
    ap.add_argument("--face-detector", choices=["haar","yolo"], default="haar", help="face backend")
    ap.add_argument("--face-yolo-weights", default="yolov8n-face.pt", help="YOLO face weights (file or hub id)")
    ap.add_argument("--method", choices=["gaussian","pixelate"], default="gaussian", help="blur method")
    ap.add_argument("--plate-weights", default="models/license_plate_detector.pt", help="Path to YOLO license plate model (.pt)")
    ap.add_argument("--blur-strength", type=float, default=4.0, help="Lower = stronger blur, higher = weaker blur (divisor for bbox size)")
    args = ap.parse_args()

    device = choose_device(args.device)
    print(f"[INFO] Using device: {device}")

    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        sys.exit(f"Could not open input: {args.input}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    plate_detector = PlateDetectorYOLO(args.plate_weights, device, args.conf, args.imgsz) \
                 if args.blur_plates else None

    if args.blur_faces:
        face_detector = FaceDetectorHaar() if args.face_detector=="haar" else FaceDetectorYOLO(
            args.face_yolo_weights, device, args.conf, args.imgsz
        )
    else:
        face_detector = None

    while True:
        ok, frame = cap.read()
        if not ok: break

        boxes = []
        if plate_detector:
            boxes += plate_detector(frame)
        if face_detector:
            boxes += face_detector(frame)

        boxes = nms_merge(boxes, iou_thresh=0.5)

        for (x1,y1,x2,y2) in boxes:
            x1,y1,x2,y2 = expand_box(x1,y1,x2,y2, args.scale, W, H)
            if args.method == "gaussian":
                gaussian_inplace(frame, x1,y1,x2,y2, args.blur_strength)
            else:
                pixelate_inplace(frame, x1,y1,x2,y2)

        writer.write(frame)
        if args.show:
            cv2.imshow("privacy-blur", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); writer.release()
    if args.show: cv2.destroyAllWindows()
    print(f"[OK] Saved: {args.output}")
