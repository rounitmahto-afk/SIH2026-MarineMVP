from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import cv2
import numpy as np
import onnxruntime as ort


PROJECT = Path(".")
MODEL_PATH = PROJECT / "ml/models/gv-yolo12/weights.onnx"
CLASS_NAMES_PATH = PROJECT / "ml/models/gv-yolo12/class_names.txt"
IMAGE_DIR = PROJECT / "data/raw/ghostvision_smoke"
RUN_DIR = PROJECT / "storage/runs/baseline_inference"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def nms(boxes, scores, threshold):
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        current = int(order[0])
        keep.append(current)

        remaining = []

        for idx in order[1:]:
            if iou(boxes[current], boxes[int(idx)]) < threshold:
                remaining.append(idx)

        order = np.asarray(remaining, dtype=np.int64)

    return keep


def preprocess(image):
    resized = cv2.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor)


def decode_output(output):
    # Expected baseline shape:
    # [1, 5, 8400]
    #
    # Four box values + one class confidence
    # for the single GhostVision target class.

    predictions = np.asarray(output)

    if predictions.ndim != 3:
        raise RuntimeError(
            f"Unexpected output dimensions: {predictions.shape}"
        )

    predictions = predictions[0].T

    if predictions.shape[1] < 5:
        raise RuntimeError(
            f"Unexpected output feature count: {predictions.shape}"
        )

    boxes = []
    scores = []

    for row in predictions:
        cx, cy, width, height = row[:4]
        confidence = float(row[4])

        if confidence < CONF_THRESHOLD:
            continue

        x1 = float(cx - width / 2.0)
        y1 = float(cy - height / 2.0)
        x2 = float(cx + width / 2.0)
        y2 = float(cy + height / 2.0)

        x1 = max(0.0, min(float(INPUT_SIZE), x1))
        y1 = max(0.0, min(float(INPUT_SIZE), y1))
        x2 = max(0.0, min(float(INPUT_SIZE), x2))
        y2 = max(0.0, min(float(INPUT_SIZE), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)

    if not boxes:
        return []

    keep = nms(boxes, scores, IOU_THRESHOLD)

    detections = []

    for idx in keep:
        box = boxes[idx]

        detections.append({
            "class_name": "Crab-Pot",
            "confidence": round(float(scores[idx]), 6),
            "bbox_xyxy_640": [round(float(v), 3) for v in box]
        })

    return detections


def annotate(image, detections):
    result = image.copy()

    source_h, source_w = result.shape[:2]

    scale_x = source_w / INPUT_SIZE
    scale_y = source_h / INPUT_SIZE

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy_640"]

        x1 = int(round(x1 * scale_x))
        y1 = int(round(y1 * scale_y))
        x2 = int(round(x2 * scale_x))
        y2 = int(round(y2 * scale_y))

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.3f}'
        )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        cv2.rectangle(
            result,
            (x1, max(0, y1 - 24)),
            (x1 + max(120, len(label) * 8), y1),
            (0, 255, 255),
            -1
        )

        cv2.putText(
            result,
            label,
            (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return result


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names missing: {CLASS_NAMES_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Real image directory missing: {IMAGE_DIR}"
        )

    class_names = [
        line.strip()
        for line in CLASS_NAMES_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print("\nMODEL CLASSES:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")

    print("\nLoading ONNX model once...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    print(f"Input: {input_name}")
    print(f"Shape: {input_shape}")

    images = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not images:
        raise RuntimeError("No real sonar images found.")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Warmup
    first_image = cv2.imread(
        str(images[0]),
        cv2.IMREAD_COLOR
    )

    if first_image is None:
        raise RuntimeError(
            f"Could not decode warmup image: {images[0]}"
        )

    warmup_tensor = preprocess(first_image)

    session.run(
        None,
        {input_name: warmup_tensor}
    )

    print("\nMODEL WARMUP COMPLETE")

    results = []

    for image_path in images:
        print(f"\nProcessing: {image_path.name}")

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Failed to decode: {image_path}"
            )

        tensor = preprocess(image)

        started = __import__("time").perf_counter()

        outputs = session.run(
            None,
            {input_name: tensor}
        )

        elapsed_ms = (
            __import__("time").perf_counter()
            - started
        ) * 1000.0

        detections = decode_output(outputs[0])

        annotated = annotate(
            image,
            detections
        )

        output_image = RUN_DIR / (
            f"{image_path.stem}_annotated.jpg"
        )

        if not cv2.imwrite(
            str(output_image),
            annotated
        ):
            raise RuntimeError(
                f"Failed to write preview: {output_image}"
            )

        item = {
            "source_image": str(image_path),
            "source_sha256": sha256_file(image_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "detections": detections,
            "detection_count": len(detections),
            "inference_ms": round(elapsed_ms, 3),
            "model": "PINGEcosystem/gv-yolo12",
            "model_file": str(MODEL_PATH),
            "model_sha256": sha256_file(MODEL_PATH),
            "confidence_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        }

        results.append(item)

        print(f"Detections: {len(detections)}")
        print(f"Inference : {elapsed_ms:.2f} ms")

        for detection in detections:
            print(
                f'  {detection["class_name"]} '
                f'{detection["confidence"]:.3f} '
                f'{detection["bbox_xyxy_640"]}'
            )

    total_time = sum(
        item["inference_ms"]
        for item in results
    )

    manifest = {
        "run_type": "real_baseline_inference",
        "started_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": {
            "repository": "PINGEcosystem/gv-yolo12",
            "local_path": str(MODEL_PATH),
            "sha256": sha256_file(MODEL_PATH),
        },
        "input": {
            "directory": str(IMAGE_DIR),
            "image_count": len(images),
        },
        "configuration": {
            "input_size": INPUT_SIZE,
            "confidence_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        "results": results,
        "summary": {
            "images_processed": len(results),
            "images_with_detections": sum(
                1
                for item in results
                if item["detection_count"] > 0
            ),
            "total_detections": sum(
                item["detection_count"]
                for item in results
            ),
            "total_inference_ms": round(
                total_time,
                3
            ),
            "mean_inference_ms": round(
                total_time / len(results),
                3
            ),
        },
    }

    manifest_path = RUN_DIR / "baseline_inference.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n============================================================")
    print("REAL BASELINE INFERENCE COMPLETE")
    print("============================================================")

    print(
        json.dumps(
            manifest["summary"],
            indent=2
        )
    )

    print(f"\nManifest: {manifest_path}")
    print(f"Annotated images: {RUN_DIR}")


if __name__ == "__main__":
    main()
