from __future__ import annotations

import json
import statistics
from pathlib import Path

import cv2

from ml.inference import load_default_detector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "ghostvision_smoke"


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Smoke image directory not found: {IMAGE_DIR}"
        )

    images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise RuntimeError(
            "No real smoke-test images found."
        )

    print("Loading detector once...")

    detector = load_default_detector()

    print(f"Model: {detector.model_path}")
    print(f"Classes: {detector.class_names}")
    print(f"Images: {len(images)}")

    first_image = cv2.imread(
        str(images[0]),
        cv2.IMREAD_COLOR,
    )

    if first_image is None:
        raise RuntimeError(
            f"Failed to decode warm-up image: {images[0]}"
        )

    warmup_ms = detector.warm_up(first_image)

    print("")
    print(f"Real-image warm-up: {warmup_ms:.2f} ms")

    results = []
    inference_times = []

    for image_path in images:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to decode image: {image_path}"
            )

        result = detector.predict(image)

        inference_times.append(result.inference_ms)

        results.append(
            {
                "source_image": image_path.name,
                **result.to_dict(),
            }
        )

        print("")
        print(f"Image: {image_path.name}")
        print(f"Inference: {result.inference_ms:.2f} ms")
        print(f"Detections: {len(result.detections)}")

        for detection in result.detections:
            print(
                f"  {detection.class_name} "
                f"{detection.confidence:.4f} "
                f"{detection.bbox_xyxy}"
            )

    p50_ms = statistics.median(inference_times)

    if len(inference_times) >= 2:
        sorted_times = sorted(inference_times)
        p95_index = min(
            len(sorted_times) - 1,
            int(0.95 * (len(sorted_times) - 1)),
        )
        p95_ms = sorted_times[p95_index]
    else:
        p95_ms = inference_times[0]

    output_path = (
        PROJECT_ROOT
        / "storage"
        / "runs"
        / "detector_adapter_smoke.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            {
                "run_type": "detector_adapter_smoke",
                "model_path": str(detector.model_path),
                "image_count": len(results),
                "warmup_ms": round(warmup_ms, 3),
                "runtime_summary": {
                    "p50_inference_ms": round(p50_ms, 3),
                    "p95_inference_ms": round(p95_ms, 3),
                    "mean_inference_ms": round(
                        statistics.mean(inference_times),
                        3,
                    ),
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("============================================================")
    print("DETECTOR ADAPTER WARMUP TEST COMPLETE")
    print("============================================================")
    print(
        f"P50 inference: {p50_ms:.2f} ms"
    )
    print(
        f"P95 inference: {p95_ms:.2f} ms"
    )
    print(
        f"Mean inference: "
        f"{statistics.mean(inference_times):.2f} ms"
    )
    print(f"Manifest: {output_path}")


if __name__ == "__main__":
    main()
