from __future__ import annotations

import json
from pathlib import Path

import cv2

from ml.evidence import (
    AcousticEvidenceExtractor,
    EvidenceFusionEngine,
)
from ml.inference import load_default_detector


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ghostvision_smoke"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "storage"
    / "runs"
    / "evidence_fusion_smoke.json"
)


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Real image directory missing: {IMAGE_DIR}"
        )

    images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise RuntimeError(
            "No real sonar images found."
        )

    detector = load_default_detector()
    extractor = AcousticEvidenceExtractor()
    fusion_engine = EvidenceFusionEngine()

    results = []

    for image_path in images:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to decode: {image_path}"
            )

        detection_result = detector.predict(image)

        fused_detections = []

        for detection in detection_result.detections:
            evidence = extractor.extract(
                image,
                detection.bbox_xyxy,
            )

            fusion = fusion_engine.fuse(
                detector_confidence=detection.confidence,
                evidence=evidence,
            )

            fused_detections.append(
                {
                    "detection": detection.to_dict(),
                    "evidence": evidence.to_dict(),
                    "fusion": fusion.to_dict(),
                }
            )

        results.append(
            {
                "source_image": image_path.name,
                "detection_count": len(
                    detection_result.detections
                ),
                "detections": fused_detections,
            }
        )

        print("")
        print(f"Image: {image_path.name}")
        print(
            f"Detections: "
            f"{len(detection_result.detections)}"
        )

        for index, item in enumerate(
            fused_detections,
            start=1,
        ):
            detection = item["detection"]
            fusion = item["fusion"]

            print(
                f"  Detection {index}: "
                f"class={detection['class_name']}, "
                f"confidence={detection['confidence']:.3f}, "
                f"evidence_index="
                f"{fusion['heuristic_evidence_index']:.3f}, "
                f"signals="
                f"{fusion['supporting_signal_count']}, "
                f"interpretation="
                f"{fusion['interpretation']}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "run_type": "real_detection_evidence_fusion_smoke",
                "image_count": len(results),
                "detected_images": sum(
                    1
                    for item in results
                    if item["detection_count"] > 0
                ),
                "total_detections": sum(
                    item["detection_count"]
                    for item in results
                ),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("============================================================")
    print("STEP 8D COMPLETE")
    print("============================================================")
    print(f"Manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
