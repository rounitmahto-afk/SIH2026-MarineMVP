from __future__ import annotations

import json
from pathlib import Path

import cv2

from ml.evidence import AcousticEvidenceExtractor
from ml.inference import load_default_detector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ghostvision_smoke"
)


def main() -> None:
    images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise RuntimeError(
            "No real sonar smoke images found."
        )

    detector = load_default_detector()
    extractor = AcousticEvidenceExtractor()

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

        detections = detector.predict(
            image
        )

        evidence_items = []

        for detection in detections.detections:
            evidence = extractor.extract(
                image,
                detection.bbox_xyxy,
            )

            evidence_items.append(
                {
                    "detection": detection.to_dict(),
                    "evidence": evidence.to_dict(),
                }
            )

        results.append(
            {
                "source_image": image_path.name,
                "detection_count": len(
                    detections.detections
                ),
                "evidence": evidence_items,
            }
        )

        print("")
        print(f"Image: {image_path.name}")
        print(
            f"Detections: "
            f"{len(detections.detections)}"
        )

        for index, item in enumerate(
            evidence_items,
            start=1,
        ):
            evidence = item["evidence"]

            print(
                f"  Detection {index}: "
                f"contrast="
                f"{evidence['intensity_contrast']:.3f}, "
                f"edge="
                f"{evidence['edge_density']:.3f}, "
                f"compactness="
                f"{evidence['shape_compactness']:.3f}, "
                f"shadow_candidate="
                f"{evidence['shadow_candidate_support']:.3f}, "
                f"direction="
                f"{evidence['shadow_candidate_direction']}"
            )

    output_path = (
        PROJECT_ROOT
        / "storage"
        / "runs"
        / "acoustic_evidence_smoke.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            {
                "run_type": "acoustic_evidence_smoke",
                "image_count": len(results),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("============================================================")
    print("STEP 8A EVIDENCE SMOKE COMPLETE")
    print("============================================================")
    print(f"Manifest: {output_path}")


if __name__ == "__main__":
    main()
