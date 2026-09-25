from pathlib import Path

import cv2

from ml.quality import SSSQualityAssessor


def main() -> None:
    image_dir = Path(
        "data/derived/ghostvision_yolo/test/images"
    )

    images = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        ]
    )[:10]

    if not images:
        raise SystemExit(
            "No real GhostVision SSS images found."
        )

    assessor = SSSQualityAssessor()

    print(f"Real SSS images tested: {len(images)}")
    print("")

    for image_path in images:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise SystemExit(
                f"Could not decode real SSS image: {image_path}"
            )

        result = assessor.assess(image)

        print(image_path.name)
        print(
            f"  size={result.image_width}x{result.image_height}"
        )
        print(
            f"  mean={result.mean_intensity:.2f} "
            f"std={result.std_intensity:.2f}"
        )
        print(
            f"  dynamic_range={result.dynamic_range:.2f}"
        )
        print(
            f"  low_clip={result.low_clip_fraction:.4f} "
            f"high_clip={result.high_clip_fraction:.4f}"
        )
        print(
            f"  edge_density={result.edge_density:.4f}"
        )
        print(
            f"  near_uniform={result.near_uniform_fraction:.4f}"
        )
        print(
            f"  quality_index={result.quality_index:.4f} "
            f"status={result.status} "
            f"usable={result.usable}"
        )
        print("")

    print("Real SSS quality smoke test: PASS")


if __name__ == "__main__":
    main()
