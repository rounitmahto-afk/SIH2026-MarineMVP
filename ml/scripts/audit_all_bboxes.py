from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json
import cv2
import numpy as np


ARCHIVE = Path(
    "data/raw/ghostvision/GhostVision_DatasetAndModels.zip"
)

ANNOTATION_ROOT = Path(
    "data/raw/ghostvision_annotations"
)

OUTPUT = Path(
    "data/manifests/ghostvision_bbox_audit.json"
)

# Boundary tolerance in SOURCE IMAGE PIXELS.
# <= 1 pixel is considered a boundary-rounding case.
BOUNDARY_TOLERANCE = 1.0


def detect_split(path: str) -> str:
    p = path.replace("\\", "/").lower()

    if "/train/" in p:
        return "train"

    if "/valid/" in p:
        return "validation"

    if "/test/" in p:
        return "test"

    return "other"


def load_records():
    metadata_files = {
        "train":
            ANNOTATION_ROOT / "train" / "metadata.jsonl",

        "validation":
            ANNOTATION_ROOT / "valid" / "metadata.jsonl",

        "test":
            ANNOTATION_ROOT / "test" / "metadata.jsonl",
    }

    records = []

    for split, path in metadata_files.items():

        if not path.exists():
            raise FileNotFoundError(path)

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                records.append({
                    "split": split,
                    "line": line_number,
                    "file_name":
                        record["file_name"],
                    "objects":
                        record.get(
                            "objects",
                            {}
                        )
                })

    return records


if not ARCHIVE.exists():
    raise FileNotFoundError(
        f"Archive missing: {ARCHIVE}"
    )


records = load_records()

print("")
print(
    f"Annotation records: {len(records)}"
)

# ------------------------------------------------------------
# Build exact archive lookup
# ------------------------------------------------------------

with ZipFile(
    ARCHIVE,
    "r"
) as archive:

    image_members = [
        name
        for name in archive.namelist()
        if (
            not name.endswith("/")
            and Path(name).suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        )
    ]

    image_lookup = {
        Path(name).name: name
        for name in image_members
    }

    counters = Counter()

    examples = []

    max_x_overflow = 0.0
    max_y_overflow = 0.0

    max_negative_x = 0.0
    max_negative_y = 0.0

    total_boxes = 0

    for index, record in enumerate(
        records,
        start=1
    ):

        basename = Path(
            record["file_name"]
        ).name

        archive_member = image_lookup.get(
            basename
        )

        if archive_member is None:
            raise RuntimeError(
                f"Image not found: {basename}"
            )

        image_bytes = archive.read(
            archive_member
        )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Image decode failed: {basename}"
            )

        height, width = image.shape[:2]

        objects = record["objects"]

        bboxes = objects.get(
            "bbox",
            []
        )

        categories = objects.get(
            "category",
            []
        )

        if len(bboxes) != len(categories):
            raise RuntimeError(
                f"bbox/category mismatch: "
                f"{basename}"
            )

        for box_index, (
            bbox,
            category
        ) in enumerate(
            zip(bboxes, categories)
        ):

            if len(bbox) != 4:
                raise RuntimeError(
                    f"Invalid bbox length: "
                    f"{basename}"
                )

            x, y, box_width, box_height = [
                float(v)
                for v in bbox
            ]

            right = x + box_width
            bottom = y + box_height

            x_overflow = max(
                0.0,
                right - width
            )

            y_overflow = max(
                0.0,
                bottom - height
            )

            negative_x = max(
                0.0,
                -x
            )

            negative_y = max(
                0.0,
                -y
            )

            max_x_overflow = max(
                max_x_overflow,
                x_overflow
            )

            max_y_overflow = max(
                max_y_overflow,
                y_overflow
            )

            max_negative_x = max(
                max_negative_x,
                negative_x
            )

            max_negative_y = max(
                max_negative_y,
                negative_y
            )

            total_boxes += 1

            outside = (
                x_overflow > 0
                or y_overflow > 0
                or negative_x > 0
                or negative_y > 0
            )

            if not outside:
                counters["valid"] += 1
                continue

            tiny_boundary = (
                x_overflow <= BOUNDARY_TOLERANCE
                and y_overflow <= BOUNDARY_TOLERANCE
                and negative_x <= BOUNDARY_TOLERANCE
                and negative_y <= BOUNDARY_TOLERANCE
            )

            if tiny_boundary:

                counters[
                    "boundary_rounding"
                ] += 1

                if len(examples) < 25:

                    examples.append({
                        "type":
                            "boundary_rounding",

                        "split":
                            record["split"],

                        "file_name":
                            record["file_name"],

                        "archive_member":
                            archive_member,

                        "box_index":
                            box_index,

                        "category":
                            category,

                        "image_width":
                            width,

                        "image_height":
                            height,

                        "bbox":
                            [
                                x,
                                y,
                                box_width,
                                box_height
                            ],

                        "x_overflow":
                            x_overflow,

                        "y_overflow":
                            y_overflow,

                        "negative_x":
                            negative_x,

                        "negative_y":
                            negative_y
                    })

            else:

                counters[
                    "substantial_mismatch"
                ] += 1

                if len(examples) < 25:

                    examples.append({
                        "type":
                            "substantial_mismatch",

                        "split":
                            record["split"],

                        "file_name":
                            record["file_name"],

                        "archive_member":
                            archive_member,

                        "box_index":
                            box_index,

                        "category":
                            category,

                        "image_width":
                            width,

                        "image_height":
                            height,

                        "bbox":
                            [
                                x,
                                y,
                                box_width,
                                box_height
                            ],

                        "x_overflow":
                            x_overflow,

                        "y_overflow":
                            y_overflow,

                        "negative_x":
                            negative_x,

                        "negative_y":
                            negative_y
                    })

        if index % 500 == 0:
            print(
                f"Audited "
                f"{index}/{len(records)} images..."
            )


audit = {

    "source": {
        "archive":
            str(ARCHIVE),

        "records":
            len(records)
    },

    "configuration": {
        "boundary_tolerance_pixels":
            BOUNDARY_TOLERANCE
    },

    "counts": {
        "images_audited":
            len(records),

        "total_bounding_boxes":
            total_boxes,

        "valid":
            counters["valid"],

        "boundary_rounding":
            counters[
                "boundary_rounding"
            ],

        "substantial_mismatch":
            counters[
                "substantial_mismatch"
            ]
    },

    "extremes": {
        "max_x_overflow":
            max_x_overflow,

        "max_y_overflow":
            max_y_overflow,

        "max_negative_x":
            max_negative_x,

        "max_negative_y":
            max_negative_y
    },

    "examples":
        examples
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT.write_text(
    json.dumps(
        audit,
        indent=2
    ),
    encoding="utf-8"
)

print("")
print("============================================================")
print("GLOBAL BBOX AUDIT RESULT")
print("============================================================")

print(
    f"Images audited       : "
    f"{audit['counts']['images_audited']}"
)

print(
    f"Bounding boxes       : "
    f"{audit['counts']['total_bounding_boxes']}"
)

print(
    f"Perfectly valid      : "
    f"{audit['counts']['valid']}"
)

print(
    f"Boundary cases       : "
    f"{audit['counts']['boundary_rounding']}"
)

print(
    f"Substantial mismatch : "
    f"{audit['counts']['substantial_mismatch']}"
)

print("")
print("Maximum deviations:")

print(
    f"  X overflow   : "
    f"{audit['extremes']['max_x_overflow']}"
)

print(
    f"  Y overflow   : "
    f"{audit['extremes']['max_y_overflow']}"
)

print(
    f"  Negative X   : "
    f"{audit['extremes']['max_negative_x']}"
)

print(
    f"  Negative Y   : "
    f"{audit['extremes']['max_negative_y']}"
)

print("")
print(
    f"Audit manifest: {OUTPUT}"
)

# ------------------------------------------------------------
# HARD SAFETY GATE
# ------------------------------------------------------------

if counters["substantial_mismatch"] > 0:

    print("")
    print(
        "TRAINING BLOCKED."
    )

    print(
        "Substantial annotation/image mismatches "
        "exist and must be investigated."
    )

    raise RuntimeError(
        "Substantial bounding-box mismatches detected."
    )

print("")
print(
    "GLOBAL BBOX AUDIT PASSED."
)

print(
    "Only valid boxes and/or tiny boundary-rounding "
    "cases are present."
)
