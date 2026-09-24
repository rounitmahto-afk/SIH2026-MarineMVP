from pathlib import Path
from zipfile import ZipFile
import json
import cv2


ARCHIVE = Path(
    "data/raw/ghostvision/GhostVision_DatasetAndModels.zip"
)

ANNOTATION_ROOT = Path(
    "data/raw/ghostvision_annotations"
)

TARGET = (
    "Rec16_wcp_ss_port_00053_png_jpg."
    "rf.e169e93d1e337ed1fe1af8ef3c9952e9.jpg"
)

metadata_files = [
    ANNOTATION_ROOT / "train" / "metadata.jsonl",
    ANNOTATION_ROOT / "valid" / "metadata.jsonl",
    ANNOTATION_ROOT / "test" / "metadata.jsonl",
]


def find_record():

    for metadata_path in metadata_files:

        if not metadata_path.exists():
            continue

        with metadata_path.open(
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

                file_name = record.get(
                    "file_name",
                    ""
                )

                if Path(file_name).name == TARGET:
                    return (
                        metadata_path,
                        line_number,
                        record
                    )

    return None, None, None


metadata_path, line_number, record = find_record()

if record is None:
    raise RuntimeError(
        f"Target annotation not found: {TARGET}"
    )


print("")
print("FOUND SOURCE ANNOTATION")
print("==============================")
print(
    f"Metadata : {metadata_path}"
)
print(
    f"Line     : {line_number}"
)
print(
    f"file_name: {record['file_name']}"
)

objects = record.get(
    "objects",
    {}
)

print("")
print("RAW OBJECT DATA")
print("==============================")
print(
    json.dumps(
        objects,
        indent=2
    )
)


# ------------------------------------------------------------
# Locate source image inside archive
# ------------------------------------------------------------

with ZipFile(
    ARCHIVE,
    "r"
) as archive:

    members = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
    ]

    matching = [
        name
        for name in members
        if Path(name).name == TARGET
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"Expected exactly one archive match, "
            f"found {len(matching)}"
        )

    archive_member = matching[0]

    print("")
    print("SOURCE IMAGE")
    print("==============================")
    print(
        f"Archive member: {archive_member}"
    )

    image_bytes = archive.read(
        archive_member
    )


# ------------------------------------------------------------
# Decode actual source image
# ------------------------------------------------------------

import numpy as np

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
        "OpenCV failed to decode the source image."
    )

height, width = image.shape[:2]

print(
    f"Actual image dimensions: "
    f"{width} x {height}"
)

print(
    f"Encoded source bytes   : "
    f"{len(image_bytes):,}"
)


# ------------------------------------------------------------
# Inspect every bbox mathematically
# ------------------------------------------------------------

bboxes = objects.get(
    "bbox",
    []
)

categories = objects.get(
    "category",
    []
)

print("")
print("BOUNDING BOX ANALYSIS")
print("==============================")

if len(bboxes) != len(categories):
    raise RuntimeError(
        "bbox/category length mismatch."
    )

for index, (bbox, category) in enumerate(
    zip(bboxes, categories)
):

    x, y, box_width, box_height = [
        float(value)
        for value in bbox
    ]

    right = x + box_width
    bottom = y + box_height

    print("")
    print(
        f"OBJECT #{index}"
    )

    print(
        f"  class       : {category}"
    )

    print(
        f"  bbox        : "
        f"x={x}, y={y}, "
        f"w={box_width}, h={box_height}"
    )

    print(
        f"  right edge  : {right}"
    )

    print(
        f"  bottom edge : {bottom}"
    )

    print(
        f"  image width : {width}"
    )

    print(
        f"  image height: {height}"
    )

    print(
        f"  exceeds X   : {right > width}"
    )

    print(
        f"  exceeds Y   : {bottom > height}"
    )

    print(
        f"  negative X  : {x < 0}"
    )

    print(
        f"  negative Y  : {y < 0}"
    )

    print(
        f"  width <= 0  : {box_width <= 0}"
    )

    print(
        f"  height <= 0 : {box_height <= 0}"
    )

    # --------------------------------------------------------
    # Show boundary distance
    # --------------------------------------------------------

    print(
        f"  X overflow  : {right - width:.6f}"
    )

    print(
        f"  Y overflow  : {bottom - height:.6f}"
    )


# ------------------------------------------------------------
# Check whether this is only a tiny floating-point boundary
# issue or a substantial mismatch.
# ------------------------------------------------------------

print("")
print("CLASSIFICATION")
print("==============================")

for bbox in bboxes:

    x, y, box_width, box_height = [
        float(value)
        for value in bbox
    ]

    right = x + box_width
    bottom = y + box_height

    x_overflow = right - width
    y_overflow = bottom - height

    if (
        x_overflow > 0
        or y_overflow > 0
    ):

        if (
            x_overflow <= 1.0
            and y_overflow <= 1.0
        ):
            print(
                "BOUNDARY CASE: "
                "annotation extends outside the image "
                "by <= 1 pixel."
            )
        else:
            print(
                "SUBSTANTIAL MISMATCH: "
                "annotation extends more than 1 pixel "
                "outside the decoded image."
            )

print("")
print("STEP 5A DIAGNOSTIC COMPLETE")
