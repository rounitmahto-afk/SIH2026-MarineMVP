from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import shutil

import cv2


ARCHIVE = Path(
    "data/raw/ghostvision/GhostVision_DatasetAndModels.zip"
)

SOURCE_MANIFEST = Path(
    "data/manifests/ghostvision_training_data_manifest.json"
)

AUDIT_MANIFEST = Path(
    "data/manifests/ghostvision_bbox_audit.json"
)

ANNOTATION_ROOT = Path(
    "data/raw/ghostvision_annotations"
)

OUTPUT_ROOT = Path(
    "data/derived/ghostvision_yolo"
)

BOUNDARY_TOLERANCE = 1.0

CLASS_MAP = {
    "Crab-Pot": 0
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def validate_source_bbox(
    bbox,
    image_width,
    image_height
):
    if not isinstance(
        bbox,
        (list, tuple)
    ):
        raise RuntimeError(
            "Bounding box is not a list."
        )

    if len(bbox) != 4:
        raise RuntimeError(
            "Bounding box must contain "
            "x,y,width,height."
        )

    x, y, width, height = [
        float(value)
        for value in bbox
    ]

    right = x + width
    bottom = y + height

    x_overflow = max(
        0.0,
        right - image_width
    )

    y_overflow = max(
        0.0,
        bottom - image_height
    )

    negative_x = max(
        0.0,
        -x
    )

    negative_y = max(
        0.0,
        -y
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid source bbox size: {bbox}"
        )

    substantial = (
        x_overflow > BOUNDARY_TOLERANCE
        or y_overflow > BOUNDARY_TOLERANCE
        or negative_x > BOUNDARY_TOLERANCE
        or negative_y > BOUNDARY_TOLERANCE
    )

    if substantial:
        raise RuntimeError(
            "Substantial bbox mismatch encountered "
            "during materialization."
        )

    return (
        x,
        y,
        width,
        height,
        x_overflow,
        y_overflow,
        negative_x,
        negative_y
    )


def normalize_bbox(
    bbox,
    image_width,
    image_height
):
    (
        x,
        y,
        width,
        height,
        x_overflow,
        y_overflow,
        negative_x,
        negative_y
    ) = validate_source_bbox(
        bbox,
        image_width,
        image_height
    )

    original_right = x + width
    original_bottom = y + height

    normalized_x1 = max(
        0.0,
        x
    )

    normalized_y1 = max(
        0.0,
        y
    )

    normalized_x2 = min(
        float(image_width),
        original_right
    )

    normalized_y2 = min(
        float(image_height),
        original_bottom
    )

    normalized_width = (
        normalized_x2
        - normalized_x1
    )

    normalized_height = (
        normalized_y2
        - normalized_y1
    )

    if normalized_width <= 0:
        raise RuntimeError(
            "Normalized bbox width became <= 0."
        )

    if normalized_height <= 0:
        raise RuntimeError(
            "Normalized bbox height became <= 0."
        )

    changed = (
        abs(
            normalized_x1 - x
        ) > 1e-12
        or
        abs(
            normalized_y1 - y
        ) > 1e-12
        or
        abs(
            normalized_x2 - original_right
        ) > 1e-12
        or
        abs(
            normalized_y2 - original_bottom
        ) > 1e-12
    )

    transform_type = (
        "boundary_clip"
        if changed
        else "none"
    )

    return {
        "original": {
            "x": x,
            "y": y,
            "width": width,
            "height": height
        },
        "normalized": {
            "x1": normalized_x1,
            "y1": normalized_y1,
            "x2": normalized_x2,
            "y2": normalized_y2,
            "width": normalized_width,
            "height": normalized_height
        },
        "overflow": {
            "x": x_overflow,
            "y": y_overflow,
            "negative_x": negative_x,
            "negative_y": negative_y
        },
        "transform": transform_type
    }


def to_yolo(
    normalized_bbox,
    image_width,
    image_height
):
    data = normalized_bbox["normalized"]

    x1 = data["x1"]
    y1 = data["y1"]
    x2 = data["x2"]
    y2 = data["y2"]

    width = x2 - x1
    height = y2 - y1

    center_x = (
        x1 + width / 2.0
    )

    center_y = (
        y1 + height / 2.0
    )

    values = [
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height
    ]

    if not all(
        0.0 <= value <= 1.0
        for value in values
    ):
        raise RuntimeError(
            f"YOLO values outside [0,1]: {values}"
        )

    return values


# ============================================================
# LOAD SOURCE MANIFESTS
# ============================================================

source_manifest = load_json(
    SOURCE_MANIFEST
)

audit_manifest = load_json(
    AUDIT_MANIFEST
)

expected_images = (
    source_manifest[
        "image_inventory"
    ][
        "total"
    ]
)

expected_records = (
    source_manifest[
        "records"
    ][
        "total"
    ]
)

audit_substantial = (
    audit_manifest[
        "counts"
    ][
        "substantial_mismatch"
    ]
)

audit_boundary = (
    audit_manifest[
        "counts"
    ][
        "boundary_rounding"
    ]
)

if audit_substantial != 0:
    raise RuntimeError(
        "Source bbox audit contains substantial mismatches. "
        "Materialization is blocked."
    )


print("")
print("SOURCE VALIDATION")
print("==============================")

print(
    f"Expected images       : {expected_images}"
)

print(
    f"Expected records      : {expected_records}"
)

print(
    f"Boundary cases        : {audit_boundary}"
)

print(
    f"Substantial mismatches: {audit_substantial}"
)


# ============================================================
# BUILD METADATA PATHS
# ============================================================

metadata_paths = {
    "train":
        ANNOTATION_ROOT
        / "train"
        / "metadata.jsonl",

    "validation":
        ANNOTATION_ROOT
        / "valid"
        / "metadata.jsonl",

    "test":
        ANNOTATION_ROOT
        / "test"
        / "metadata.jsonl"
}


for path in metadata_paths.values():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {path}"
        )


# ============================================================
# LOAD RECORDS
# ============================================================

records = []

for split, path in metadata_paths.items():

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
                "split":
                    split,

                "metadata_line":
                    line_number,

                "file_name":
                    record["file_name"],

                "objects":
                    record.get(
                        "objects",
                        {}
                    )
            })


if len(records) != expected_records:
    raise RuntimeError(
        "Source record count changed."
    )


# ============================================================
# INDEX IMAGE ARCHIVE
# ============================================================

print("")
print("INDEXING REAL IMAGE ARCHIVE")
print("==============================")

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

image_lookup = {}

for member in image_members:

    basename = Path(
        member
    ).name

    if basename in image_lookup:
        raise RuntimeError(
            f"Duplicate image basename: {basename}"
        )

    image_lookup[basename] = member


if len(image_members) != expected_images:
    raise RuntimeError(
        "Archive image count does not match "
        "Step 4B manifest."
    )


# ============================================================
# CLEAN OLD DERIVED OUTPUT
# ============================================================

if OUTPUT_ROOT.exists():

    print("")
    print(
        "Removing previous incomplete derived dataset:"
    )

    print(
        OUTPUT_ROOT.resolve()
    )

    shutil.rmtree(
        OUTPUT_ROOT
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in (
    "train",
    "validation",
    "test"
):

    (
        OUTPUT_ROOT
        / split
        / "images"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_ROOT
        / split
        / "labels"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# MATERIALIZE DATA
# ============================================================

records_written = 0
images_written = 0
objects_written = 0

split_images = Counter()
split_objects = Counter()
class_objects = Counter()

transform_counts = Counter()
transform_examples = []

generated_files = []

record_basenames = set()

print("")
print("MATERIALIZING REAL DATA")
print("==============================")

with ZipFile(
    ARCHIVE,
    "r"
) as archive:

    for index, record in enumerate(
        records,
        start=1
    ):

        split = record["split"]

        source_name = record[
            "file_name"
        ]

        basename = Path(
            source_name
        ).name

        if basename in record_basenames:
            raise RuntimeError(
                f"Duplicate annotation record: "
                f"{basename}"
            )

        record_basenames.add(
            basename
        )

        archive_member = image_lookup.get(
            basename
        )

        if archive_member is None:
            raise RuntimeError(
                f"Image missing from archive: "
                f"{basename}"
            )

        # ----------------------------------------------------
        # Extract real source image
        # ----------------------------------------------------

        destination_image = (
            OUTPUT_ROOT
            / split
            / "images"
            / basename
        )

        with archive.open(
            archive_member,
            "r"
        ) as source, destination_image.open(
            "wb"
        ) as destination:

            shutil.copyfileobj(
                source,
                destination,
                length=1024 * 1024
            )

        # ----------------------------------------------------
        # Verify image decode
        # ----------------------------------------------------

        image = cv2.imread(
            str(destination_image),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Failed to decode: "
                f"{basename}"
            )

        image_height, image_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Convert annotations
        # ----------------------------------------------------

        objects = record[
            "objects"
        ]

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

        label_lines = []

        record_transformations = []

        for object_index, (
            bbox,
            category
        ) in enumerate(
            zip(
                bboxes,
                categories
            )
        ):

            if category not in CLASS_MAP:
                raise RuntimeError(
                    f"Unexpected source class "
                    f"'{category}' in {basename}"
                )

            normalized = normalize_bbox(
                bbox,
                image_width,
                image_height
            )

            yolo = to_yolo(
                normalized,
                image_width,
                image_height
            )

            class_id = CLASS_MAP[
                category
            ]

            label_lines.append(
                (
                    f"{class_id} "
                    f"{yolo[0]:.8f} "
                    f"{yolo[1]:.8f} "
                    f"{yolo[2]:.8f} "
                    f"{yolo[3]:.8f}"
                )
            )

            class_objects[
                category
            ] += 1

            split_objects[
                split
            ] += 1

            objects_written += 1

            transform_type = (
                normalized[
                    "transform"
                ]
            )

            transform_counts[
                transform_type
            ] += 1

            if transform_type != "none":

                transformation = {
                    "split":
                        split,

                    "file_name":
                        source_name,

                    "object_index":
                        object_index,

                    "category":
                        category,

                    "original_bbox":
                        bbox,

                    "normalized_bbox":
                        normalized[
                            "normalized"
                        ],

                    "overflow":
                        normalized[
                            "overflow"
                        ],

                    "rule":
                        "clip bbox to image bounds "
                        "when boundary deviation <= 1 pixel"
                }

                record_transformations.append(
                    transformation
                )

                if len(transform_examples) < 100:

                    transform_examples.append(
                        transformation
                    )

        destination_label = (
            OUTPUT_ROOT
            / split
            / "labels"
            / (
                Path(basename).stem
                + ".txt"
            )
        )

        destination_label.write_text(
            "\n".join(label_lines)
            + (
                "\n"
                if label_lines
                else ""
            ),
            encoding="utf-8"
        )

        split_images[
            split
        ] += 1

        images_written += 1
        records_written += 1

        generated_files.append({
            "split":
                split,

            "source_archive_member":
                archive_member,

            "source_file_name":
                source_name,

            "image":
                str(destination_image),

            "label":
                str(destination_label),

            "width":
                int(image_width),

            "height":
                int(image_height),

            "object_count":
                len(label_lines),

            "transformed_bbox_count":
                len(
                    record_transformations
                ),

            "image_sha256":
                sha256_file(
                    destination_image
                ),

            "label_sha256":
                sha256_file(
                    destination_label
                )
        })

        if index % 250 == 0:

            print(
                f"Processed "
                f"{index}/{len(records)}"
            )


# ============================================================
# COVERAGE VALIDATION
# ============================================================

if records_written != expected_records:
    raise RuntimeError(
        f"Records written {records_written} "
        f"!= expected {expected_records}"
    )

if images_written != expected_images:
    raise RuntimeError(
        f"Images written {images_written} "
        f"!= expected {expected_images}"
    )


archive_basenames = {
    Path(member).name
    for member in image_members
}

if archive_basenames != record_basenames:

    raise RuntimeError(
        "Final image/annotation coverage mismatch."
    )


# ============================================================
# IMAGE/LABEL PAIR VALIDATION
# ============================================================

print("")
print("VALIDATING GENERATED DATASET")
print("==============================")

for split in (
    "train",
    "validation",
    "test"
):

    image_dir = (
        OUTPUT_ROOT
        / split
        / "images"
    )

    label_dir = (
        OUTPUT_ROOT
        / split
        / "labels"
    )

    images = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file()
    }

    labels = {
        path.stem
        for path in label_dir.iterdir()
        if path.is_file()
    }

    if images != labels:
        missing_labels = sorted(
            images - labels
        )

        missing_images = sorted(
            labels - images
        )

        raise RuntimeError(
            f"{split} image/label mismatch. "
            f"Missing labels={missing_labels[:5]} "
            f"Missing images={missing_images[:5]}"
        )

    print(
        f"{split:<12} "
        f"images={len(images):>5} "
        f"labels={len(labels):>5}"
    )


# ============================================================
# DATASET YAML
# ============================================================

dataset_yaml = """path: .
train: train/images
val: validation/images
test: test/images

names:
  0: Crab-Pot
"""


(
    OUTPUT_ROOT
    / "dataset.yaml"
).write_text(
    dataset_yaml,
    encoding="utf-8"
)


# ============================================================
# CLASS REGISTRY
# ============================================================

class_registry = {

    "version":
        "1.0",

    "classes": [
        {
            "class_id": 0,
            "source_name": "Crab-Pot",
            "training_name": "Crab-Pot",
            "product_role":
                "ghost_gear_proxy",
            "source_dataset":
                "PINGEcosystem/sss-crab-pot-detection-ds",
            "semantic_warning":
                "Do not present this class as ghost-net."
        }
    ]
}


(
    OUTPUT_ROOT
    / "class_registry.json"
).write_text(
    json.dumps(
        class_registry,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# TRANSFORMATION MANIFEST
# ============================================================

transformation_manifest = {

    "version":
        "1.0",

    "rule":
        "Raw source annotations remain immutable. "
        "Only bbox boundaries exceeding image bounds "
        "by <= 1 pixel are clipped to image bounds.",

    "boundary_tolerance_pixels":
        BOUNDARY_TOLERANCE,

    "audit_source":
        str(AUDIT_MANIFEST),

    "audit_boundary_case_count":
        audit_boundary,

    "transform_counts":
        dict(transform_counts),

    "examples":
        transform_examples
}


(
    OUTPUT_ROOT
    / "bbox_transformations.json"
).write_text(
    json.dumps(
        transformation_manifest,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# FINAL MATERIALIZATION MANIFEST
# ============================================================

final_manifest = {

    "version":
        "1.0",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source": {

        "dataset":
            "PINGEcosystem/sss-crab-pot-detection-ds",

        "dataset_revision":
            source_manifest[
                "source"
            ][
                "dataset_revision"
            ],

        "archive":
            str(ARCHIVE),

        "archive_sha256":
            source_manifest[
                "source"
            ][
                "image_archive_sha256"
            ]
    },

    "preprocessing": {

        "image_transform":
            "none",

        "annotation_transform":
            "boundary clipping <= 1 pixel",

        "class_mapping":
            {
                "Crab-Pot": 0
            }
    },

    "counts": {

        "records":
            records_written,

        "images":
            images_written,

        "objects":
            objects_written,

        "images_by_split":
            dict(split_images),

        "objects_by_split":
            dict(split_objects),

        "objects_by_class":
            dict(class_objects),

        "bbox_transformations":
            dict(transform_counts)
    },

    "artifacts": {

        "dataset_yaml":
            str(
                OUTPUT_ROOT
                / "dataset.yaml"
            ),

        "class_registry":
            str(
                OUTPUT_ROOT
                / "class_registry.json"
            ),

        "bbox_transformations":
            str(
                OUTPUT_ROOT
                / "bbox_transformations.json"
            )
    },

    "generated_file_count":
        len(generated_files)
}


(
    OUTPUT_ROOT
    / "materialization_manifest.json"
).write_text(
    json.dumps(
        final_manifest,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("")
print("============================================================")
print("REAL YOLO DATASET MATERIALIZATION PASSED")
print("============================================================")

print(
    f"Images              : {images_written}"
)

print(
    f"Objects             : {objects_written}"
)

print("")
print("Images by split:")

for split, count in sorted(
    split_images.items()
):

    print(
        f"  {split:<12}: {count}"
    )

print("")
print("Objects by split:")

for split, count in sorted(
    split_objects.items()
):

    print(
        f"  {split:<12}: {count}"
    )

print("")
print("Objects by class:")

for name, count in sorted(
    class_objects.items()
):

    print(
        f"  {name:<20}: {count}"
    )

print("")
print("Bounding-box transformations:")

for name, count in sorted(
    transform_counts.items()
):

    print(
        f"  {name:<20}: {count}"
    )

print("")
print(
    f"Dataset root: {OUTPUT_ROOT}"
)

print(
    f"Dataset YAML: "
    f"{OUTPUT_ROOT / 'dataset.yaml'}"
)

print(
    f"Transformation log: "
    f"{OUTPUT_ROOT / 'bbox_transformations.json'}"
)

print(
    f"Final manifest: "
    f"{OUTPUT_ROOT / 'materialization_manifest.json'}"
)
