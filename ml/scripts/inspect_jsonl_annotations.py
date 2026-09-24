from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
import hashlib
import json


ARCHIVE = Path(
    "data/raw/ghostvision/GhostVision_DatasetAndModels.zip"
)

OUTPUT = Path(
    "data/manifests/ghostvision_annotation_inventory.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def detect_split(path: str) -> str:
    p = path.lower().replace("\\", "/")

    if "/train/" in p:
        return "train"

    if "/valid/" in p or "/val/" in p:
        return "validation"

    if "/test/" in p:
        return "test"

    return "other"


if not ARCHIVE.exists():
    raise FileNotFoundError(
        f"Archive not found: {ARCHIVE}"
    )


with ZipFile(ARCHIVE, "r") as archive:

    members = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
    ]

    image_files = [
        name
        for name in members
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    json_files = [
        name
        for name in members
        if name.lower().endswith(".json")
    ]

    txt_files = [
        name
        for name in members
        if name.lower().endswith(".txt")
    ]

    print("")
    print("ARCHIVE CONTENT")
    print("==============================")
    print(f"Total members : {len(members)}")
    print(f"Images        : {len(image_files)}")
    print(f"JSON files    : {len(json_files)}")
    print(f"TXT files     : {len(txt_files)}")

    print("")
    print("JSON FILES")
    print("==============================")

    for name in json_files:
        info = archive.getinfo(name)
        print(
            f"{name} | {info.file_size:,} bytes"
        )

    # --------------------------------------------------------
    # Inspect JSON/JSONL files
    # --------------------------------------------------------

    annotation_candidates = []

    dataset_records = []

    class_counts = Counter()
    annotation_counts_by_split = Counter()
    image_records_by_split = Counter()

    matched_images = set()
    malformed_json_lines = []

    for json_name in json_files:

        raw = archive.read(json_name).decode(
            "utf-8",
            errors="replace"
        )

        stripped = raw.strip()

        # --------------------------------------------
        # Try regular JSON first
        # --------------------------------------------

        parsed_objects = []

        try:
            parsed = json.loads(stripped)

            if isinstance(parsed, list):
                parsed_objects = parsed

            elif isinstance(parsed, dict):
                parsed_objects = [parsed]

        except json.JSONDecodeError:
            # ----------------------------------------
            # Try JSONL
            # ----------------------------------------
            for line_number, line in enumerate(
                raw.splitlines(),
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                    parsed_objects.append(parsed)

                except json.JSONDecodeError:
                    # Some JSON files may contain
                    # non-dataset metadata. Do not fail
                    # immediately; record it.
                    malformed_json_lines.append(
                        {
                            "file": json_name,
                            "line": line_number,
                            "content": line[:300]
                        }
                    )

        # --------------------------------------------
        # Inspect records
        # --------------------------------------------

        file_has_annotations = False

        for record in parsed_objects:

            if not isinstance(record, dict):
                continue

            file_name = record.get("file_name")

            objects = record.get("objects")

            if file_name is None:
                continue

            image_records_by_split[
                detect_split(file_name)
            ] += 1

            object_categories = []
            object_boxes = []

            if isinstance(objects, dict):

                categories = objects.get(
                    "category",
                    []
                )

                boxes = objects.get(
                    "bbox",
                    []
                )

                if isinstance(
                    categories,
                    list
                ):
                    object_categories = categories

                if isinstance(
                    boxes,
                    list
                ):
                    object_boxes = boxes

            if object_categories:

                file_has_annotations = True

                for category in object_categories:
                    class_counts[
                        str(category)
                    ] += 1

                annotation_counts_by_split[
                    detect_split(file_name)
                ] += len(object_categories)

            annotation_candidates.append(
                {
                    "source_file": json_name,
                    "file_name": file_name,
                    "category_count": len(
                        object_categories
                    ),
                    "bbox_count": len(
                        object_boxes
                    )
                }
            )

    # --------------------------------------------------------
    # Build archive image lookup
    # --------------------------------------------------------

    image_lookup = {}

    for image in image_files:

        image_name = Path(image).name

        image_lookup[image_name] = image

    for record in annotation_candidates:

        file_name = Path(
            str(record["file_name"])
        ).name

        if file_name in image_lookup:
            matched_images.add(
                image_lookup[file_name]
            )

    images_without_annotation_record = [
        image
        for image in image_files
        if image not in matched_images
    ]

    # --------------------------------------------------------
    # Class information
    # --------------------------------------------------------

    class_registry = []

    for class_name, count in sorted(
        class_counts.items()
    ):
        class_registry.append(
            {
                "class_name": class_name,
                "annotation_count": count
            }
        )

    # --------------------------------------------------------
    # Sample annotation records
    # --------------------------------------------------------

    samples = [
        record
        for record in annotation_candidates
        if record["category_count"] > 0
    ][:20]

    inventory = {

        "source": {
            "archive": str(ARCHIVE),
            "size_bytes": ARCHIVE.stat().st_size,
            "sha256": sha256_file(ARCHIVE)
        },

        "archive_counts": {
            "members": len(members),
            "images": len(image_files),
            "json_files": len(json_files),
            "txt_files": len(txt_files)
        },

        "annotation_files": [
            name
            for name in json_files
            if any(
                token in name.lower()
                for token in (
                    "train",
                    "valid",
                    "val",
                    "test",
                    "metadata",
                    "annotation"
                )
            )
        ],

        "dataset_records": {
            "total_records": len(
                annotation_candidates
            ),
            "records_by_split": dict(
                image_records_by_split
            ),
            "matched_source_images": len(
                matched_images
            ),
            "images_without_annotation_record": len(
                images_without_annotation_record
            )
        },

        "annotations": {
            "total_annotations": sum(
                class_counts.values()
            ),
            "annotations_by_split": dict(
                annotation_counts_by_split
            ),
            "classes": class_registry
        },

        "quality": {
            "malformed_json_lines": len(
                malformed_json_lines
            ),
            "malformed_examples":
                malformed_json_lines[:20]
        },

        "sample_annotation_records": samples,

        "json_file_samples": [
            {
                "file": name,
                "size_bytes":
                    archive.getinfo(name).file_size
            }
            for name in json_files
        ]
    }


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT.write_text(
    json.dumps(
        inventory,
        indent=2
    ),
    encoding="utf-8"
)

print("")
print("============================================================")
print("REAL JSONL DATASET INVENTORY")
print("============================================================")

print(
    json.dumps(
        inventory,
        indent=2
    )
)

if len(image_files) == 0:
    raise RuntimeError(
        "No real images found."
    )

if len(json_files) == 0:
    raise RuntimeError(
        "No JSON annotation/metadata files found."
    )

if len(annotation_candidates) == 0:
    raise RuntimeError(
        "No image annotation records were detected."
    )

if sum(class_counts.values()) == 0:
    raise RuntimeError(
        "No object annotations were detected."
    )

print("")
print("============================================================")
print("JSONL ANNOTATION INVENTORY PASSED")
print("============================================================")

print(
    f"Images                  : {len(image_files)}"
)

print(
    f"Annotation records      : "
    f"{len(annotation_candidates)}"
)

print(
    f"Matched source images   : "
    f"{len(matched_images)}"
)

print(
    f"Total object annotations: "
    f"{sum(class_counts.values())}"
)

print(
    f"Classes                 : "
    f"{list(class_counts.keys())}"
)

print(
    f"Malformed JSON lines    : "
    f"{len(malformed_json_lines)}"
)

print(
    f"Inventory file          : "
    f"{OUTPUT}"
)
