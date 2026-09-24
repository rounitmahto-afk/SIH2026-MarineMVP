from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
import hashlib
import json


ARCHIVE = Path(
    "data/raw/ghostvision/GhostVision_DatasetAndModels.zip"
)

OUTPUT = Path(
    "data/manifests/ghostvision_dataset_inventory.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


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

    txt_files = [
        name
        for name in members
        if name.lower().endswith(".txt")
    ]

    yaml_files = [
        name
        for name in members
        if name.lower().endswith(
            (".yaml", ".yml")
        )
    ]

    json_files = [
        name
        for name in members
        if name.lower().endswith(".json")
    ]

    split_images = defaultdict(list)
    split_labels = defaultdict(list)

    def detect_split(path: str) -> str:
        p = path.lower()

        if "/train/" in p:
            return "train"

        if "/valid/" in p or "/val/" in p:
            return "validation"

        if "/test/" in p:
            return "test"

        return "other"

    for name in image_files:
        split_images[detect_split(name)].append(name)

    for name in txt_files:
        split_labels[detect_split(name)].append(name)

    # --------------------------------------------------------
    # Image / label matching
    # --------------------------------------------------------

    image_stems = {}

    for image in image_files:
        stem = Path(image).stem
        image_stems[stem] = image

    label_stems = {}

    for label in txt_files:
        stem = Path(label).stem
        label_stems[stem] = label

    matched_pairs = []
    missing_images = []
    orphan_labels = []

    for stem, label in label_stems.items():

        image = image_stems.get(stem)

        if image is None:
            missing_images.append(label)
        else:
            matched_pairs.append(
                {
                    "stem": stem,
                    "image": image,
                    "label": label
                }
            )

    for stem, image in image_stems.items():

        if stem not in label_stems:
            orphan_labels.append(image)

    # --------------------------------------------------------
    # YOLO label inspection
    # --------------------------------------------------------

    class_counts = Counter()
    annotation_counts_by_split = Counter()
    malformed = []

    for label_name in txt_files:

        split = detect_split(label_name)

        raw = archive.read(label_name).decode(
            "utf-8",
            errors="replace"
        )

        rows = [
            line.strip()
            for line in raw.splitlines()
            if line.strip()
        ]

        annotation_counts_by_split[split] += len(rows)

        for line_number, row in enumerate(
            rows,
            start=1
        ):

            parts = row.split()

            if len(parts) != 5:
                malformed.append(
                    {
                        "file": label_name,
                        "line": line_number,
                        "content": row,
                        "reason": "expected 5 YOLO fields"
                    }
                )
                continue

            try:
                class_id = int(parts[0])

                values = [
                    float(x)
                    for x in parts[1:]
                ]

            except ValueError:
                malformed.append(
                    {
                        "file": label_name,
                        "line": line_number,
                        "content": row,
                        "reason": "non-numeric value"
                    }
                )
                continue

            if not all(
                0.0 <= value <= 1.0
                for value in values
            ):
                malformed.append(
                    {
                        "file": label_name,
                        "line": line_number,
                        "content": row,
                        "reason": "normalized value outside 0..1"
                    }
                )
                continue

            class_counts[class_id] += 1

    # --------------------------------------------------------
    # Metadata candidates
    # --------------------------------------------------------

    metadata_files = []

    for name in yaml_files + json_files:

        lower = name.lower()

        if any(
            keyword in lower
            for keyword in (
                "data",
                "dataset",
                "class",
                "names",
                "config",
                "readme"
            )
        ):
            metadata_files.append(name)

    # --------------------------------------------------------
    # Build inventory
    # --------------------------------------------------------

    inventory = {
        "source": {
            "archive": str(ARCHIVE),
            "size_bytes": ARCHIVE.stat().st_size,
            "sha256": sha256_file(ARCHIVE)
        },

        "members": {
            "total": len(members),
            "images": len(image_files),
            "txt_files": len(txt_files),
            "yaml_files": len(yaml_files),
            "json_files": len(json_files)
        },

        "images_by_split": {
            split: len(files)
            for split, files in sorted(
                split_images.items()
            )
        },

        "labels_by_split": {
            split: len(files)
            for split, files in sorted(
                split_labels.items()
            )
        },

        "annotations_by_split": {
            split: count
            for split, count in sorted(
                annotation_counts_by_split.items()
            )
        },

        "class_id_annotation_counts": {
            str(class_id): count
            for class_id, count in sorted(
                class_counts.items()
            )
        },

        "matching": {
            "matched_pairs": len(matched_pairs),
            "missing_images_for_labels": len(
                missing_images
            ),
            "images_without_labels": len(
                orphan_labels
            )
        },

        "quality": {
            "malformed_annotation_rows": len(
                malformed
            ),
            "malformed_examples": malformed[:25]
        },

        "metadata_candidates": metadata_files[:100],

        "sample_images": {
            split: files[:10]
            for split, files in sorted(
                split_images.items()
            )
        },

        "sample_labels": {
            split: files[:10]
            for split, files in sorted(
                split_labels.items()
            )
        }
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
print("REAL DATASET INVENTORY")
print("============================================================")

print(
    json.dumps(
        inventory,
        indent=2
    )
)

# ------------------------------------------------------------
# Hard safety checks
# ------------------------------------------------------------

if len(image_files) == 0:
    raise RuntimeError(
        "No real sonar images were found."
    )

if len(txt_files) == 0:
    raise RuntimeError(
        "No annotation TXT files were found."
    )

if len(matched_pairs) == 0:
    raise RuntimeError(
        "No image/label pairs were found."
    )

if len(malformed) > 0:
    raise RuntimeError(
        f"Found {len(malformed)} malformed annotation rows. "
        "Training will NOT start until this is investigated."
    )

print("")
print("============================================================")
print("DATASET INVENTORY PASSED")
print("============================================================")
print(f"Images              : {len(image_files)}")
print(f"TXT annotations     : {len(txt_files)}")
print(f"Matched pairs       : {len(matched_pairs)}")
print(f"Missing images      : {len(missing_images)}")
print(f"Images without label: {len(orphan_labels)}")
print(f"Malformed rows      : {len(malformed)}")
print(f"Inventory file      : {OUTPUT}")
