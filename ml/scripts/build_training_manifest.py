from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "PINGEcosystem/sss-crab-pot-detection-ds"
REPO_TYPE = "dataset"

REVISION = None

ARCHIVE = Path(
    "data/raw/ghostvision/GhostVision_DatasetAndModels.zip"
)

ANNOTATION_ROOT = Path(
    "data/raw/ghostvision_annotations"
)

MANIFEST = Path(
    "data/manifests/ghostvision_training_data_manifest.json"
)


METADATA_FILES = {
    "train": "train/metadata.jsonl",
    "validation": "valid/metadata.jsonl",
    "test": "test/metadata.jsonl",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def detect_image_split(path: str):
    path = path.replace("\\", "/").lower()

    if "/train/" in path:
        return "train"

    if "/valid/" in path:
        return "validation"

    if "/test/" in path:
        return "test"

    return "other"


def validate_bbox(box):
    if not isinstance(box, (list, tuple)):
        return False

    if len(box) != 4:
        return False

    try:
        x, y, width, height = [
            float(v)
            for v in box
        ]
    except (TypeError, ValueError):
        return False

    if x < 0 or y < 0:
        return False

    if width <= 0 or height <= 0:
        return False

    return True


if not ARCHIVE.exists():
    raise FileNotFoundError(
        f"Missing archive: {ARCHIVE}"
    )


# ============================================================
# 1. Hugging Face revision
# ============================================================

api = HfApi()

repo_info = api.repo_info(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

REVISION = getattr(
    repo_info,
    "sha",
    None
)

print("")
print("DATASET REPOSITORY")
print("==============================")
print(f"Repository : {REPO_ID}")
print(f"Revision   : {REVISION}")


# ============================================================
# 2. Index images inside the existing verified archive
# ============================================================

print("")
print("INDEXING EXISTING REAL IMAGE ARCHIVE")
print("==============================")

with ZipFile(ARCHIVE, "r") as archive:

    members = [
        name
        for name in archive.namelist()
        if not name.endswith("/")
    ]

    image_members = [
        name
        for name in members
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


image_by_basename = defaultdict(list)

for member in image_members:
    image_by_basename[
        Path(member).name
    ].append(member)


duplicate_basenames = {
    name: paths
    for name, paths in image_by_basename.items()
    if len(paths) > 1
}


print(
    f"Images in archive: {len(image_members)}"
)

print(
    f"Duplicate basenames: {len(duplicate_basenames)}"
)


# ============================================================
# 3. Download all metadata files
# ============================================================

ANNOTATION_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

downloaded = {}

print("")
print("DOWNLOADING REAL ANNOTATION FILES")
print("==============================")

for split, repo_path in METADATA_FILES.items():

    print(
        f"Downloading {split}: {repo_path}"
    )

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_path,
        revision=REVISION,
        local_dir=str(ANNOTATION_ROOT)
    )

    local_path = Path(local_path)

    downloaded[split] = {
        "repository_path": repo_path,
        "local_path": str(local_path),
        "size_bytes": local_path.stat().st_size,
        "sha256": sha256_file(local_path)
    }


# ============================================================
# 4. Parse + validate annotations
# ============================================================

all_records = []

records_by_split = Counter()
annotations_by_split = Counter()
classes = Counter()

invalid_records = []
missing_images = []
duplicate_image_matches = []

seen_image_records = Counter()

print("")
print("VALIDATING REAL ANNOTATIONS")
print("==============================")

for split, info in downloaded.items():

    metadata_path = Path(
        info["local_path"]
    )

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

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": f"invalid JSON: {exc}"
                })
                continue

            if not isinstance(record, dict):
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": "record is not an object"
                })
                continue

            file_name = record.get(
                "file_name"
            )

            objects = record.get(
                "objects"
            )

            if not isinstance(
                file_name,
                str
            ):
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": "missing file_name"
                })
                continue

            if not isinstance(
                objects,
                dict
            ):
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": "objects is not an object",
                    "file_name": file_name
                })
                continue

            bboxes = objects.get(
                "bbox",
                []
            )

            categories = objects.get(
                "category",
                []
            )

            areas = objects.get(
                "area",
                []
            )

            if not isinstance(
                bboxes,
                list
            ):
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": "bbox is not a list",
                    "file_name": file_name
                })
                continue

            if not isinstance(
                categories,
                list
            ):
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": "category is not a list",
                    "file_name": file_name
                })
                continue

            if len(bboxes) != len(categories):
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": "bbox/category length mismatch",
                    "file_name": file_name,
                    "bbox_count": len(bboxes),
                    "category_count": len(categories)
                })
                continue

            if areas and len(areas) != len(
                categories
            ):
                invalid_records.append({
                    "split": split,
                    "line": line_number,
                    "reason": "area/category length mismatch",
                    "file_name": file_name
                })
                continue

            for bbox in bboxes:

                if not validate_bbox(bbox):

                    invalid_records.append({
                        "split": split,
                        "line": line_number,
                        "reason": "invalid bounding box",
                        "file_name": file_name,
                        "bbox": bbox
                    })

                    break

            else:

                # -----------------------------------------
                # Image matching
                # -----------------------------------------

                basename = Path(
                    file_name
                ).name

                matches = image_by_basename.get(
                    basename,
                    []
                )

                if len(matches) == 0:

                    missing_images.append({
                        "split": split,
                        "file_name": file_name
                    })

                elif len(matches) > 1:

                    duplicate_image_matches.append({
                        "split": split,
                        "file_name": file_name,
                        "matches": matches
                    })

                else:

                    matched_image = matches[0]

                    seen_image_records[
                        matched_image
                    ] += 1

                # -----------------------------------------
                # Class statistics
                # -----------------------------------------

                for category in categories:

                    if not isinstance(
                        category,
                        str
                    ):

                        invalid_records.append({
                            "split": split,
                            "line": line_number,
                            "reason":
                                "category is not string",
                            "file_name":
                                file_name
                        })

                    else:

                        classes[category] += 1

                records_by_split[
                    split
                ] += 1

                annotations_by_split[
                    split
                ] += len(categories)

                all_records.append({
                    "split": split,
                    "file_name": file_name,
                    "basename": basename,
                    "bbox_count": len(bboxes),
                    "categories": categories,
                    "bboxes": bboxes
                })


# ============================================================
# 5. Duplicate annotation records
# ============================================================

duplicate_annotation_records = {
    image: count
    for image, count
    in seen_image_records.items()
    if count > 1
}


# ============================================================
# 6. Determine complete coverage
# ============================================================

matched_image_set = set(
    seen_image_records.keys()
)

unannotated_archive_images = [
    member
    for member in image_members
    if member not in matched_image_set
]


# ============================================================
# 7. Build final manifest
# ============================================================

manifest = {

    "manifest_version": "1.0",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source": {

        "dataset_repository":
            REPO_ID,

        "dataset_revision":
            REVISION,

        "image_archive":
            str(ARCHIVE),

        "image_archive_sha256":
            sha256_file(ARCHIVE),

        "image_archive_size_bytes":
            ARCHIVE.stat().st_size
    },

    "metadata_files":
        downloaded,

    "image_inventory": {

        "total":
            len(image_members),

        "duplicate_basenames":
            len(duplicate_basenames),

        "duplicate_basename_examples":
            list(
                duplicate_basenames.items()
            )[:20],

        "matched_annotation_images":
            len(matched_image_set),

        "unannotated_archive_images":
            len(unannotated_archive_images),

        "unannotated_examples":
            unannotated_archive_images[:25]
    },

    "records": {

        "total":
            len(all_records),

        "by_split":
            dict(records_by_split)
    },

    "annotations": {

        "total":
            sum(classes.values()),

        "by_split":
            dict(annotations_by_split),

        "classes":
            [
                {
                    "name": name,
                    "count": count
                }
                for name, count
                in sorted(classes.items())
            ]
    },

    "validation": {

        "invalid_records":
            len(invalid_records),

        "invalid_examples":
            invalid_records[:25],

        "missing_images":
            len(missing_images),

        "missing_image_examples":
            missing_images[:25],

        "duplicate_image_matches":
            len(duplicate_image_matches),

        "duplicate_image_examples":
            duplicate_image_matches[:25],

        "duplicate_annotation_records":
            len(duplicate_annotation_records),

        "duplicate_annotation_examples":
            list(
                duplicate_annotation_records.items()
            )[:25]
    }
}


MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# 8. Print summary
# ============================================================

print("")
print("============================================================")
print("REAL TRAINING DATA SUMMARY")
print("============================================================")

print(
    f"Repository revision       : {REVISION}"
)

print(
    f"Archive images            : {len(image_members)}"
)

print(
    f"Annotation records        : {len(all_records)}"
)

print(
    f"Matched images            : {len(matched_image_set)}"
)

print(
    f"Unannotated images        : "
    f"{len(unannotated_archive_images)}"
)

print(
    f"Total annotations         : "
    f"{sum(classes.values())}"
)

print("")
print("Records by split:")

for split, count in sorted(
    records_by_split.items()
):
    print(
        f"  {split:<12}: {count}"
    )

print("")
print("Annotations by split:")

for split, count in sorted(
    annotations_by_split.items()
):
    print(
        f"  {split:<12}: {count}"
    )

print("")
print("Source classes:")

for name, count in sorted(
    classes.items()
):
    print(
        f"  {name:<20}: {count}"
    )

print("")
print("Validation:")

print(
    f"  Invalid records         : "
    f"{len(invalid_records)}"
)

print(
    f"  Missing images          : "
    f"{len(missing_images)}"
)

print(
    f"  Duplicate image matches : "
    f"{len(duplicate_image_matches)}"
)

print(
    f"  Duplicate annotations   : "
    f"{len(duplicate_annotation_records)}"
)

print("")
print(
    f"Manifest: {MANIFEST}"
)


# ============================================================
# Safety gates
# ============================================================

if invalid_records:
    raise RuntimeError(
        f"Dataset validation failed: "
        f"{len(invalid_records)} invalid records."
    )

if missing_images:
    raise RuntimeError(
        f"Dataset validation failed: "
        f"{len(missing_images)} annotation records "
        f"reference missing images."
    )

if duplicate_image_matches:
    raise RuntimeError(
        "Dataset validation failed: duplicate image matches."
    )

if duplicate_annotation_records:
    raise RuntimeError(
        "Dataset validation failed: duplicate annotation records."
    )

if sum(classes.values()) == 0:
    raise RuntimeError(
        "Dataset validation failed: no annotations."
    )

print("")
print("============================================================")
print("STEP 4B PASSED")
print("============================================================")
