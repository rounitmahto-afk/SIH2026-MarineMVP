from pathlib import Path
from zipfile import ZipFile
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "PINGEcosystem/sss-crab-pot-detection-ds"
REPO_TYPE = "dataset"

IMAGE_ARCHIVE = Path(
    "data/raw/ghostvision/GhostVision_DatasetAndModels.zip"
)

ANNOTATION_DIR = Path(
    "data/raw/ghostvision_annotations"
)

MANIFEST_PATH = Path(
    "data/manifests/ghostvision_training_data_manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def archive_image_index(archive_path: Path):
    with ZipFile(archive_path, "r") as archive:

        image_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

    by_name = defaultdict(list)

    for member in image_members:
        by_name[Path(member).name].append(member)

    duplicates = {
        name: paths
        for name, paths in by_name.items()
        if len(paths) > 1
    }

    return image_members, by_name, duplicates


def detect_split(repo_path: str) -> str:
    normalized = repo_path.replace("\\", "/").lower()

    if normalized.startswith("train/"):
        return "train"

    if normalized.startswith("valid/"):
        return "validation"

    if normalized.startswith("val/"):
        return "validation"

    if normalized.startswith("test/"):
        return "test"

    return "other"


if not IMAGE_ARCHIVE.exists():
    raise FileNotFoundError(
        f"Missing image archive: {IMAGE_ARCHIVE}"
    )


api = HfApi()

print("")
print("DISCOVERING DATASET REPOSITORY")
print("==============================")
print(f"Repository: {REPO_ID}")

repo_info = api.repo_info(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

revision = getattr(
    repo_info,
    "sha",
    None
)

print(f"Repository revision: {revision}")

repo_files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

metadata_files = sorted([
    path
    for path in repo_files
    if path.lower().endswith("metadata.jsonl")
])

print("")
print("METADATA FILES FOUND")
print("==============================")

for path in metadata_files:
    print(path)

if not metadata_files:
    raise RuntimeError(
        "No metadata.jsonl files were found in the public dataset repository."
    )


print("")
print("INDEXING EXISTING REAL IMAGE ARCHIVE")
print("==============================")

image_members, image_by_name, duplicate_images = (
    archive_image_index(IMAGE_ARCHIVE)
)

print(f"Real images in archive: {len(image_members)}")

if duplicate_images:
    print(
        f"WARNING: duplicate image basenames found: "
        f"{len(duplicate_images)}"
    )

ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

all_records = []
split_records = Counter()
class_counts = Counter()
annotation_counts_by_split = Counter()

matched_images = set()
missing_images = []
duplicate_record_images = []
invalid_records = []

downloaded_files = []

print("")
print("DOWNLOADING ONLY ANNOTATION METADATA")
print("==============================")

for repo_path in metadata_files:

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_path,
        revision=revision if revision else None,
        local_dir=str(ANNOTATION_DIR)
    )

    local_path = Path(local_path)

    downloaded_files.append({
        "repository_path": repo_path,
        "local_path": str(local_path),
        "size_bytes": local_path.stat().st_size,
        "sha256": sha256_file(local_path),
        "split": detect_split(repo_path)
    })

    split = detect_split(repo_path)

    print(
        f"{split:12} "
        f"{repo_path} "
        f"-> {local_path}"
    )

    with local_path.open(
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
                    "file": repo_path,
                    "line": line_number,
                    "reason": f"invalid JSON: {exc}"
                })
                continue

            if not isinstance(record, dict):
                invalid_records.append({
                    "file": repo_path,
                    "line": line_number,
                    "reason": "record is not an object"
                })
                continue

            file_name = record.get(
                "file_name"
            )

            objects = record.get(
                "objects",
                {}
            )

            if not isinstance(
                file_name,
                str
            ):
                invalid_records.append({
                    "file": repo_path,
                    "line": line_number,
                    "reason": "missing file_name"
                })
                continue

            if not isinstance(
                objects,
                dict
            ):
                invalid_records.append({
                    "file": repo_path,
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
                    "file": repo_path,
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
                    "file": repo_path,
                    "line": line_number,
                    "reason": "category is not a list",
                    "file_name": file_name
                })
                continue

            if len(bboxes) != len(categories):
                invalid_records.append({
                    "file": repo_path,
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
                    "file": repo_path,
                    "line": line_number,
                    "reason": "area/category length mismatch",
                    "file_name": file_name
                })
                continue

            base_name = Path(
                file_name
            ).name

            matching_members = image_by_name.get(
                base_name,
                []
            )

            if len(matching_members) == 0:
                missing_images.append({
                    "split": split,
                    "file_name": file_name,
                    "metadata_file": repo_path
                })

            elif len(matching_members) > 1:
                duplicate_record_images.append({
                    "split": split,
                    "file_name": file_name,
                    "matches": matching_members
                })

            else:
                matched_images.add(
                    matching_members[0]
                )

            split_records[split] += 1

            for category in categories:

                if not isinstance(
                    category,
                    str
                ):
                    invalid_records.append({
                        "file": repo_path,
                        "line": line_number,
                        "reason": "category is not a string",
                        "file_name": file_name
                    })
                    continue

                class_counts[category] += 1
                annotation_counts_by_split[
                    split
                ] += 1

            all_records.append({
                "split": split,
                "file_name": file_name,
                "basename": base_name,
                "bbox_count": len(bboxes),
                "categories": categories
            })


# ------------------------------------------------------------
# Identify images that exist in the archive but are not
# represented in annotation metadata.
# ------------------------------------------------------------

annotation_basenames = {
    record["basename"]
    for record in all_records
}

unrepresented_archive_images = [
    member
    for member in image_members
    if Path(member).name not in annotation_basenames
]


# ------------------------------------------------------------
# Class registry
# ------------------------------------------------------------

class_registry = []

for class_name, count in sorted(
    class_counts.items()
):

    class_registry.append({
        "source_class": class_name,
        "annotation_count": count
    })


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

manifest = {

    "source": {
        "dataset_repository": REPO_ID,
        "repository_revision": revision,
        "accessed_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "image_archive": {
            "path": str(IMAGE_ARCHIVE),
            "sha256": sha256_file(
                IMAGE_ARCHIVE
            ),
            "size_bytes":
                IMAGE_ARCHIVE.stat().st_size
        }
    },

    "metadata": {
        "downloaded_files": downloaded_files
    },

    "images": {
        "archive_total":
            len(image_members),

        "archive_duplicate_basenames":
            len(duplicate_images),

        "matched_to_annotation_records":
            len(matched_images),

        "metadata_images_missing_from_archive":
            len(missing_images),

        "archive_images_without_metadata_record":
            len(unrepresented_archive_images)
    },

    "records": {
        "total":
            len(all_records),

        "by_split":
            dict(split_records)
    },

    "annotations": {
        "total":
            sum(class_counts.values()),

        "by_split":
            dict(annotation_counts_by_split),

        "classes":
            class_registry
    },

    "validation": {
        "invalid_records":
            len(invalid_records),

        "invalid_record_examples":
            invalid_records[:25],

        "missing_image_examples":
            missing_images[:25],

        "duplicate_image_examples":
            duplicate_record_images[:25],

        "unrepresented_archive_examples":
            unrepresented_archive_images[:25]
    }
}


MANIFEST_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)


print("")
print("============================================================")
print("REAL TRAINING-DATA INVENTORY")
print("============================================================")

print(
    json.dumps(
        manifest,
        indent=2
    )
)


# ------------------------------------------------------------
# Safety gates
# ------------------------------------------------------------

if len(all_records) == 0:
    raise RuntimeError(
        "No annotation records were loaded."
    )

if sum(class_counts.values()) == 0:
    raise RuntimeError(
        "No object annotations were found."
    )

if invalid_records:
    raise RuntimeError(
        f"Found {len(invalid_records)} invalid annotation records."
    )

if duplicate_record_images:
    raise RuntimeError(
        "Duplicate archive image matches detected."
    )


print("")
print("============================================================")
print("STEP 4 PASSED")
print("============================================================")

print(
    f"Archive images                 : {len(image_members)}"
)

print(
    f"Annotation records             : {len(all_records)}"
)

print(
    f"Matched images                 : {len(matched_images)}"
)

print(
    f"Archive images without metadata: "
    f"{len(unrepresented_archive_images)}"
)

print(
    f"Total annotations              : "
    f"{sum(class_counts.values())}"
)

print(
    f"Classes                        : "
    f"{list(class_counts.keys())}"
)

print(
    f"Invalid records                : "
    f"{len(invalid_records)}"
)

print(
    f"Manifest                       : "
    f"{MANIFEST_PATH}"
)
