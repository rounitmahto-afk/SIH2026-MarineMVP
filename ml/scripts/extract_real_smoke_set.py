from pathlib import Path
from zipfile import ZipFile
import json

archive = Path(r"data/raw/ghostvision/GhostVision_DatasetAndModels.zip")
output = Path(r"data/raw/ghostvision_smoke")
output.mkdir(parents=True, exist_ok=True)

selected = []

with ZipFile(archive, "r") as z:
    files = [
        name for name in z.namelist()
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Prefer images living inside a test directory.
    test_files = [
        name for name in files
        if "/test/" in name.lower()
        or "\\test\\" in name.lower()
    ]

    candidates = test_files if test_files else files

    # Pick only five actual source images.
    for name in candidates:
        if len(selected) >= 5:
            break

        info = z.getinfo(name)

        # Refuse suspiciously large members during smoke test extraction.
        if info.file_size > 20 * 1024 * 1024:
            continue

        destination_name = Path(name).name

        # Prevent duplicate basename collisions.
        destination = output / destination_name

        counter = 1
        while destination.exists():
            destination = output / f"{counter}_{destination_name}"
            counter += 1

        with z.open(info) as src, destination.open("wb") as dst:
            dst.write(src.read())

        selected.append({
            "archive_path": name,
            "extracted_path": str(destination),
            "size_bytes": destination.stat().st_size
        })

manifest = {
    "source": {
        "archive": "GhostVision_DatasetAndModels.zip",
        "zenodo_record": "10.5281/zenodo.20056679"
    },
    "purpose": "real-data-smoke-test",
    "images": selected
}

Path("data/manifests/ghostvision_smoke.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print(json.dumps(manifest, indent=2))

if not selected:
    raise RuntimeError("No real sonar images were extracted.")
