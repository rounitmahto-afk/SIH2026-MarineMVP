from pathlib import Path
from zipfile import ZipFile

archive = Path(r"data/raw/ghostvision/GhostVision_DatasetAndModels.zip")

with ZipFile(archive, "r") as z:
    files = [
        name for name in z.namelist()
        if not name.endswith("/")
    ]

    print(f"ARCHIVE_FILES={len(files)}")

    image_files = [
        name for name in files
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"IMAGE_FILES={len(image_files)}")

    print("\nFIRST_20_IMAGE_FILES:")
    for name in image_files[:20]:
        print(name)

    test_candidates = [
        name for name in image_files
        if "/test/" in name.lower()
        or "\\test\\" in name.lower()
    ]

    print(f"\nTEST_IMAGE_CANDIDATES={len(test_candidates)}")

    for name in test_candidates[:20]:
        print(name)
