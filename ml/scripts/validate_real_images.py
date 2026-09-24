from pathlib import Path
import cv2
import json

root = Path("data/raw/ghostvision_smoke")

images = []

for path in sorted(root.iterdir()):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Failed to decode real image: {path}")

    height, width = image.shape

    if width < 32 or height < 32:
        raise RuntimeError(f"Image is suspiciously small: {path}")

    images.append({
        "file": str(path),
        "width": width,
        "height": height,
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std())
    })

if not images:
    raise RuntimeError("No real images found.")

report = {
    "images": images,
    "count": len(images)
}

Path("data/manifests/ghostvision_image_validation.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print(json.dumps(report, indent=2))

print("\nREAL IMAGE DECODING OK")
