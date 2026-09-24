import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("ml/models/gv-yolo12")

result = {
    "source": {
        "provider": "Hugging Face",
        "repository": "PINGEcosystem/gv-yolo12",
        "accessed_at_utc": datetime.now(timezone.utc).isoformat()
    },
    "files": {}
}

for path in sorted(root.iterdir()):
    if path.is_file():
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)

        result["files"][path.name] = {
            "sha256": sha256.hexdigest(),
            "size_bytes": path.stat().st_size
        }

Path("data/manifests/gv-yolo12.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print(json.dumps(result, indent=2))
