from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "PINGEcosystem/gv-yolo12"
target = Path("ml/models/gv-yolo12")
target.mkdir(parents=True, exist_ok=True)

files = [
    "weights.onnx",
    "class_names.txt",
    "model_type.json",
    "environment.json",
]

for filename in files:
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(target),
    )
    print(f"DOWNLOADED: {path}")
