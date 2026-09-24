from pathlib import Path
import json
import onnxruntime as ort

model_path = Path("ml/models/gv-yolo12/weights.onnx")

if not model_path.exists():
    raise FileNotFoundError(f"Missing model: {model_path}")

providers = ort.get_available_providers()

print("ONNX_RUNTIME_PROVIDERS:")
for provider in providers:
    print(" -", provider)

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"]
)

print("\nMODEL_INPUTS:")

for item in session.get_inputs():
    print({
        "name": item.name,
        "shape": item.shape,
        "type": item.type
    })

print("\nMODEL_OUTPUTS:")

for item in session.get_outputs():
    print({
        "name": item.name,
        "shape": item.shape,
        "type": item.type
    })

metadata = session.get_modelmeta()

print("\nMODEL_METADATA:")
print({
    "producer": metadata.producer_name,
    "graph_name": metadata.graph_name,
    "domain": metadata.domain,
    "description": metadata.description[:500] if metadata.description else "",
    "version": metadata.version
})

print("\nMODEL INSPECTION OK")
