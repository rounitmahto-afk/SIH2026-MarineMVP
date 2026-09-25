# Marine Detector Runtime

The production detector adapter is implemented in:

`ml/inference/detector.py`

The adapter uses the trained YOLO11n checkpoint selected from the validation workflow.

Default checkpoint:

`storage/runs/crab_pot_full/weights/best.pt`

Override with:

`MARINE_MODEL_PATH`

The adapter loads the model once and exposes structured detection results.

Current source class:

`Crab-Pot`

Product semantic mapping for later pipeline/database integration:

`Crab-Pot` -> `ghost_gear_proxy`

Do not relabel this dataset as "ghost net" detection. The current real GhostVision dataset contains the Crab-Pot class only.
