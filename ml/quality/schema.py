from dataclasses import dataclass


@dataclass(frozen=True)
class SSSQualityResult:
    image_width: int
    image_height: int

    mean_intensity: float
    std_intensity: float

    p01_intensity: float
    p99_intensity: float
    dynamic_range: float

    low_clip_fraction: float
    high_clip_fraction: float

    edge_density: float
    near_uniform_fraction: float

    quality_index: float
    status: str
    usable: bool
