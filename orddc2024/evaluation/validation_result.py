from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ValidationResult:
    """
    Standard result returned by CustomValidator.

    per_class maps actual zero-based class IDs to class-wise Ultralytics
    metrics. Classes with no validation targets retain None for metrics.
    """

    overall: dict[str, float]
    confusion_matrix: np.ndarray
    metrics: Any
    per_class: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )
    average_detections_per_image: float = 0.0
