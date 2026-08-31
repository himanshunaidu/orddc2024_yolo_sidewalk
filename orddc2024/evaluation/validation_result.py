from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics.utils.metrics import DetMetrics


@dataclass(slots=True)
class ValidationResult:
    """Results returned by CustomValidator.evaluate()."""

    overall: dict[str, float]
    confusion_matrix: np.ndarray
    metrics: DetMetrics