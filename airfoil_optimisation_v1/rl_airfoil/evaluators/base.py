from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class AeroOutput:
    cl: float
    cd: float
    cm: float
    tc: float

    is_geometry_valid: bool = True
    solver_status: str = "ok"
    solver_error_message: str = ""
    runtime_ms: float = 0.0

    geometry_features: Dict[str, float | bool] = field(default_factory=dict)


class Evaluator:
    name: str = "base"

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        raise NotImplementedError