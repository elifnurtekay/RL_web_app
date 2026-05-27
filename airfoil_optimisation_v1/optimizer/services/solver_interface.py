from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OptimizationInput:
    aoa: float
    reynolds: float
    upper_weights: list[float]
    lower_weights: list[float]
    leading_edge_weight: float
    trailing_edge_offset: float
    model: str = "PPO"


class AerodynamicOptimizer(ABC):
    @abstractmethod
    def optimize(self, optimization_input: OptimizationInput) -> dict:
        """Return metrics, geometry, and constraint verification."""
