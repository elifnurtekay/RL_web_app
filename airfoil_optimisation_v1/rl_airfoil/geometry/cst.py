from __future__ import annotations

from dataclasses import dataclass, asdict
from math import comb
from typing import Dict

import numpy as np

def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """
    NumPy sürümleri arasında uyumlu trapezoidal integral hesabı.

    Öncelik:
    1. np.trapezoid  -> yeni NumPy sürümleri
    2. np.trapz      -> eski NumPy sürümleri
    3. manuel hesap  -> ikisi de yoksa güvenli fallback
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))

    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"y and x must have the same length. Got y={y.shape}, x={x.shape}"
        )

    if y.shape[0] < 2:
        return 0.0

    dx = np.diff(x)
    avg_y = 0.5 * (y[:-1] + y[1:])
    return float(np.sum(dx * avg_y))

@dataclass
class GeometryFeatures:
    max_thickness: float
    x_max_thickness: float
    max_camber: float
    x_max_camber: float
    leading_edge_radius_proxy: float
    trailing_edge_thickness: float
    upper_surface_curvature_mean: float
    lower_surface_curvature_mean: float
    surface_smoothness: float
    min_local_thickness: float
    local_thickness_violation: float
    is_local_thickness_feasible: bool
    area_proxy: float
    is_valid: bool

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "max_thickness": float(self.max_thickness),
            "x_max_thickness": float(self.x_max_thickness),
            "max_camber": float(self.max_camber),
            "x_max_camber": float(self.x_max_camber),
            "leading_edge_radius_proxy": float(self.leading_edge_radius_proxy),
            "trailing_edge_thickness": float(self.trailing_edge_thickness),
            "upper_surface_curvature_mean": float(self.upper_surface_curvature_mean),
            "lower_surface_curvature_mean": float(self.lower_surface_curvature_mean),
            "surface_smoothness": float(self.surface_smoothness),
            "min_local_thickness": float(self.min_local_thickness),
            "local_thickness_violation": float(self.local_thickness_violation),
            "is_local_thickness_feasible": bool(self.is_local_thickness_feasible),
            "area_proxy": float(self.area_proxy),
            "is_valid": bool(self.is_valid),
        }


def _bernstein_basis(order: int, i: int, x: np.ndarray) -> np.ndarray:
    return comb(order, i) * (x ** i) * ((1.0 - x) ** (order - i))


def cst_surface(
    weights: np.ndarray,
    x: np.ndarray,
    n1: float = 0.5,
    n2: float = 1.0,
    trailing_edge_thickness: float = 0.0,
) -> np.ndarray:
    """
    CST surface equation:
        y(x) = C(x) * S(x) + x * t_TE

    Burada weights dizisi 4 katsayı içerir.
    Upper surface için katsayılar genelde pozitif,
    lower surface için katsayılar genelde negatiftir.
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    order = len(weights) - 1

    class_function = (x ** n1) * ((1.0 - x) ** n2)

    shape_function = np.zeros_like(x, dtype=np.float64)
    for i, weight in enumerate(weights):
        shape_function += weight * _bernstein_basis(order, i, x)

    return class_function * shape_function + x * trailing_edge_thickness


def compute_cst_geometry_features(
    cst: np.ndarray,
    n_points: int = 201,
    min_local_thickness_required: float = 1e-4,
    max_abs_surface_y: float = 0.75,
) -> GeometryFeatures:
    """
    CST8 vektöründen fiziksel geometri özelliklerini çıkarır.

    cst sırası:
        [u1, u2, u3, u4, l1, l2, l3, l4]

    Not:
    - t/c = max_thickness olarak yorumlanır.
    - min_local_thickness hesabında leading/trailing edge uçları dışlanır.
    """
    cst = np.asarray(cst, dtype=np.float64).reshape(-1)
    if cst.shape[0] != 8:
        raise ValueError(f"CST vector must have 8 coefficients, got {cst.shape[0]}.")

    upper_w = cst[:4]
    lower_w = cst[4:]

    x = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    y_upper = cst_surface(upper_w, x)
    y_lower = cst_surface(lower_w, x)

    thickness = y_upper - y_lower
    camber = 0.5 * (y_upper + y_lower)

    max_thickness_idx = int(np.argmax(thickness))
    max_camber_idx = int(np.argmax(np.abs(camber)))

    max_thickness = float(thickness[max_thickness_idx])
    x_max_thickness = float(x[max_thickness_idx])

    max_camber = float(camber[max_camber_idx])
    x_max_camber = float(x[max_camber_idx])

    # Leading-edge radius yerine doğrudan radius hesaplamak hassas olabilir.
    # XAI için proxy olarak x≈0.01 civarındaki kalınlık bilgisi kullanılır.
    leading_edge_radius_proxy = float(max(thickness[1], 0.0))

    trailing_edge_thickness = float(max(thickness[-1], 0.0))

    dx = float(x[1] - x[0])
    upper_second = np.gradient(np.gradient(y_upper, dx), dx)
    lower_second = np.gradient(np.gradient(y_lower, dx), dx)

    upper_surface_curvature_mean = float(np.mean(np.abs(upper_second)))
    lower_surface_curvature_mean = float(np.mean(np.abs(lower_second)))

    upper_smoothness = np.mean(np.abs(np.diff(y_upper, n=2)))
    lower_smoothness = np.mean(np.abs(np.diff(y_lower, n=2)))
    surface_smoothness = float(upper_smoothness + lower_smoothness)

    # Leading/trailing edge noktalarında thickness doğal olarak sıfıra yaklaşabilir.
    # Bu nedenle local thickness kontrolü iç bölgede yapılır.
    interior_mask = (x >= 0.02) & (x <= 0.98)
    interior_thickness = thickness[interior_mask]
    min_local_thickness = float(np.min(interior_thickness))

    area_proxy = _trapezoid_integral(np.maximum(thickness, 0.0), x)

    # finite_ok = bool(
    #     np.all(np.isfinite(y_upper))
    #     and np.all(np.isfinite(y_lower))
    #     and np.all(np.isfinite(thickness))
    # )

    # no_surface_intersection = bool(min_local_thickness >= min_local_thickness_required)
    # reasonable_scale = bool(
    #     np.max(np.abs(y_upper)) <= max_abs_surface_y
    #     and np.max(np.abs(y_lower)) <= max_abs_surface_y
    #     and max_thickness > 0.0
    #     and max_thickness <= 0.50
    # )

    min_required = float(min_local_thickness_required)

    local_thickness_violation = max(
        0.0,
        min_required - float(min_local_thickness),
    )

    is_local_thickness_feasible = bool(local_thickness_violation <= 1e-8)

    finite_ok = bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y_upper))
        and np.all(np.isfinite(y_lower))
        and np.all(np.isfinite(thickness))
        and np.isfinite(max_thickness)
        and np.isfinite(min_local_thickness)
    )

    surface_intersection_ok = bool(min_local_thickness > 0.0)

    max_abs_y = max(
        float(np.max(np.abs(y_upper))),
        float(np.max(np.abs(y_lower))),
    )

    surface_bound_ok = bool(max_abs_y <= float(max_abs_surface_y))

    thickness_scale_ok = bool(
        max_thickness > 0.0
        and max_thickness <= 0.50
    )

    is_valid = bool(
        finite_ok
        and surface_intersection_ok
        and surface_bound_ok
        and thickness_scale_ok
    )

    return GeometryFeatures(
        max_thickness=max_thickness,
        x_max_thickness=x_max_thickness,
        max_camber=max_camber,
        x_max_camber=x_max_camber,
        leading_edge_radius_proxy=leading_edge_radius_proxy,
        trailing_edge_thickness=trailing_edge_thickness,
        upper_surface_curvature_mean=upper_surface_curvature_mean,
        lower_surface_curvature_mean=lower_surface_curvature_mean,
        surface_smoothness=surface_smoothness,
        min_local_thickness=min_local_thickness,
        local_thickness_violation=local_thickness_violation,
        is_local_thickness_feasible=is_local_thickness_feasible,
        area_proxy=area_proxy,
        is_valid=is_valid,
    )