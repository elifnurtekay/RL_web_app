from __future__ import annotations

import numpy as np

from rl_airfoil.geometry.cst import compute_cst_geometry_features


def compute_geometry_features(cst: list[float] | np.ndarray) -> dict:
    """Return JSON-safe geometry descriptors for a CST8 airfoil.

    The XAI layer deliberately reuses the same geometry feature implementation as
    the optimization environment so local explanations are aligned with the
    surrogate/constraint pipeline used by the web inference path.
    """
    features = compute_cst_geometry_features(np.asarray(cst, dtype=float).reshape(8))
    row = features.to_dict()
    return {
        key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_))
    }
