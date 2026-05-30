from __future__ import annotations
import numpy as np

def split_cst(cst):
    arr = np.asarray(cst, dtype=float)
    if arr.size != 8: raise ValueError(f'CST8 bekleniyor; gelen boyut: {arr.size}')
    return arr[:4], arr[4:]

def approximate_airfoil_curve(cst, n: int = 101):
    upper, lower = split_cst(cst)
    x = np.linspace(0, 1, n)
    basis = np.vstack([(1-x)**3, 3*x*(1-x)**2, 3*x**2*(1-x), x**3]).T
    y_u = basis @ upper
    y_l = -(basis @ np.abs(lower)) if np.mean(lower) > 0 else basis @ lower
    return {'x': x, 'y_upper': y_u, 'y_lower': y_l, 'thickness': y_u-y_l, 'camber': 0.5*(y_u+y_l)}

def compute_geometry_features(cst):
    c = approximate_airfoil_curve(cst)
    x, y_u, y_l, t, camber = c['x'], c['y_upper'], c['y_lower'], c['thickness'], c['camber']
    max_t_idx, max_c_idx = int(np.argmax(t)), int(np.argmax(np.abs(camber)))
    upper_curv = float(np.mean(np.abs(np.diff(y_u, n=2))))
    lower_curv = float(np.mean(np.abs(np.diff(y_l, n=2))))
    return {
        'max_thickness': float(np.max(t)), 'x_max_thickness': float(x[max_t_idx]),
        'max_camber': float(camber[max_c_idx]), 'x_max_camber': float(x[max_c_idx]),
        'leading_edge_radius_proxy': float(abs(y_u[1]-y_l[1])),
        'trailing_edge_thickness': float(abs(y_u[-1]-y_l[-1])),
        'upper_surface_curvature_mean': upper_curv, 'lower_surface_curvature_mean': lower_curv,
        'surface_smoothness': float(upper_curv+lower_curv), 'min_local_thickness': float(np.min(t)),
        'area_proxy': float(np.trapz(t, x)),
    }
