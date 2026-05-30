from __future__ import annotations
import numpy as np
from .utils import row_from_feature_dict, top_shap_features

def explain_tree_model_local(model, feature_dict, feature_order, top_k=8):
    try:
        import shap
    except Exception as exc:
        return {'available': False, 'reason': f'shap import edilemedi: {exc}'}
    X = row_from_feature_dict(feature_dict, feature_order)
    try:
        pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return {'available': True, 'prediction': pred, 'feature_order': feature_order, 'shap': top_shap_features(feature_order, np.asarray(shap_values).reshape(-1), top_k)}
    except Exception as exc:
        return {'available': False, 'reason': f'local SHAP hesaplanamadı: {exc}'}
