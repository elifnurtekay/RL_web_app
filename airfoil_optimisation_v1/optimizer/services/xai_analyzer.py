from __future__ import annotations


class XAIPolicyAnalyzer:
    """Surrogate/policy interpretability analyzer.

    Placeholder logic computes deterministic proxy attributions from CST inputs.
    Replace with SHAP/IG/attention-map pipelines in production.
    """

    def analyze(self, payload: dict, result: dict) -> dict:
        upper = payload['upper_weights']
        lower = payload['lower_weights']

        geom_importance = {
            'upper_surface_cst': round(sum(abs(x) for x in upper), 4),
            'lower_surface_cst': round(sum(abs(x) for x in lower), 4),
            'leading_edge_weight': round(abs(payload['leading_edge_weight']), 4),
            'trailing_edge_offset': round(abs(payload['trailing_edge_offset']) * 50, 4),
            'aoa': round(abs(payload['aoa']) / 10, 4),
        }

        ranked = sorted(geom_importance.items(), key=lambda kv: kv[1], reverse=True)
        top_features = [{'feature': k, 'score': v} for k, v in ranked[:3]]

        cm = result['metrics']['cm']
        tc = result['metrics']['tc']
        sensitivity = {
            'cm_risk_zone': 'low' if -0.12 <= cm <= 0.02 else 'high',
            'tc_risk_zone': 'low' if 0.08 <= tc <= 0.18 else 'high',
            'dominant_tradeoff': 'lift-vs-drag boundary shaping via upper CST influence',
        }

        return {
            'feature_importance': top_features,
            'sensitivity': sensitivity,
            'attention_proxy': {
                'region': '0.2c-0.7c camber corridor',
                'explanation': 'Policy attention concentrates on mid-chord geometry where CL/CD response is steep.',
            },
        }
