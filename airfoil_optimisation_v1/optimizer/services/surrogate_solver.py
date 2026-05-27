import math

from .solver_interface import AerodynamicOptimizer, OptimizationInput


class ResMLPSurrogateOptimizer(AerodynamicOptimizer):
    """
    Web dashboard için DRL + surrogate inference servis katmanı.

    Şu anki sürüm:
    - PPO / TD3 / SAC model seçimini destekler.
    - Arayüz demosu için deterministik proxy sonuç üretir.
    - Gerçek model entegrasyonunda _predict_coefficients() ve _optimized_geometry()
      fonksiyonları checkpoint tabanlı inference ile değiştirilecektir.
    """

    MODEL_COEFFICIENTS = {
        "PPO": {"cl": 1.142, "cd": 0.0087, "cm": -0.038, "tc": 0.121},
        "TD3": {"cl": 1.168, "cd": 0.0081, "cm": -0.035, "tc": 0.124},
        "SAC": {"cl": 1.155, "cd": 0.0084, "cm": -0.040, "tc": 0.119},
    }

    def optimize(self, optimization_input: OptimizationInput) -> dict:
        cl, cd, cm, tc = self._predict_coefficients(optimization_input)
        cl_cd = cl / max(cd, 1e-6)

        cm_ok = -0.12 <= cm <= 0.02
        tc_ok = 0.08 <= tc <= 0.18
        feasible = cm_ok and tc_ok

        return {
            "status": "ok",
            "model": optimization_input.model,
            "metrics": {
                "cl": round(cl, 4),
                "cd": round(cd, 4),
                "cl_cd": round(cl_cd, 2),
                "cm": round(cm, 4),
                "tc": round(tc, 4),
                "is_feasible": feasible,
            },
            "constraints": {
                "cm": {
                    "value": float(cm),
                    "min": -0.12,
                    "max": 0.02,
                    "satisfied": cm_ok,
                },
                "tc": {
                    "value": float(tc),
                    "min": 0.08,
                    "max": 0.18,
                    "satisfied": tc_ok,
                },
            },
            "decision_logic": self._decision_logic(cl_cd, cm_ok, tc_ok),
            "pipeline": [
                "Initial airfoil profile analyzed",
                f"{optimization_input.model} policy inference completed",
                "Surrogate aerodynamic coefficients estimated",
                "Constraint verification completed",
                "Optimized geometry generated",
            ],
            "geometry": {
                "initial": self._baseline_geometry(),
                "optimized": self._optimized_geometry(optimization_input),
            },
        }

    def _predict_coefficients(self, inp: OptimizationInput):
        base = self.MODEL_COEFFICIENTS.get(
            inp.model.upper(),
            self.MODEL_COEFFICIENTS["PPO"],
        )

        upper_sum = sum(inp.upper_weights)
        lower_sum = sum(inp.lower_weights)

        geometry_gain = 0.015 * math.tanh(upper_sum - abs(lower_sum))
        aoa_gain = 0.004 * (inp.aoa - 2.5)

        cl = base["cl"] + geometry_gain + aoa_gain
        cd = max(0.0065, base["cd"] + 0.0001 * abs(lower_sum) - 0.0002 * aoa_gain)
        cm = base["cm"] - 0.01 * (inp.leading_edge_weight - 0.25)
        tc = max(0.08, min(0.18, base["tc"] + 0.008 * geometry_gain))

        return cl, cd, cm, tc

    def _baseline_geometry(self):
        xs = [i / 100 for i in range(101)]

        upper = [
            [x, 0.036 * math.sin(math.pi * x) * (1 - 0.10 * x)]
            for x in xs
        ]

        lower = [
            [x, -0.031 * math.sin(math.pi * x) * (1 - 0.05 * x)]
            for x in xs
        ]

        return {
            "upper": upper,
            "lower": lower,
        }

    def _optimized_geometry(self, inp: OptimizationInput):
        xs = [i / 100 for i in range(101)]

        upper_gain = max(0.0, sum(inp.upper_weights))
        lower_gain = abs(sum(inp.lower_weights))

        upper_amp = 0.046 + 0.020 * upper_gain
        lower_amp = 0.028 + 0.010 * lower_gain

        upper = [
            [x, upper_amp * math.sin(math.pi * x) * (1 - 0.20 * x)]
            for x in xs
        ]

        lower = [
            [x, -lower_amp * math.sin(math.pi * x) * (1 - 0.12 * x)]
            for x in xs
        ]

        return {
            "upper": upper,
            "lower": lower,
        }

    def _decision_logic(self, cl_cd: float, cm_ok: bool, tc_ok: bool):
        logic = []

        if cl_cd > 90:
            logic.append("If CL/CD > threshold → Accept aerodynamic improvement")
        else:
            logic.append("If CL/CD ≤ threshold → Continue policy search")

        if cm_ok:
            logic.append("If CM constraint satisfied → Maintain stability")
        else:
            logic.append("If CM constraint violated → Apply pitching moment penalty")

        if tc_ok:
            logic.append("If t/c is within structural interval → Geometry is feasible")
        else:
            logic.append("If t/c is outside [0.08, 0.18] → Reject geometry")

        return logic