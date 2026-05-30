import json
from json import JSONDecodeError

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .services.experiment_orchestrator import LLMExperimentOrchestrator
from .services.real_rl_solver import RealRLModelOptimizer
from .services.solver_interface import OptimizationInput
from django.conf import settings
from xai.xai_service import XAIService


class DashboardView(View):
    def get(self, request):
        return render(request, "optimizer/dashboard.html")


def _parse_float_list(values, field_name, expected_len=None):
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list.")

    parsed = [float(v) for v in values]

    if len(parsed) == 0:
        raise ValueError(f"{field_name} cannot be empty.")

    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError(
            f"{field_name} must contain exactly {expected_len} values."
        )

    return parsed


@csrf_exempt
@require_POST
def optimize_airfoil(request):
    try:
        payload = json.loads(request.body)

        model = str(payload.get("model", "PPO")).upper()

        if model not in {"PPO", "TD3", "SAC"}:
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Unsupported model. Use PPO, TD3, or SAC.",
                },
                status=400,
            )

        data = OptimizationInput(
            aoa=float(payload["aoa"]),
            reynolds=float(payload["reynolds"]),
            upper_weights=_parse_float_list(
                payload["upper_weights"],
                "upper_weights",
                expected_len=4,
            ),
            lower_weights=_parse_float_list(
                payload["lower_weights"],
                "lower_weights",
                expected_len=4,
            ),
            leading_edge_weight=float(payload.get("leading_edge_weight", 0.0)),
            trailing_edge_offset=float(payload.get("trailing_edge_offset", 0.0)),
            model=model,
        )

        optimizer = RealRLModelOptimizer()
        result = optimizer.optimize(data)

        orchestrator = LLMExperimentOrchestrator()
        context = orchestrator.create_context(payload)
        experiment = orchestrator.summarize(context, result)

        solver_fn = optimizer.make_solver_fn(data)
        xai = XAIService(
            artifact_root=getattr(settings, "XAI_ARTIFACT_ROOT", settings.BASE_DIR / "xai_artifacts")
        ).explain_optimized_airfoil(
            algorithm=model,
            user_input=payload,
            trajectory=result.get("trajectory", []),
            optimized_result={
                "step_id": result.get("rl_diagnostics", {}).get("selected_step_id"),
                "cst": result.get("rl_diagnostics", {}).get("optimized_cst"),
                "AoA": data.aoa,
                "Re": data.reynolds,
                "CL": result.get("metrics", {}).get("cl"),
                "CD": result.get("metrics", {}).get("cd"),
                "CM": result.get("metrics", {}).get("cm"),
                "CL_CD": result.get("metrics", {}).get("cl_cd"),
                "t_c": result.get("metrics", {}).get("tc"),
                "is_feasible": result.get("metrics", {}).get("is_feasible"),
                "is_geometry_valid": True,
            },
            solver_fn=solver_fn,
        )

        result["experiment"] = experiment
        result["xai"] = xai

        result["pipeline"] += [
            "Experiment summary generated",
            "XAI analysis completed",
        ]

        return JsonResponse(result)

    except KeyError as exc:
        return JsonResponse(
            {
                "status": "error",
                "message": f"Missing field: {str(exc)}",
            },
            status=400,
        )

    except (ValueError, TypeError, JSONDecodeError) as exc:
        return JsonResponse(
            {
                "status": "error",
                "message": str(exc),
            },
            status=400,
        )

    except Exception as exc:
        return JsonResponse(
            {
                "status": "error",
                "message": f"Unexpected backend error: {str(exc)}",
            },
            status=500,
        )