import json
from json import JSONDecodeError

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .services.experiment_orchestrator import LLMExperimentOrchestrator
from .services.solver_interface import OptimizationInput
from .services.surrogate_solver import ResMLPSurrogateOptimizer
from .services.xai_analyzer import XAIPolicyAnalyzer


class DashboardView(View):
    def get(self, request):
        return render(request, "optimizer/dashboard.html")


def _parse_float_list(values, field_name):
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list.")

    parsed = [float(v) for v in values]

    if len(parsed) == 0:
        raise ValueError(f"{field_name} cannot be empty.")

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
            upper_weights=_parse_float_list(payload["upper_weights"], "upper_weights"),
            lower_weights=_parse_float_list(payload["lower_weights"], "lower_weights"),
            leading_edge_weight=float(payload["leading_edge_weight"]),
            trailing_edge_offset=float(payload["trailing_edge_offset"]),
            model=model,
        )

        optimizer = ResMLPSurrogateOptimizer()
        result = optimizer.optimize(data)

        orchestrator = LLMExperimentOrchestrator()
        context = orchestrator.create_context(payload)
        experiment = orchestrator.summarize(context, result)

        xai = XAIPolicyAnalyzer().analyze(payload, result)

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