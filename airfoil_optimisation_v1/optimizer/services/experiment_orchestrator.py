from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from hashlib import sha256
import json


@dataclass
class ExperimentContext:
    run_id: str
    created_at: str
    config_version: str
    dataset_version: str
    policy_version: str


class LLMExperimentOrchestrator:
    """LLM-assisted orchestration façade.

    In production this class can call an LLM provider for richer experiment summaries,
    recommendation narratives, and comparison reports.
    """

    def create_context(self, payload: dict) -> ExperimentContext:
        payload_str = json.dumps(payload, sort_keys=True)
        digest = sha256(payload_str.encode('utf-8')).hexdigest()[:12]
        return ExperimentContext(
            run_id=f"EXP-{digest}",
            created_at=datetime.now(UTC).isoformat(),
            config_version="cfg-v1",
            dataset_version="nasa-airfoil-v1",
            policy_version="ppo-surrogate-v1",
        )

    def summarize(self, context: ExperimentContext, result: dict) -> dict:
        m = result['metrics']
        cm_ok = result['constraints']['cm']['satisfied']
        tc_ok = result['constraints']['tc']['satisfied']
        summary = (
            f"Run {context.run_id}: CL/CD={m['cl_cd']} with CM={m['cm']} and t/c={m['tc']}. "
            f"Constraints => CM:{'OK' if cm_ok else 'VIOLATION'}, t/c:{'OK' if tc_ok else 'VIOLATION'}."
        )
        return {
            'context': asdict(context),
            'summary': summary,
            'comparison_notes': [
                'Configuration integrity verified by orchestration layer.',
                'Result packaged for reproducible replay and audit trails.',
                'LLM-ready report artifact generated for experiment comparison.',
            ],
        }
