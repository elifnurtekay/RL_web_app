from __future__ import annotations
from .target_selector import is_step_feasible
from .utils import safe_float
class TrajectoryExplainer:
    def explain(self, trajectory, selected_step):
        if not trajectory: return {'available': False, 'reason':'trajectory boş'}
        initial, final = trajectory[0], trajectory[-1]
        clcd = [safe_float(s.get('CL_CD', s.get('cl_cd')), None) for s in trajectory]
        clcd = [v for v in clcd if v is not None]
        penalties = [safe_float(s.get('penalty_total'), 0.0) or 0.0 for s in trajectory]
        actions = [safe_float(s.get('action_norm'), None) for s in trajectory]
        actions = [v for v in actions if v is not None]
        feasible_count = sum(1 for s in trajectory if is_step_feasible(s))
        init_clcd = safe_float(initial.get('CL_CD', initial.get('cl_cd')), None)
        final_clcd = safe_float(final.get('CL_CD', final.get('cl_cd')), None)
        selected_clcd = safe_float(selected_step.get('CL_CD', selected_step.get('cl_cd')), None)
        return {'available': True, 'num_steps': len(trajectory), 'selected_step_id': selected_step.get('step_id', selected_step.get('step')), 'selection_reason': selected_step.get('selection_reason'), 'initial_CL_CD': init_clcd, 'final_CL_CD': final_clcd, 'selected_CL_CD': selected_clcd, 'best_CL_CD_in_trajectory': max(clcd) if clcd else None, 'delta_selected_vs_initial_CL_CD': (selected_clcd-init_clcd if selected_clcd is not None and init_clcd is not None else None), 'feasible_step_count': feasible_count, 'constraint_violation_step_count': len(trajectory)-feasible_count, 'max_penalty': max(penalties) if penalties else None, 'mean_action_norm': sum(actions)/len(actions) if actions else None, 'max_action_norm': max(actions) if actions else None, 'final_is_feasible': is_step_feasible(final), 'selected_is_feasible': is_step_feasible(selected_step)}
