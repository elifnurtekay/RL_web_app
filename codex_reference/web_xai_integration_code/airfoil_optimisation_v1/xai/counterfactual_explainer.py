from __future__ import annotations
from copy import deepcopy
from .feature_builder import extract_cst
from .target_selector import is_step_feasible
from .utils import safe_float
class CounterfactualExplainer:
    def __init__(self, perturbation_rates=(-0.03,-0.01,0.01,0.03)): self.perturbation_rates=perturbation_rates
    def explain(self, target_step, user_input, solver_fn, max_results=12):
        if solver_fn is None: return {'available': False, 'reason':'counterfactual için solver_fn verilmedi'}
        base_cst = extract_cst(target_step)
        aoa = safe_float(target_step.get('AoA', user_input.get('aoa', user_input.get('AoA', 0.0))), 0.0)
        re = safe_float(target_step.get('Re', user_input.get('re', user_input.get('Re', 1e6))), 1e6)
        base_clcd = safe_float(target_step.get('CL_CD', target_step.get('cl_cd')), None)
        base_cm = safe_float(target_step.get('CM', target_step.get('cm')), None)
        base_tc = safe_float(target_step.get('t_c', target_step.get('tc')), None)
        names=['CST_u1','CST_u2','CST_u3','CST_u4','CST_l1','CST_l2','CST_l3','CST_l4']; rows=[]
        for i,name in enumerate(names):
            for rate in self.perturbation_rates:
                new = deepcopy(base_cst); delta = new[i]*rate if abs(new[i])>1e-8 else rate; new[i]+=delta
                try: pred = solver_fn(new, aoa, re)
                except Exception as exc:
                    rows.append({'change':f'{name} {rate:+.0%}','feature':name,'rate':rate,'available':False,'reason':str(exc)}); continue
                clcd=safe_float(pred.get('CL_CD', pred.get('cl_cd')), None); cm=safe_float(pred.get('CM', pred.get('cm')), None); tc=safe_float(pred.get('t_c', pred.get('tc')), None)
                rows.append({'change':f'{name} {rate:+.0%}','feature':name,'rate':rate,'available':True,'CL':safe_float(pred.get('CL', pred.get('cl')), None),'CD':safe_float(pred.get('CD', pred.get('cd')), None),'CM':cm,'CL_CD':clcd,'t_c':tc,'delta_CL_CD':clcd-base_clcd if clcd is not None and base_clcd is not None else None,'delta_CM':cm-base_cm if cm is not None and base_cm is not None else None,'delta_t_c':tc-base_tc if tc is not None and base_tc is not None else None,'constraint_status':'safe' if is_step_feasible({'CM':cm,'t_c':tc,'is_geometry_valid':pred.get('is_geometry_valid', True)}) else 'risky'})
        valid=[r for r in rows if r.get('available') and r.get('delta_CL_CD') is not None]
        return {'available': True, 'base': {'CL_CD': base_clcd, 'CM': base_cm, 't_c': base_tc}, 'top_sensitive_changes': sorted(valid, key=lambda r: abs(r['delta_CL_CD']), reverse=True)[:max_results], 'all_evaluated_count': len(rows)}
