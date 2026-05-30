from __future__ import annotations
import math
from .config import CM_MAX, CM_MIN, TC_MAX, TC_MIN
from .geometry_features import compute_geometry_features
from .utils import l2_norm, safe_float

def _get_any(data, keys, default=None):
    for key in keys:
        if key in data and data[key] is not None: return data[key]
    return default

def normalize_cst_dict(cst):
    if len(cst) != 8: raise ValueError(f'CST8 bekleniyor; gelen boyut: {len(cst)}')
    return {f'CST_u{i+1}': float(cst[i]) for i in range(4)} | {f'CST_l{i+1}': float(cst[i+4]) for i in range(4)}

def extract_cst(step):
    if step.get('cst') is not None: return [float(x) for x in step['cst']]
    keys = ['CST_u1','CST_u2','CST_u3','CST_u4','CST_l1','CST_l2','CST_l3','CST_l4']
    if all(k in step for k in keys): return [float(step[k]) for k in keys]
    skeys = ['state_CST_u1','state_CST_u2','state_CST_u3','state_CST_u4','state_CST_l1','state_CST_l2','state_CST_l3','state_CST_l4']
    if all(k in step for k in skeys): return [float(step[k]) for k in skeys]
    raise ValueError('Step içinde CST bilgisi bulunamadı.')

def extract_action(step):
    if step.get('action') is not None: return [float(x) for x in step['action']]
    keys = ['action_u1','action_u2','action_u3','action_u4','action_l1','action_l2','action_l3','action_l4']
    return [float(step[k]) for k in keys] if all(k in step for k in keys) else None

def build_feature_row(user_input, target_step):
    cst = extract_cst(target_step)
    geo = compute_geometry_features(cst)
    aoa = safe_float(_get_any(target_step, ['AoA','aoa'], _get_any(user_input, ['AoA','aoa'], 0.0)), 0.0)
    re = safe_float(_get_any(target_step, ['Re','re'], _get_any(user_input, ['Re','re','reynolds'], 1e6)), 1e6)
    log10_re = safe_float(_get_any(target_step, ['log10_Re','log10_re'], None), None) or math.log10(max(re, 1.0))
    cl = safe_float(_get_any(target_step, ['CL','cl'], 0.0), 0.0)
    cd = safe_float(_get_any(target_step, ['CD','cd'], 1.0), 1.0)
    cm = safe_float(_get_any(target_step, ['CM','cm'], 0.0), 0.0)
    cl_cd = safe_float(_get_any(target_step, ['CL_CD','cl_cd'], None), None) or cl / max(cd, 1e-8)
    tc = safe_float(_get_any(target_step, ['t_c','tc'], geo.get('max_thickness')), geo.get('max_thickness'))
    row = {**normalize_cst_dict(cst), 'AoA': aoa, 'Re': re, 'log10_Re': log10_re, 'CL': cl, 'CD': cd, 'CM': cm, 'CL_CD': cl_cd, 't_c': tc, **geo,
           'CM_lower_distance': cm-CM_MIN, 'CM_upper_distance': CM_MAX-cm, 'CM_abs': abs(cm), 'tc_lower_distance': tc-TC_MIN, 'tc_upper_distance': TC_MAX-tc}
    action = extract_action(target_step)
    if action:
        row.update({f'action_u{i+1}': action[i] for i in range(4)})
        row.update({f'action_l{i+1}': action[i+4] for i in range(4)})
        row.update({'action_norm': l2_norm(action), 'upper_action_norm': l2_norm(action[:4]), 'lower_action_norm': l2_norm(action[4:])})
    for key in ['Q_min','Q_disagreement','Q1','Q2','actor_std_mean','policy_entropy','alpha','value_V','advantage','policy_std_mean']:
        if key in target_step: row[key] = safe_float(target_step[key], None)
    return row
