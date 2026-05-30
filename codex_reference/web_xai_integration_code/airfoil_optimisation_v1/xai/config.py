from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
CM_MIN, CM_MAX = -0.12, 0.02
TC_MIN, TC_MAX = 0.08, 0.18
STATE_FEATURES = [
    'CST_u1','CST_u2','CST_u3','CST_u4','CST_l1','CST_l2','CST_l3','CST_l4',
    'AoA','log10_Re','CL','CD','CM','CL_CD','t_c',
    'max_thickness','x_max_thickness','max_camber','x_max_camber',
    'leading_edge_radius_proxy','trailing_edge_thickness',
    'upper_surface_curvature_mean','lower_surface_curvature_mean','surface_smoothness',
    'min_local_thickness','area_proxy','CM_lower_distance','CM_upper_distance','CM_abs',
    'tc_lower_distance','tc_upper_distance'
]
ACTION_FEATURES = ['action_u1','action_u2','action_u3','action_u4','action_l1','action_l2','action_l3','action_l4','action_norm','upper_action_norm','lower_action_norm']
SURROGATE_TARGETS = ['CL','CD','CM','CL_CD']
TD3_POLICY_TARGETS = ['action_norm','upper_action_norm','lower_action_norm','Q_min','Q_disagreement']
SAC_POLICY_TARGETS = ['action_norm','upper_action_norm','lower_action_norm','actor_std_mean','policy_entropy','Q_min','Q_disagreement']
PPO_POLICY_TARGETS = ['action_norm','upper_action_norm','lower_action_norm','value_V','advantage','policy_std_mean','policy_entropy']
@dataclass(frozen=True)
class XAIArtifactConfig:
    artifact_root: Path
    def surrogate_dir(self) -> Path: return self.artifact_root / 'surrogate'
    def algorithm_dir(self, algorithm: str) -> Path: return self.artifact_root / algorithm.lower()
