from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
SURROGATE_TARGETS=['CL','CD','CM','CL_CD']
COMMON_STATE_FEATURES=['CST_u1','CST_u2','CST_u3','CST_u4','CST_l1','CST_l2','CST_l3','CST_l4','AoA','log10_Re','CL','CD','CM','CL_CD','t_c','max_thickness','x_max_thickness','max_camber','x_max_camber','leading_edge_radius_proxy','trailing_edge_thickness','upper_surface_curvature_mean','lower_surface_curvature_mean','surface_smoothness','min_local_thickness','area_proxy','CM_lower_distance','CM_upper_distance','CM_abs','tc_lower_distance','tc_upper_distance']
ACTION_FEATURES=['action_u1','action_u2','action_u3','action_u4','action_l1','action_l2','action_l3','action_l4','action_norm','upper_action_norm','lower_action_norm']
ALGORITHM_TARGETS={'td3':['action_norm','upper_action_norm','lower_action_norm','Q_min','Q_disagreement'],'sac':['action_norm','upper_action_norm','lower_action_norm','actor_std_mean','policy_entropy','Q_min','Q_disagreement'],'ppo':['action_norm','upper_action_norm','lower_action_norm','value_V','advantage','policy_std_mean','policy_entropy']}
def existing(df, cols): return [c for c in cols if c in df.columns]
def normalize(df):
    df=df.copy(); mp={'state_CST_u1':'CST_u1','state_CST_u2':'CST_u2','state_CST_u3':'CST_u3','state_CST_u4':'CST_u4','state_CST_l1':'CST_l1','state_CST_l2':'CST_l2','state_CST_l3':'CST_l3','state_CST_l4':'CST_l4','Q1_s_a':'Q1','Q2_s_a':'Q2','Q_min_s_a':'Q_min','Q_disagreement_abs':'Q_disagreement','entropy':'policy_entropy'}
    for s,d in mp.items():
        if s in df.columns and d not in df.columns: df[d]=df[s]
    if 'Q1' in df.columns and 'Q2' in df.columns:
        if 'Q_min' not in df.columns: df['Q_min']=df[['Q1','Q2']].min(axis=1)
        if 'Q_disagreement' not in df.columns: df['Q_disagreement']=(df['Q1']-df['Q2']).abs()
    return df
def train_save(df, target, feature_candidates, out_dir):
    if target not in df.columns: print(f'Atlandı, target yok: {target}'); return
    feats=existing(df, feature_candidates); data=df[feats+[target]].replace([np.inf,-np.inf],np.nan).dropna()
    if len(data)<30 or data[target].nunique()<=1: print(f'Atlandı: {target}, rows={len(data)}, unique={data[target].nunique() if len(data) else 0}'); return
    model=ExtraTreesRegressor(n_estimators=500, random_state=42, min_samples_leaf=2, max_features='sqrt', n_jobs=-1).fit(data[feats], data[target])
    out_dir.mkdir(parents=True, exist_ok=True); joblib.dump(model, out_dir/f'{target}.joblib'); (out_dir/f'{target}_features.json').write_text(json.dumps(feats, indent=2), encoding='utf-8')
    print(f'Kaydedildi: {out_dir/(target+".joblib")} | features={len(feats)} | rows={len(data)}')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--xai-dataset', required=True); ap.add_argument('--artifact-root', required=True); ap.add_argument('--algorithm', choices=['td3','sac','ppo','surrogate'], required=True); args=ap.parse_args()
    df=normalize(pd.read_csv(args.xai_dataset)); root=Path(args.artifact_root)
    if args.algorithm=='surrogate':
        for t in SURROGATE_TARGETS: train_save(df,t,COMMON_STATE_FEATURES,root/'surrogate')
    else:
        for t in ALGORITHM_TARGETS[args.algorithm]: train_save(df,t,COMMON_STATE_FEATURES+ACTION_FEATURES,root/args.algorithm)
if __name__=='__main__': main()
