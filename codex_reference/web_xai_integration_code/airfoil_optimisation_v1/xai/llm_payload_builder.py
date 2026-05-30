from __future__ import annotations
class LLMPayloadBuilder:
    def build(self, algorithm, target_step, surrogate_xai, policy_xai, trajectory_xai, counterfactual_xai):
        return {'algorithm': algorithm.upper(), 'selected_step': {'step_id': target_step.get('step_id', target_step.get('step')), 'selection_reason': target_step.get('selection_reason'), 'CL': target_step.get('CL', target_step.get('cl')), 'CD': target_step.get('CD', target_step.get('cd')), 'CM': target_step.get('CM', target_step.get('cm')), 'CL_CD': target_step.get('CL_CD', target_step.get('cl_cd')), 't_c': target_step.get('t_c', target_step.get('tc'))}, 'surrogate_xai': self._compact_xai(surrogate_xai), 'policy_xai': self._compact_xai(policy_xai), 'trajectory_xai': trajectory_xai, 'counterfactual_xai': counterfactual_xai, 'instruction': 'Bu JSON verilerine dayanarak optimize edilmiş airfoil tasarımını teknik ama anlaşılır şekilde açıkla. Kesin olmayan iddia kurma. CM ve t/c kısıt durumunu belirt. SHAP katkılarını sebep-sonuç gibi değil, model açıklaması olarak ifade et.'}
    def _compact_xai(self, xai_block):
        compact={}
        for target,result in xai_block.get('targets',{}).items():
            if not result.get('available'): compact[target]={'available':False,'reason':result.get('reason')}; continue
            shap=result.get('shap',{})
            compact[target]={'prediction':result.get('prediction'),'top_positive':shap.get('top_positive',[])[:5],'top_negative':shap.get('top_negative',[])[:5],'top_overall':shap.get('top_overall',[])[:5]}
        return compact
