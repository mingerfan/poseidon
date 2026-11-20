"""A2 new tiny symbolic planners and callback replay only; no model/compiler/API."""
import copy
from itertools import combinations
import math
import unittest

from pt_agent.experiments import a2_native_profile_planner as p
from pt_agent.experiments import a2_native_profile_runtime as r
from pt_agent.backends import CountBackend
from pt_agent import stage_plan


def config():
    keys=('add_double','add_single','bootstrap_single','modswitch_single','negate_single','mul_double','mul_single','rescale_single','rotate_single')
    table={'earth.'+k:[float(j+1)for i in range(30)]for j,k in enumerate(keys)}
    return dict(runtime='HEAAN-HEVM',polynomialDegree=131072,rescalingFactor=51,
        levelLowerBound=1,levelUpperBound=29,bootstrapLevelLowerBound=3,bootstrapLevelUpperBound=16,latencyTable=table)


def chain(length,shortcut=False,group=1):
    def run(active):
        active.op_context='tiny';x=active.input(0);y=x
        for i in range(length):
            active.stage('s'+str(i//group));y=y*y
        if shortcut:y=y+x
        active.output([y])
    return run


class NativePlannerCases(unittest.TestCase):
    def test_original_profile_and_normalization_are_explicit(self):
        original=config();model=p.model_for(original)
        self.assertEqual((model.input_level,model.boot_in_level,model.raw_boot_out_level,model.boot_out_level),(16,3,16,15))
        self.assertEqual(model.boot_us,original['latencyTable']['earth.bootstrap_single'][3]+original['latencyTable']['earth.rescale_single'][16])
        for key,value in(('polynomialDegree',65536),('rescalingFactor',40),('bootstrapLevelLowerBound',1)):
            changed=copy.deepcopy(original);changed[key]=value
            with self.subTest(key=key),self.assertRaises(p.NativePlanError):p.model_for(changed)

    def test_initial13_depths_need_no_boot_14_need_one(self):
        model=p.model_for(config())
        for length,boots in((13,0),(14,1)):
            run=chain(length);plans,summary,selected=r.plan(run,8,model)
            self.assertEqual(set(plans),{'depth-dp','depth-dp+prune','stage-cut','stage-cut+prune'})
            self.assertTrue(all(row['boots']==boots for row in summary.values()))
            proofs=r.validate_all(run,8,model,plans)
            self.assertTrue(all(row['status']=='ok'for row in proofs.values()))
            self.assertEqual(summary[selected]['latency_us'],min(row['latency_us']for row in summary.values()))

    def test_normalized_post_boot12_depth_budget_is_enforced(self):
        model=p.model_for(config())
        for length,boots in((25,1),(26,2)):
            run=chain(length);plans,summary,_=r.plan(run,8,model)
            self.assertEqual(summary['depth-dp']['boots'],boots)
            self.assertEqual(summary['stage-cut']['boots'],boots)
            proof=r.validate_all(run,8,model,plans)
            self.assertTrue(all(v['count']['normalizations']==v['count']['callback_bootstraps']for v in proof.values()))
            self.assertTrue(all(v['count']['logical_min_level']>=3 for v in proof.values()))

    def test_fresh_high_level_shortcut_crosses_cuts_with_same_exact_replay(self):
        model=p.model_for(config());run=chain(39,shortcut=True)
        plans,summary,_=r.plan(run,8,model);proof=r.validate_all(run,8,model,plans)
        self.assertEqual(len(proof),4)
        for name,row in proof.items():
            self.assertEqual(row['count'],row['emit']);self.assertEqual(row['counts']['boot'],summary[name]['boots'])

    def test_stage_frontier_matches_exhaustive_label_cut_paths(self):
        from dataclasses import replace
        model=replace(p.model_for(config()),input_level=3,boot_in_level=1,raw_boot_out_level=3)
        run=chain(4,shortcut=True);dag,recorder=r.record(run,8,model)
        labels=stage_plan.labels(dag,recorder);distinct,li,users,last,groups=p._stage_structure(dag,labels)
        costs=[]
        for size in range(len(groups)):
            for cuts in combinations(range(len(groups)-1),size):
                state={nid:(model.input_level,False)for nid,k in enumerate(dag.kind)if k=='input'and last[nid]>=0};cost=0.;valid=True
                for label,group in enumerate(groups):
                    observed=p._advance(dag,group,state,last,label,model)
                    if observed is None:valid=False;break
                    state,delta=observed;cost+=delta
                    if label in cuts:state,delta,_=p._cut(state,model);cost+=delta
                if valid:costs.append(cost)
        selected,_=p.stage_candidate(dag,labels,model)
        self.assertTrue(costs);self.assertTrue(math.isclose(selected.latency_us,min(costs),rel_tol=1e-12))

    def test_missing_boot_or_wrong_normalization_replay_is_rejected(self):
        model=p.model_for(config());run=chain(27);plans,_,_=r.plan(run,8,model)
        bad=copy.deepcopy(plans['depth-dp']);bad['substitutions']=[];bad['outputs']=[];bad['boots']=0
        with self.assertRaises(Exception):r.run_checked(run,8,model,bad,CountBackend(8,input_level=16,boot_out_level=16))

    def test_dead_raw_calls_are_kept_and_original_count_scope_restores(self):
        model=p.model_for(config())
        def run(active):
            active.op_context='dead-fixture';x=active.input(0);unused=x
            active.stage('unused')
            for _ in range(40):unused=unused*unused
            active.stage('live');y=x+x;active.output([y])
        dag,recorder=r.record(run,8,model,frontend_normalize=True)
        found=p.candidates(dag,recorder,model);plans={}
        for name,row in found.items():plans[name]=r.express(dag,recorder,row['placement'],model,name)[0]
        self.assertTrue(all(plan['dead']for plan in plans.values()))
        proof=r.validate_all(run,8,model,plans)
        self.assertTrue(all(row['counts']['mul_cc']==40 for row in proof.values()))
        self.assertTrue(all(row['counts']['boot']==0 for row in proof.values()))
        inner=CountBackend(8,input_level=16,boot_out_level=16)
        r.run_checked(run,8,model,plans['depth-dp'],inner)
        self.assertEqual(inner.input_level,16)
        self.assertNotIn('_apply',inner.__dict__)
        self.assertNotIn('_component_dead_count_scope',inner.__dict__)

    def test_frontier_limits_fail_explicitly_without_partial_success(self):
        model=p.model_for(config());run=chain(30);dag,recorder=r.record(run,8,model)
        with self.assertRaisesRegex(p.NativePlanError,'state allowance'):
            p.stage_candidate(dag,stage_plan.labels(dag,recorder),model,state_limit=1)
        with self.assertRaises(TimeoutError):p.stage_candidate(dag,stage_plan.labels(dag,recorder),model,time_limit=-1)


if __name__=='__main__':unittest.main(verbosity=2)
