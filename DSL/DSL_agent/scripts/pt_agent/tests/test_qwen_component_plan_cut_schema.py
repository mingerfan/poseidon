"""Selected-placement saved-plan schemas; no planner, Count or model execution."""
import copy
import unittest

from pt_agent.experiments.qwen_component_plan_cache import validate_plan
from pt_agent.plan import PLAN_FORMAT


def saved_plan(placement, cuts):
    return dict(format=PLAN_FORMAT, source='pt_agent.lower', placement=placement,
        calls=40, boots=1, barriers=1, levels=dict(input_level=5,boot_in_level=1,boot_out_level=5),
        substitutions=[[34,0,7]], outputs=[], dead=[], merged=[], cuts=cuts,
        latency_us=1., timing=dict(lower_s=0.,dp_s=0.,candidates_s=0.,map_s=0.),
        candidates={placement:dict(seconds=0.)})


class SavedPlanCutSchemas(unittest.TestCase):
    def accepts_unchanged(self, plan):
        before=copy.deepcopy(plan)
        self.assertIs(validate_plan(plan,32),plan)
        self.assertEqual(plan,before)

    def test_depth_dp_integer_depth_cuts_remain_valid(self):
        for placement in ('depth-dp','depth-dp+prune'):
            with self.subTest(placement=placement):
                self.accepts_unchanged(saved_plan(placement,[0,3,7]))

    def test_stage_cut_statement_stage_local_depth_triples_are_valid(self):
        for placement in ('stage-cut','stage-cut+prune'):
            with self.subTest(placement=placement):
                self.accepts_unchanged(saved_plan(placement,[[0,0,0],[3,3,4],[3,4,1],[3,4,5],[4,0,12]]))

    def test_empty_cut_list_remains_valid_for_all_selected_placements(self):
        for placement in ('depth-dp','depth-dp+prune','stage-cut','stage-cut+prune'):
            with self.subTest(placement=placement):self.accepts_unchanged(saved_plan(placement,[]))

    def test_cuts_cannot_mix_or_use_the_other_selected_placement_schema(self):
        for placement in ('depth-dp','depth-dp+prune'):
            for cuts in ([[1,2,3]],[1,[1,2,3]]):
                with self.subTest(placement=placement,cuts=cuts),self.assertRaises(ValueError):
                    validate_plan(saved_plan(placement,cuts),32)
        for placement in ('stage-cut','stage-cut+prune'):
            for cuts in ([1],[1,[1,2,3]],[[1,2,3],1]):
                with self.subTest(placement=placement,cuts=cuts),self.assertRaises(ValueError):
                    validate_plan(saved_plan(placement,cuts),32)

    def test_stage_cut_triple_container_and_length_are_strict(self):
        for row in ([],[1],[1,2],[1,2,3,4],(1,2,3),'123',None):
            with self.subTest(row=row),self.assertRaises(ValueError):
                validate_plan(saved_plan('stage-cut+prune',[row]),32)

    def test_bools_negative_and_noninteger_cut_values_are_rejected(self):
        for value in (True,False,-1,1.0,'1',None):
            for placement in ('depth-dp','depth-dp+prune'):
                with self.subTest(value=value,placement=placement),self.assertRaises(ValueError):
                    validate_plan(saved_plan(placement,[value]),32)
            for placement in ('stage-cut','stage-cut+prune'):
                for index in range(3):
                    row=[1,2,3];row[index]=value
                    with self.subTest(value=value,placement=placement,index=index),self.assertRaises(ValueError):
                        validate_plan(saved_plan(placement,[row]),32)

    def test_merged_indices_stay_strict_and_bootstrap_coverage_still_required(self):
        for placement,cuts in (('depth-dp',[3]),('depth-dp+prune',[3]),
                ('stage-cut',[[1,2,3]]),('stage-cut+prune',[[1,2,3]])):
            plan=saved_plan(placement,cuts);plan['merged']=[0,4]
            self.accepts_unchanged(plan)
            for value in (True,False,-1,1.0,'1',[1,2,3],None):
                bad=copy.deepcopy(plan);bad['merged']=[value]
                with self.subTest(placement=placement,merged=value),self.assertRaises(ValueError):validate_plan(bad,32)
            bad=copy.deepcopy(plan);bad['boots']=2
            with self.subTest(placement=placement,coverage='wrong'),self.assertRaisesRegex(ValueError,'bootstrap coverage'):
                validate_plan(bad,32)


if __name__=='__main__':unittest.main(verbosity=2)
