"""E6r: same-baseline alternatives, structural composition and bounded rounds."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.planners import transformer_tuner as tuner
from pt_agent.planners.prompts import build_tune_prompt
from pt_agent.tests.test_e6_tuning import Client,simulation
from pt_agent.tests.test_r624_floor import measured
from pt_agent.tests.test_u4_squared_softmax import fixture


def move(mid,op=1,degree=7,*,prefix='act',kind='approximation',changes=None,attrs=None):
    updates=changes or {'deg':dict(old=15,new=degree)}
    return dict(id=mid,line=5,call_lines=[],point=dict(op_id=op,prefix=prefix,role=prefix),
                move=dict(kind=kind,role=prefix,attrs=attrs or dict(deg=degree)),changes=updates)


class AlternativesTests(unittest.TestCase):
    def test_all_same_point_alternatives_are_estimated_before_lowest_valid_wins(self):
        menu=[move('a',degree=3),move('b',degree=5),move('c',degree=7)]
        def menus(program,*_):return menu if program.ops[1].attrs['deg']==15 else []
        def validate(program,*_,**kwargs):return measured(program.ops[1].attrs['deg']!=3)
        result,estimates,_=simulation(Client([dict(candidates=[dict(moves=[k]) for k in ('c','a','b')])]),
            menu_override=menus,validation_override=validate)
        row=result.history[-2];self.assertEqual(result.program.ops[1].attrs['deg'],5)
        self.assertEqual(len(estimates),5)
        self.assertTrue(all(r['evaluated'] for r in row['candidates']))
        self.assertEqual([r.get('rejected') for r in row['candidates']],['not_reached','coarse',None])
        self.assertNotIn('stale',json.dumps(row))
        self.assertEqual(row['accepted_candidates'],1)

    def test_nonconflicting_candidates_merge_on_round_baseline(self):
        client=Client([dict(candidates=[dict(moves=['m2']),dict(moves=['m1'])])])
        result,estimates,_=simulation(client)
        row=result.history[-2]
        self.assertEqual([p.attrs['deg'] for p in tuner.points(result.program)],[7,7])
        self.assertEqual((row['accepted_candidates'],row['merge_attempts'],row['merge_accepted']),(2,1,1))
        self.assertEqual(len(estimates),5)
        self.assertTrue(row['candidates'][1]['merge']['accepted'])

    def test_failed_standalone_candidate_can_pass_after_merging(self):
        def validate(program,*_,**kwargs):
            degrees=[p.attrs['deg'] for p in tuner.points(program)]
            return measured(degrees!=[7,15])
        result,_,_=simulation(Client([dict(candidates=[dict(moves=['m1']),dict(moves=['m2'])])]),
                              validation_override=validate)
        row=result.history[-2]
        self.assertEqual([p.attrs['deg'] for p in tuner.points(result.program)],[7,7])
        self.assertEqual(row['candidates'][0]['rejected'],'coarse')
        self.assertTrue(row['candidates'][0]['merge']['accepted'])
        self.assertEqual(row['merge_accepted'],1)

    def test_length_continues_and_two_empty_rounds_stop(self):
        length=dict(content='',finish_reason='length')
        client=Client([length,dict(candidates=[dict(moves=['m1'])]),length,length])
        result,_,_=simulation(client,degree=127)
        rounds=[r for r in result.history if r['phase']=='llm_round']
        self.assertEqual(client.calls,4)
        self.assertEqual([r['accepted_candidates'] for r in rounds],[0,1,0,0])
        self.assertTrue(result.converged)
        prior=json.loads(client.requests[1][0][1]['content'])['previous_rounds']
        self.assertEqual(prior[0]['finish_reason'],'length')
        client=Client([dict(content='bad',finish_reason='stop')]*3)
        result,_,_=simulation(client)
        self.assertEqual(client.calls,2)

    def test_four_round_cap_with_accepted_steps_and_call_budget(self):
        client=Client([dict(candidates=[dict(moves=['m1'])])]*5)
        result,_,_=simulation(client,degree=127)
        self.assertEqual((client.calls,result.sweeps),(4,4));self.assertFalse(result.converged)
        client=Client([dict(content='',finish_reason='length')],limit=1)
        simulation(client);self.assertEqual(client.calls,1)


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.program=parse(fixture(2));self.op=next(o for o in self.program.ops if o.kind=='softmax')
        self.struct=move('s',self.op.id,prefix='softmax',kind='steps',changes={'steps':dict(old=2,new=3)})
        self.role=move('r',self.op.id,prefix='exp',attrs=dict(method='remez',deg=7,fit='relative'),
                       changes={'exp_deg':dict(old=15,new=7)})

    def test_structure_then_legal_role_uses_rebuilt_attributes(self):
        observed=[]
        def moves(program,ctx,index):
            observed.append(program.ops[index].attrs['steps'])
            attrs=dict(program.ops[index].attrs,exp_deg=7,exp_domain=(-3,0))
            yield self.role['move'],attrs
        with patch.object(tuner,'softmax_moves',side_effect=moves):
            candidate,error=tuner._compose_moves(self.program,[self.role,self.struct],None)
        self.assertIsNone(error);self.assertEqual(observed,[3])
        self.assertEqual(candidate.ops[self.op.id].attrs['exp_domain'],(-3,0))
        self.assertEqual(self.program.ops[self.op.id].attrs['steps'],2)

    def test_role_illegal_after_structure(self):
        other=dict(self.role['move'],attrs=dict(method='remez',deg=15,fit='relative'))
        with patch.object(tuner,'softmax_moves',return_value=iter([(other,{})])):
            candidate,error=tuner._compose_moves(self.program,[self.struct,self.role],None)
        self.assertIsNone(candidate);self.assertEqual(error,'not_legal_after_structural')

    def test_roles_fields_and_structural_conflicts(self):
        inv=move('i',self.op.id,prefix='inv',changes={'inv_deg':dict(old=15,new=7)})
        self.assertFalse(tuner._conflicting_points([self.role,inv]))
        self.assertFalse(tuner._conflicting_points([self.struct,self.role,inv]))
        for pair in ([self.role,self.role],[self.struct,self.struct],
                     [move('a',changes={'deg':{}}),move('b',prefix='other',changes={'deg':{}})]):
            self.assertTrue(tuner._conflicting_points(pair))
        rows,_=tuner.parse_tune_reply(dict(content=json.dumps({'candidates':[{'moves':['r','r']}]})),[self.role])
        self.assertEqual(rows[0]['rejected'],'conflict')

    def test_old_t1_saved_and_new_snapshot(self):
        folder=Path(__file__).parent
        self.assertEqual(build_tune_prompt({},[],[],version='t1-e6')[0]['content'],
                         (folder/'e6_t1-e6_prompt.txt').read_text().rstrip('\n'))
        text=build_tune_prompt({},[],[])[0]['content']
        self.assertEqual(text,(folder/'e6_t1_prompt.txt').read_text().rstrip('\n'))
        for term in ('alternatives','3 nonconflicting merges','rejection reason','structure'):
            self.assertIn(term,text)

if __name__=='__main__':unittest.main()
