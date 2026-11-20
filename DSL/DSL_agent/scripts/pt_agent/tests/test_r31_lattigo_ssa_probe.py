"""Small algebra checks; no saved samples or native cryptography."""
from collections import Counter
import unittest

from pt_agent.experiments.r31_lattigo_ssa_probe import branch_plan, expected_values
from pt_agent.experiments.r31_lattigo_runtime_plan import plan_inventory


class BranchedSSAProbeChecks(unittest.TestCase):
    def test_every_native_kind_reuses_an_input_after_its_target(self):
        plan=branch_plan({'target_id':'fixture'})
        manifest=dict(bundle_format_version=1,version=1,bundle_id=plan['plaintext_bundle']['id'],
            blobs=[dict(content='ssa-fixture:'+name,byte_length=16384*8) for name in ('offset','half')])
        inventory=plan_inventory(plan,manifest)
        expected={'rotate','mul_cc','relinearize','rescale','mod_switch','add_cc','add_cp','negate','mul_cp'}
        reused={op['op'] for i,op in enumerate(plan['execution'])
            if any(set(op['inputs'])&set(later['inputs']) for later in plan['execution'][i+1:])}
        self.assertEqual(set(inventory['op_counts']),expected);self.assertEqual(reused,expected)
        self.assertEqual(inventory['bootstrap_count'],0)
        self.assertEqual(inventory['terminal_ciphertext_ids'],['y'])

    def test_independent_expansion_matches_all_branches(self):
        plan=branch_plan({'target_id':'fixture'})
        x=[(i%17-8)/32 for i in range(16384)]
        pool={'x':x,'offset':[.125]*16384,'half':[.5]*16384}
        for op in plan['execution']:
            a=pool[op['inputs'][0]];kind=op['op']
            b=pool[op['inputs'][1]] if len(op['inputs'])==2 else None
            if kind.startswith('add'):out=[v+w for v,w in zip(a,b)]
            elif kind.startswith('mul'):out=[v*w for v,w in zip(a,b)]
            elif kind=='negate':out=[-v for v in a]
            elif kind=='rotate':
                step=op['attrs']['steps'];out=[a[(i+step)%len(a)] for i in range(len(a))]
            else:out=list(a)
            pool[op['output']]=out
        self.assertEqual(pool['y'],expected_values(x))


if __name__=='__main__':unittest.main()
