"""Tiny capture callback and immutable source checks; no real sample reads."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class PrebootCaptureChecks(unittest.TestCase):
    def test_exact_readonly_vector_capture_preserves_input_and_ranges(self):
        import numpy as np
        from pt_agent.experiments.r31_lattigo_input_worker import make_bootstrap_plan
        from pt_agent.experiments.r31_lattigo_math_worker import replay_plan
        from pt_agent.experiments.r31_lattigo_preboot_capture import capture_vector
        plan=make_bootstrap_plan(dict(target_id='fixture'))
        plan['values']=[v for v in plan['values'] if v['id'] in ('x','refreshed0')]
        plan['initialization']=[];plan['execution']=plan['execution'][:1];plan['final_outputs']=['refreshed0']
        manifest=dict(bundle_format_version=1,version=1,bundle_id=plan['plaintext_bundle']['id'],blobs=[])
        x=np.arange(16384,dtype=np.float64)/32768;original=x.copy()
        with TemporaryDirectory() as temp:
            _,summary=replay_plan(plan,manifest,Path(temp),x)
            out=Path(temp)/'captured.bin';row,replay=capture_vector(plan,manifest,Path(temp),x,summary['bootstrap_inputs'][0],out)
            self.assertTrue(np.array_equal(np.fromfile(out,dtype='<f8'),original))
            self.assertTrue(np.array_equal(x,original));self.assertEqual(row['absmax'],float(x.max()))
            self.assertEqual(replay['execution_completed'],1)
            changed=dict(row,absmax=row['absmax']+.1)
            with self.assertRaises(ValueError):capture_vector(plan,manifest,Path(temp),x,changed,out)


if __name__=='__main__':unittest.main()
