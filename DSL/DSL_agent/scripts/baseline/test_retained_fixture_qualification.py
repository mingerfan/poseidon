"""Two inert intake controls; no native tool, array load or model forward."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery_component import bound, canonical, sha, write_once
from deepseek_provider import ProviderError
import retained_fixture_qualification as adapter


class RetainedFixtureIntakeControls(unittest.TestCase):
    def fixture(self, root):
        donor = root / 'donor'
        donor.mkdir()
        (donor / 'attempt-00').mkdir()
        raw_request = Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/candidate-replay-8h8lvb2n/request.json').read_bytes()
        self.assertEqual(sha(raw_request), '9376cd1592b4f1935dab27e8d9855622484d44ddb883c92f90a5e4ddab79947e')
        request = json.loads(raw_request)
        source = b'def retained_source():\n    pass\n'
        values = {'model.json': canonical(request['model']), 'weights.npz': b'opaque weights bytes',
            'arrays.npz': b'opaque input and reference bytes', 'request.json': canonical(request),
            'logical-bindings.json': canonical(request['layout']), 'attempt-00/candidate.py': source}
        for name, raw in values.items():
            write_once(donor / name, raw)
        proof = bound(dict(evidence=str(donor), source_sha256=sha(source), request_id=request['request_id'],
            candidate_preserved=True, compiler_input_bound=True, real_compilation_observed=True,
            real_encrypted_execution_observed=False,
            files={name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in values.items()}))
        p = root / 'proof.json'
        write_once(p, canonical(proof))
        return dict(path=str(p), sha256=sha(p.read_bytes())), request, source, values

    def test_exact_opaque_bytes_and_fresh_install_without_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor, request, source, values = self.fixture(root)
            before = copy.deepcopy(request)
            # Opaque non-NPZ bytes deliberately prove that intake cannot deserialize.
            with patch('subprocess.run', side_effect=AssertionError('No child permitted')):
                copied, receipt = adapter.retained_inputs(descriptor, request, source, evidence_root=root)
                target = root / 'fresh'
                target.mkdir()
                frozen = adapter.install_inputs(target, copied)
            self.assertEqual(request, before)
            self.assertEqual(set(frozen), set(adapter.INPUTS))
            self.assertEqual({name: (target / name).read_bytes() for name in adapter.INPUTS}, copied)
            self.assertEqual(copied, {name: values[name] for name in adapter.INPUTS})
            self.assertEqual(receipt['model_or_reference_forwards'], 0)
            self.assertEqual(receipt['array_deserializations'], 0)
            self.assertFalse(receipt['mathematical_or_CKKS_or_first_delivery_certified'])
            with self.assertRaises(ValueError):
                adapter.install_inputs(target, copied)

    def test_donor_or_request_change_refused_before_any_target_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptor, request, source, _ = self.fixture(root)
            wrong = copy.deepcopy(request)
            wrong['layout']['input_slot_period'] = request['layout']['input_slot_period'] + 1
            with self.assertRaises((ValueError, ProviderError)):
                adapter.retained_inputs(descriptor, wrong, source, evidence_root=root)
            p = root / 'donor/arrays.npz'
            p.chmod(0o600)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Retained input bytes changed'):
                adapter.retained_inputs(descriptor, request, source, evidence_root=root)
            self.assertFalse((root / 'fresh').exists())


if __name__ == '__main__':
    unittest.main()
