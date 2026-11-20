"""New callback validation delta; no real model or original tests."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_component_cost as entry


class ScopedLoweringShapeChecks(unittest.TestCase):
    def test_original_weight_shape_rejection_and_callback_identity(self):
        from pt_agent.experiments import qwen_layered_structure as layered
        segment=SimpleNamespace(program=object(),weight_shapes=(('w',(3,)),))
        weights={'w':SimpleNamespace(shape=(4,))}
        callback=layered.SegmentLowering(segment,weights)
        adapted=entry.already_scoped_lowering(callback)
        self.assertIs(adapted.segment,segment)
        self.assertIs(adapted.weights,weights)
        with patch.object(layered,'source_binding',return_value={})as source:
            with self.assertRaisesRegex(ValueError,'The original weight shape changed before segment lowering'):
                adapted(object())
            source.assert_called_once_with(segment)
        with self.assertRaises(TypeError):entry.already_scoped_lowering(object())


if __name__=='__main__':unittest.main()
