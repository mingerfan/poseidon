"""Offline regression checks for the explicitly approved HF loading boundaries."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from pt_agent.intake.hf_frontend import load_local


class LocalHFLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'config.json').write_text('{}')
        self.torch = MagicMock()
        self.transformers = MagicMock()
        self.config = SimpleNamespace(architectures=['BertForSequenceClassification'])
        self.transformers.AutoConfig.from_pretrained.return_value = self.config
        self.module = MagicMock()
        self.module.double.return_value = self.module
        self.module.eval.return_value = self.module
        self.factory = self.transformers.AutoModelForSequenceClassification
        self.factory.from_config.return_value = self.module
        self.factory.from_pretrained.return_value = self.module
        self.modules = patch.dict(sys.modules, torch=self.torch, transformers=self.transformers)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_binary_only_uses_restricted_torch_loader(self):
        (self.path / 'pytorch_model.bin').touch()
        state = {'layer.weight': object()}
        self.torch.load.return_value = state
        self.torch.is_tensor.return_value = True
        result = load_local(self.path)
        self.torch.load.assert_called_once_with(self.path / 'pytorch_model.bin',
                                                map_location='cpu', weights_only=True)
        self.factory.from_pretrained.assert_called_once_with(str(self.path), config=self.config,
            state_dict=state, use_safetensors=False, trust_remote_code=False, local_files_only=True)
        self.factory.from_config.assert_not_called()
        self.assertIs(result.module, self.module)

    def test_safetensors_does_not_fall_back_to_bin(self):
        (self.path / 'model.safetensors').touch()
        (self.path / 'pytorch_model.bin').touch()
        load_local(self.path)
        self.factory.from_pretrained.assert_called_once_with(str(self.path), config=self.config,
            use_safetensors=True, trust_remote_code=False, local_files_only=True)
        self.torch.load.assert_not_called()

    def test_config_and_tokenizer_require_local_files_and_no_remote_code(self):
        (self.path / 'model.safetensors').touch()
        load_local(self.path)
        flags = dict(trust_remote_code=False, local_files_only=True)
        self.transformers.AutoConfig.from_pretrained.assert_called_once_with(str(self.path), **flags)
        self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(str(self.path), **flags)

    def test_restricted_loader_error_has_no_unsafe_retry(self):
        (self.path / 'pytorch_model.bin').touch()
        self.torch.load.side_effect = RuntimeError('restricted loader rejected checkpoint')
        with self.assertRaisesRegex(RuntimeError, 'restricted loader rejected'):
            load_local(self.path)
        self.assertEqual(self.torch.load.call_count, 1)
        self.factory.from_pretrained.assert_not_called()
        self.factory.from_config.assert_not_called()

    def test_non_tensor_checkpoint_is_rejected(self):
        (self.path / 'pytorch_model.bin').touch()
        self.torch.load.return_value = {'object': 'not a tensor'}
        self.torch.is_tensor.return_value = False
        with self.assertRaisesRegex(ValueError, 'tensor state_dict'):
            load_local(self.path)
        self.factory.from_config.assert_not_called()
        self.factory.from_pretrained.assert_not_called()


if __name__ == '__main__':
    unittest.main()
