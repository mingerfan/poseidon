import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRY = ROOT / "bench/resnet20_s2c_first/main.cpp"
BUILD = ROOT / "bench/resnet20_s2c_first/build.sh"
PLAIN_REFERENCE = ROOT / "bench/resnet20_s2c_first/plain_resnet20_reference.cpp"
ACTIVITY_TIMING = ROOT / "bench/resnet20_s2c_first/gpu_activity_timing.cpp"
NETWORK = ROOT / "Doc/analysis/resnet20_joint_chain_20260911/bootstrap_network_precision.cpp"
BASELINE = ROOT / "Doc/analysis/resnet20_joint_chain_20260911/bootstrap_baseline_precision.cpp"


class DedicatedApplicationIsolationTest(unittest.TestCase):
    def test_entry_selects_only_the_s2c_first_application(self):
        text = ENTRY.read_text()
        self.assertIn("run_s2c_first_resnet20_application", text)
        self.assertNotIn("GpuCkksRuntime::bootstrap", text)
        includes = re.findall(r'^\s*#include\s+[<\"]([^>\"]+)', text, re.MULTILINE)
        self.assertNotIn("gpu_resnet20_inference.cpp", includes)
        self.assertNotIn("gpu_ckks_runtime.cpp", includes)
        self.assertNotIn("gpu_relu.cpp", includes)

    def test_build_does_not_link_legacy_scheduler_sources(self):
        text = BUILD.read_text()
        for forbidden in (
            "gpu_ckks_runtime.cpp",
            "gpu_resnet20_inference.cpp",
            "gpu_relu.cpp",
            "main.cpp\"",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("bench/resnet20_s2c_first/gpu_activity_timing.cpp", text)
        self.assertIn("-lcupti", text)

    def test_plain_reference_does_not_include_legacy_inference(self):
        text = PLAIN_REFERENCE.read_text()
        self.assertNotIn("gpu_resnet20_inference", text)
        self.assertNotIn("GpuCkksRuntime", text)
        self.assertNotIn("gpu_relu.cpp", text)

    def test_strict_activity_collector_is_local_and_fail_closed(self):
        text = ACTIVITY_TIMING.read_text()
        self.assertIn("CUPTI dropped records", text)
        self.assertIn("included host/device transfers", text)
        self.assertIn("has no measured-stage attribution", text)
        self.assertNotIn("gpu_ckks_runtime", text)
        self.assertNotIn("gpu_resnet20_inference", text)

    def test_default_performance_path_has_one_resident_online_boundary(self):
        text = NETWORK.read_text()
        self.assertIn('mode=="--unsafe-performance-only"', text)
        self.assertNotIn("--unsafe-continuous-performance-only", text)
        self.assertNotIn("class TimingNetwork", text)
        self.assertNotIn("class MeasuredGpuStages", text)
        self.assertIn("activity.begin_continuous()", text)
        self.assertNotIn('activity.stage("resnet20"', text)
        for category in ("bootstrap.S2C", "bootstrap.C2S", "bootstrap.EvalMod",
                         "relu", "conv_bn", "residual.add", "head"):
            self.assertIn(f'"{category}"', text)
        self.assertIn("GPU_ACTIVITY_BREAKDOWN category=", text)
        self.assertIn('mode=single_resident_online_inference', text)
        self.assertIn('per_stage_sync=false observer_decryptions=0', text)
        self.assertIn('post_timing_decryptions=1 accuracy=', text)
        self.assertIn('POST_TIMING_ACCURACY result=', text)
        self.assertIn('intermediate_reencryptions=0 swaps=false', text)
        self.assertIn('conv_plaintexts!=1696 || shortcut_plaintexts!=48', text)

    def test_stem_starts_at_q32_and_naturally_outputs_q31(self):
        baseline = BASELINE.read_text()
        network = NETWORK.read_text()
        self.assertIn('parms_id_map().at(31)', baseline)
        self.assertIn('encrypted.coeff_modulus_size()!=32', baseline)
        self.assertIn('stem.packs[0].meta.q_count!=31', network)
        self.assertIn('input_q=32 stem_output_q=31', network)
        self.assertNotIn('"stem.drop_q31"', network)


if __name__ == "__main__":
    unittest.main()
