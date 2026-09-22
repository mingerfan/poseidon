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
RELU = ROOT / "Doc/analysis/resnet20_joint_chain_20260911/relu_precision.cpp"
GPU_EVALUATOR = ROOT / "src/poseidon/gpu/gpu_evaluator.cpp"


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

    def test_conv_batches_preserve_inventory_and_only_check_offline(self):
        network = NETWORK.read_text()
        adapter = (ROOT / "Doc/analysis/resnet20_joint_chain_20260911/conv_probe_runtime.h").read_text()
        self.assertIn("POSEIDON_CONV_PLAIN_BATCH", network)
        self.assertIn("total.terms!=(conv_plain_batch?1440:0)", network)
        self.assertIn("total.calls!=(conv_plain_batch?480:0)", network)
        self.assertIn("conv_plain_batch && preparation?566231040:0", network)
        self.assertIn("batch.sources.clear();batch.plains.clear();", adapter)
        self.assertIn("prepared_mode==PreparedMode::capture", adapter)
        self.assertIn("m.ctx,*destination.gpu,reference", adapter)
        self.assertIn("destination.pending->sources.size()==4", adapter)
        self.assertIn("plain_batch_stats.pending_terms", adapter)

    def test_bootstrap_uses_fail_closed_exact_qp_plaintext_compression(self):
        text = NETWORK.read_text()
        self.assertEqual(text.count(
            "s.metadata.ctx,0,1,true);"), 2)
        self.assertIn("BOOTSTRAP_QP_COMPRESSION enabled=true exact=true", text)
        self.assertIn("plain.exact_device_reconstruction", text)
        self.assertIn('compressed_qp_plaintexts=true', text)

    def test_application_defaults_to_full_baby_c2s_tiles(self):
        text = NETWORK.read_text()
        self.assertIn("default_c2s_baby_tile=8", text)
        self.assertIn("cts_workspace.baby_tile_size=default_c2s_baby_tile", text)
        self.assertIn("continuous default C2S full-baby tile contract changed", text)
        self.assertIn('bootstrap_c2s_baby_tile=', text)

    def test_relu_owned_moddrops_use_zero_copy_q_prefixes(self):
        relu = RELU.read_text()
        evaluator = GPU_EVALUATOR.read_text()
        self.assertIn("POSEIDON_RELU_ZERO_COPY_MODDROP", relu)
        self.assertIn("drop_owned(std::move(quotient)", relu)
        self.assertIn("drop_owned(std::move(remainder)", relu)
        self.assertIn("basis.emplace(1,drop_owned(std::move(input)", relu)
        start = evaluator.index("void GpuEvaluator::drop_modulus_inplace")
        end = evaluator.index("void GpuEvaluator::multiply_scalar", start)
        inplace = evaluator[start:end]
        self.assertIn("poly.shards.front().limb_count", inplace)
        self.assertNotIn("launch_copy_poly_shard", inplace)

    def test_relu_shared_operands_use_read_only_q_prefix_views(self):
        relu = RELU.read_text()
        evaluator = GPU_EVALUATOR.read_text()
        self.assertIn("POSEIDON_RELU_Q_PREFIX_VIEWS", relu)
        self.assertIn("mul_q_prefix(\n                basis.at(left),basis.at(right)", relu)
        self.assertIn("multiply_plain_q_prefix(term,n.work", relu)
        self.assertIn("eval.multiply_q_prefix(a,b,raw", relu)
        self.assertIn("GpuEvaluator::multiply_q_prefix", evaluator)
        self.assertIn("GpuEvaluator::multiply_plain_q_prefix", evaluator)
        self.assertIn("make_q_prefix_view", evaluator)
        self.assertIn("prefix_source_views!=1463", NETWORK.read_text())

    def test_relu_leaf_fusion_preserves_terms_and_checks_exact_residues(self):
        relu = RELU.read_text()
        start = relu.index("CT fused_leaf(const Node &n)")
        leaf = relu[start:relu.index("CT node(", start)]
        self.assertIn("POSEIDON_RELU_LEAF_FUSION", relu)
        self.assertIn("i<n.coefficients.size()", leaf)
        self.assertIn("basis.at(2*i+1)", leaf)
        self.assertIn("std::exp2(n.pre)/term.meta.scale", leaf)
        self.assertIn("if(observe)", leaf)
        self.assertIn("require_exact_ciphertexts", leaf)
        self.assertNotIn("continue;", leaf)
        self.assertNotIn("rescale(", leaf)
        self.assertIn("Replay baseline(ctx,encoder,decryptor,parameters,keys,false,false)", relu)
        self.assertIn("moddrop_stats.exact_leaf_checks!=0", NETWORK.read_text())

    def test_relu_basis_fusion_does_not_move_rescale_or_relinearization(self):
        relu = RELU.read_text()
        begin = relu.index("CT fused_basis_correction(")
        correction = relu[begin:relu.index("CT component(", begin)]
        self.assertIn("scalar(1,product,plain_scale)", correction)
        self.assertIn("eval.double_sub_plain_q_prefix(product,plain,out,correction)", correction)
        self.assertIn("if(observe || basis_preparation_checks)", correction)
        self.assertIn("require_exact_ciphertexts", correction)
        self.assertNotIn("eval.rescale", correction)
        self.assertNotIn("eval.relinearize", correction)
        network = NETWORK.read_text()
        self.assertIn("enabled?285:0", network)
        self.assertIn("enabled && preparation?824311808:0", network)
        self.assertLess(network.index("set_basis_preparation_checks(false)"),
                        network.index("activity.begin_continuous()"))


if __name__ == "__main__":
    unittest.main()
