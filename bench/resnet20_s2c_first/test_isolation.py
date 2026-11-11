import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRY = ROOT / "bench/resnet20_s2c_first/main.cpp"
BUILD = ROOT / "bench/resnet20_s2c_first/build.sh"
CMAKE = ROOT / "bench/resnet20_s2c_first/CMakeLists.txt"
PLAIN_REFERENCE = ROOT / "bench/resnet20_s2c_first/plain_resnet20_reference.cpp"
ACTIVITY_TIMING = ROOT / "bench/resnet20_s2c_first/gpu_activity_timing.cpp"
NETWORK = ROOT / "Doc/analysis/resnet20_joint_chain_20260911/bootstrap_network_precision.cpp"
BASELINE = ROOT / "Doc/analysis/resnet20_joint_chain_20260911/bootstrap_baseline_precision.cpp"
RELU = ROOT / "Doc/analysis/resnet20_joint_chain_20260911/relu_precision.cpp"
GPU_EVALUATOR = ROOT / "src/poseidon/gpu/gpu_evaluator.cpp"
TRACE_FACADE = ROOT / "bench/resnet20_s2c_first/traced_gpu_evaluator.h"
TRACE_RECORDER = ROOT / "bench/resnet20_s2c_first/gpu_api_trace.h"
CONV_ADAPTER = ROOT / "Doc/analysis/resnet20_joint_chain_20260911/conv_probe_runtime.h"


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
        text = BUILD.read_text() + CMAKE.read_text()
        for forbidden in (
            "gpu_ckks_runtime.cpp",
            "gpu_resnet20_inference.cpp",
            "gpu_relu.cpp",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("gpu_activity_timing.cpp", text)
        self.assertIn("CUDA::cupti", text)

    def test_application_uses_local_sources_and_rebuildable_cmake_target(self):
        build = BUILD.read_text()
        cmake = CMAKE.read_text()
        adapter = (ROOT / "Doc/analysis/resnet20_joint_chain_20260911/conv_probe_runtime.h").read_text()
        for text in (build, cmake, adapter):
            self.assertNotIn("/home/liufuyao", text)
            self.assertNotIn("poseidon_gpu_other", text)
        self.assertNotIn("*.cpp.o", build)
        self.assertIn("--target poseidon_resnet20_s2c_first", build)
        self.assertIn("output=${1:-/tmp/poseidon_resnet20_s2c_first}", build)
        self.assertIn("support/resnet20_weights.cpp", cmake)
        self.assertIn("support/gpu_multiplexed_tensor.cpp", adapter)
        self.assertNotIn('#include "gpu_ckks_runtime.h"', adapter)

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
        self.assertIn("std::optional<poseidon::benchmark::s2c_first::GpuActivityTiming> activity_storage;", text)
        self.assertIn("if(cupti)activity_storage.emplace();", text)
        self.assertIn("if(cupti)activity_storage->begin_continuous();", text)
        self.assertIn("execute_graph(false,cupti?&*activity_storage:nullptr)", text)
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
        self.assertIn('expected_conv_plaintexts=1440+18+(conv_merged_masks?672:160+78)', text)
        self.assertIn('conv_plaintexts!=expected_conv_plaintexts || shortcut_plaintexts!=48', text)
        self.assertIn('head_runtime->prepared_plaintext_count()!=93', text)

    def test_level_layout_and_stem_follow_selected_chain(self):
        baseline = BASELINE.read_text()
        network = NETWORK.read_text()
        relu = RELU.read_text()
        configure = network[network.index('void configure_resnet20_levels('):
                            network.index('std::string resnet20_levels_summary()')]
        self.assertIn('int entry_q = 31;', relu)
        self.assertIn('int removed_q_prime = -1;', relu)
        self.assertIn('conv_merged_masks=setting("POSEIDON_S2C_CONV_MERGED_MASK").value_or(true)', configure)
        self.assertIn('const auto requested=setting("POSEIDON_S2C_SHORT_CHAIN")', configure)
        self.assertIn('requested.value_or(conv_merged_masks && library_boot && !accuracy)', configure)
        for guard in ('if(!conv_merged_masks)', 'if(!library_boot)', 'if(accuracy)'):
            self.assertIn(guard, configure)
        self.assertIn('removed_q_prime=8;\n        entry_q=30;', configure)
        self.assertIn('if(removed_q_prime<0)return;', relu)
        self.assertIn('q.erase(q.begin()+removed_q_prime);', relu)
        self.assertIn('apply_removed_q_prime(q);\n        p.set_modulus(q,aux);', baseline)
        fixture = (ROOT / 'Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt').read_text().split()
        self.assertEqual(fixture[:3], ['RELU_PRECISION_V1', '50', '25'])
        q = list(map(int, fixture[3:53]))
        self.assertEqual(q[8], 4267442177)
        short_q = q[:8] + q[9:]
        self.assertEqual(len(short_q), 49)
        self.assertEqual(short_q[:8], q[:8])
        self.assertEqual(short_q[8:], q[9:])
        self.assertEqual(short_q[29], q[30])  # Q30 and historical Q31 share a physical prime.
        self.assertEqual(short_q[30], q[31])  # Likewise input Q31 and Q32.
        self.assertIn('int relu_output_q() {return entry_q-22;}', relu)
        self.assertIn('int conv_input_q() {return 6+(conv_merged_masks?2:3);}', relu)
        self.assertIn('parms_id_map().at(entry_q)', baseline)
        self.assertIn('encrypted.coeff_modulus_size()!=static_cast<std::size_t>(entry_q+1)', baseline)
        self.assertIn('stem.packs[0].meta.q_count!=static_cast<std::size_t>(entry_q)', network)
        self.assertIn('refreshed.meta.q_count!=static_cast<std::size_t>(entry_q)', network)
        self.assertIn('output.meta.q_count!=static_cast<std::size_t>(relu_output_q())', network)
        self.assertIn('input.packs[0]=runtime.drop_to_q_count(input.packs[0],conv_q)', network)
        self.assertIn('output.packs[0].meta.q_count!=6', network)
        self.assertIn('" input_q="+std::to_string(entry_q+1)', network)
        self.assertIn('" stem_output_q="+std::to_string(entry_q)', network)
        self.assertNotIn('"stem.drop_q31"', network)

    def test_conv_batches_preserve_inventory_and_only_check_offline(self):
        network = NETWORK.read_text()
        adapter = (ROOT / "Doc/analysis/resnet20_joint_chain_20260911/conv_probe_runtime.h").read_text()
        self.assertIn("POSEIDON_CONV_PLAIN_BATCH", network)
        self.assertIn("total.terms!=(conv_plain_batch?1440:0)", network)
        self.assertIn("total.calls!=(conv_plain_batch?480:0)", network)
        self.assertIn("conv_residues=std::size_t(480)*2*65536*conv_input_q()", network)
        self.assertIn("total.chains!=(conv_plain_batch?160:0)", network)
        self.assertIn("total.exact_checks!=(conv_plain_batch && preparation?480:0)", network)
        self.assertIn("total.exact_residues!=(conv_plain_batch && preparation?conv_residues:0)", network)
        self.assertIn("const auto level=static_cast<std::size_t>(conv_input_q())", adapter)
        self.assertIn("x.meta.q_count==level", adapter)
        self.assertIn("scale==double(m.ctx.parameters_literal()->q().at(level-1).value())", adapter)
        self.assertIn("trace_gpu(*eval).multiply_plain_sum_q_prefix(sources,batch.plains,*destination.gpu,accumulate)", adapter)
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
        self.assertIn("trace_gpu(eval).multiply_q_prefix(a,b,raw", relu)
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
        self.assertIn("trace_gpu(eval).double_sub_plain_q_prefix(product,plain,out,correction)", correction)
        self.assertIn("if(observe || basis_preparation_checks)", correction)
        self.assertIn("require_exact_ciphertexts", correction)
        self.assertNotIn("eval.rescale", correction)
        self.assertNotIn("eval.relinearize", correction)
        network = NETWORK.read_text()
        self.assertIn("enabled?285:0", network)
        self.assertIn("basis_residues=824311808-\n            std::size_t(31-entry_q)*285*2*65536", network)
        self.assertIn("stats.fused_basis_constants!=(enabled?190:0)", network)
        self.assertIn("stats.fused_basis_products!=(enabled?95:0)", network)
        self.assertIn("stats.exact_basis_checks!=(enabled && preparation?285:0)", network)
        self.assertIn("stats.exact_basis_residues!=(enabled && preparation?basis_residues:0)", network)
        self.assertLess(network.index("set_basis_preparation_checks(false)"),
                        network.index("activity_storage->begin_continuous()"))

    def test_trace_intercepts_real_split_hoist_and_native_boot(self):
        facade = TRACE_FACADE.read_text()
        adapter = CONV_ADAPTER.read_text()
        network = NETWORK.read_text()
        recorder = TRACE_RECORDER.read_text()
        self.assertIn('trace_gpu(*backend.evaluator).rotate_many_hoisted(*x.gpu,direct_steps,', adapter)
        self.assertIn('if(!direct_steps.empty())', adapter)
        self.assertIn('if(!step) {\n                result[index]=drop_to_q_count(x,x.meta.q_count);\n                continue;', adapter)
        self.assertIn('runtime.rotate_many_composed(', (ROOT / 'bench/resnet20_s2c_first/support/gpu_multiplexed_tensor.cpp').read_text())
        begin = facade.index('    void rotate_many_hoisted(')
        hoist = facade[begin:facade.index('    void bootstrap(', begin)]
        self.assertIn('if (!recorder) { e->rotate_many_hoisted(source, steps, keys, workspace, out); return; }', hoist)
        self.assertLess(hoist.index('if (steps.empty()) return;'), hoist.index('e->rotation_hoist(source, keys, hoisted, workspace);'))
        self.assertIn('".step" + std::to_string(step)', hoist)
        self.assertIn('e->rotation_hoist(source, keys, hoisted, workspace);', hoist)
        self.assertIn('recorder->record_buffer("rotation_hoist", prefix_callee, {source_id}, hoisted,', hoist)
        self.assertIn('for (std::size_t i = 0; i < steps.size(); ++i)', hoist)
        self.assertIn('e->rotate_from_hoist(source, hoisted, step, keys, workspace, out[i]);', hoist)
        self.assertIn('recorder->record_ct("rotate_from_hoist", callee, {source_id, hoist_id}, out[i],', hoist)
        self.assertIn('e->rotation_hoist(*v[0], *k, result, *w);', hoist)
        self.assertIn('e->rotate_from_hoist(*v[0], v[1].hoisted(), step, *k, *w, result);', hoist)
        begin = facade.index('    void bootstrap(')
        boot = facade[begin:facade.index('    // Old diagnostic/', begin)]
        self.assertIn('capture("bootstrap", {&a}', boot)
        self.assertIn('e->bootstrap(*v[0], *d, *r, *g, *w, result)', boot)
        self.assertIn('e->bootstrap(a, data, relin, galois, workspace, out)', boot)
        self.assertIn('trace_gpu(eval).bootstrap(encrypted_input,library->bootstrap_data,', network)
        self.assertIn('if(!bootstrap->library_boot())', network)
        self.assertIn('if (node.opcode == "bootstrap") ++boots_;', recorder)
        self.assertIn('if (node.opcode == "rotation_hoist") ++hoists_;', recorder)
        self.assertIn('if (node.opcode == "rotate_from_hoist") ++suffixes_;', recorder)
        self.assertIn('api.boot_calls()!=18 || api.hoist_calls()==0 || api.rotation_suffix_calls()<=api.hoist_calls()', network)
        self.assertIn('api.live_hoist_bytes()!=0 || api.hoist_releases()!=api.hoist_calls()', network)

    def test_trace_records_metadata_only_scale_reconciliation(self):
        relu = RELU.read_text()
        facade = TRACE_FACADE.read_text()
        self.assertIn('scale_check(b, std::log2(a.meta.scale)); set_trace_scale(eval,b,a.meta.scale);', relu)
        begin = facade.index('    void set_scale(')
        scale = facade[begin:facade.index('    void multiply_plain_accumulate(', begin)]
        self.assertIn('if (!std::isfinite(scale) || scale <= 0)', scale)
        self.assertIn('capture("set_scale", {&a}, {}, ".bits" + api_trace_detail::scalar_signature(scale), a,', scale)
        self.assertIn('result.meta.scale = scale; return Value(std::move(result));', scale)
        self.assertIn('[&] { a.meta.scale = scale; }, true);', scale)
        self.assertNotIn('e->', scale)
        self.assertIn('trace_gpu(eval).set_scale(value, scale);', facade)
        self.assertIn('value.meta.scale = scale;', facade)
        recorder = TRACE_RECORDER.read_text()
        self.assertIn('if (!api_trace_detail::same_meta(ct.meta, found->second.meta))', recorder)
        self.assertIn('unrecorded API trace ciphertext metadata change', recorder)

    def test_trace_capture_excludes_preparation_and_reference(self):
        network = NETWORK.read_text()
        facade = TRACE_FACADE.read_text()
        recorder = TRACE_RECORDER.read_text()
        begin = network.index('        if(!bootstrap->library_boot())')
        end = network.index('        gpu_check_cuda(cudaDeviceSynchronize(),"API trace forward completion");', begin)
        trace = network[begin:end]
        ordered = ('s.replay.set_basis_preparation_checks(true);', 'auto warm=execute_graph(true);',
                   's.replay.set_basis_preparation_checks(false);', 'auto reference=execute_graph(false);',
                   'ApiTraceRecorder recorder(s.metadata.ctx,s.eval);',
                   'recorder.register_input(input);', 'ScopedApiTrace active(recorder);',
                   'traced=execute_graph(false);', 'recorder.finish(traced);')
        positions = [trace.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('inline thread_local ApiTraceRecorder *active_api_trace_recorder = nullptr;', recorder)
        self.assertIn('if (active_api_trace_recorder) throw std::runtime_error("nested API tracing is unsupported");', recorder)
        self.assertIn('if (recorder.finished()) throw std::runtime_error("cannot activate a finished API trace");', recorder)
        self.assertIn('active_api_trace_recorder = &recorder;', recorder)
        self.assertIn('~ScopedApiTrace() { active_api_trace_recorder = nullptr; }', recorder)
        self.assertIn('if (!recorder) { run(); return; }', facade)
        self.assertLess(facade.index('auto dependencies = recorder->inputs(sources)'),
                        facade.index('recorder->record_ct(opcode, std::move(callee)'))
        self.assertIn('require_exact_ciphertexts(s.metadata.ctx,reference,traced,"manual_vs_API_trace")', network)
        self.assertIn('require_exact_ciphertexts(s.metadata.ctx,reference,logits,"manual_vs_traced_RuntimePlan")', network)


if __name__ == "__main__":
    unittest.main()
