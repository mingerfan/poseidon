import unittest

from bench.resnet20_s2c_first.summarize_performance import summarize


GOOD = """\
MEMORY pool_cap_GiB=30 pool_initial_GiB=30 startup_free_GiB=31.1
APPLICATION_KEY_PREFLIGHT old_global_rotation_MiB=8175 level_rotation_MiB=1469 level_relin_MiB=363 level_total_MiB=1832 active_parameter_ntt_MiB=862.75 resident=true swaps=false
APPLICATION_KEYS_READY rotation_levels=6 relin_levels=13 relin_contexts=11 parameter_levels=19 payload_MiB=1832 dnum=2 resident=true swaps=false
CONTINUOUS_PREPARED inputs=27 stem_plaintexts=28 conv_plaintexts=1696 shortcut_plaintexts=48 relu_plaintexts=46 head_plaintexts=93 free_MiB=1470.375
GPU_ACTIVITY_TIMING gpu_total_ms=8079.808307 activities=108241 host_transfers=0 host_transfer_bytes=0
GPU_ACTIVITY_BREAKDOWN category=bootstrap.C2S gpu_ms=2100 calls=18
GPU_ACTIVITY_BREAKDOWN category=bootstrap.EvalMod gpu_ms=2990 calls=36
GPU_ACTIVITY_BREAKDOWN category=bootstrap.ModRaise gpu_ms=8.8 calls=18
GPU_ACTIVITY_BREAKDOWN category=bootstrap.S2C gpu_ms=350 calls=18
GPU_ACTIVITY_BREAKDOWN category=bootstrap.prepare gpu_ms=0.2 calls=18
GPU_ACTIVITY_BREAKDOWN category=bootstrap.recombine gpu_ms=51 calls=18
GPU_ACTIVITY_BREAKDOWN category=conv_bn gpu_ms=1100 calls=18
GPU_ACTIVITY_BREAKDOWN category=head gpu_ms=30 calls=1
GPU_ACTIVITY_BREAKDOWN category=relu gpu_ms=1400 calls=19
GPU_ACTIVITY_BREAKDOWN category=residual.add gpu_ms=0.4 calls=9
GPU_ACTIVITY_BREAKDOWN category=residual.clone gpu_ms=0.2 calls=9
GPU_ACTIVITY_BREAKDOWN category=shortcut gpu_ms=45 calls=2
GPU_ACTIVITY_BREAKDOWN category=stem.conv_bn gpu_ms=4.208307 calls=1
POST_TIMING_ACCURACY result=PASS max_logit_error=0.012 max_logit_imag=0.0001 true_label=3 plain_prediction=3 gpu_prediction=3
CONTINUOUS_TIMING_RESULT mode=single_resident_online_inference wall_ms=8173.719444 cuda_event_ms=8173.80419921875 gpu_activity_union_ms=8079.808307 host_enqueue_ms=8138.745191 activities=108241 host_transfers=0 blocks=9 bootstraps=18 dnum=2 input_q=32 stem_output_q=31 application_keyswitch=fixed_dnum2 hoisted_rotation=true input_upload_included=false public_material_upload_included=false per_stage_sync=false observer_decryptions=0 post_timing_decryptions=1 accuracy=PASS max_logit_error=0.012 intermediate_reencryptions=0 swaps=false encrypted_logits_q=4 security_approved=false
"""


class PerformanceSummaryTest(unittest.TestCase):
    def test_accepts_complete_contract(self):
        summary = summarize(GOOD)
        self.assertAlmostEqual(summary["wall_ms"], 8173.719444)
        self.assertAlmostEqual(summary["conv_bn_gpu_ms"], 1104.208307)
        self.assertAlmostEqual(summary["relu_gpu_ms"], 1400)
        self.assertAlmostEqual(summary["bootstrap_gpu_ms"], 5500)
        self.assertAlmostEqual(summary["shortcut_gpu_ms"], 45)
        self.assertAlmostEqual(summary["pool_fc_gpu_ms"], 30)
        self.assertAlmostEqual(summary["residual_copy_gpu_ms"], 0.6)
        self.assertAlmostEqual(summary["category_sum_ms"], 8079.808307)
        self.assertAlmostEqual(summary["max_logit_error"], 0.012)
        self.assertEqual(summary["gpu_prediction"], 3)

    def test_rejects_transfer_or_fragmentation(self):
        with self.assertRaises(ValueError):
            summarize(GOOD.replace("host_transfers=0", "host_transfers=1", 1))
        with self.assertRaises(ValueError):
            summarize(GOOD + "GPU_ACTIVITY_STAGE stage=conv gpu_ms=1\n")

    def test_rejects_missing_inventory(self):
        with self.assertRaises(ValueError):
            summarize(GOOD.replace("conv_plaintexts=1696", "conv_plaintexts=1695"))

    def test_accepts_small_event_wall_instrumentation_offset(self):
        summary = summarize(GOOD.replace(
            "cuda_event_ms=8173.80419921875", "cuda_event_ms=8220"))
        self.assertAlmostEqual(summary["cuda_event_ms"], 8220)

    def test_rejects_large_event_wall_mismatch_or_dense_parameter_plan(self):
        with self.assertRaises(ValueError):
            summarize(GOOD.replace(
                "cuda_event_ms=8173.80419921875", "cuda_event_ms=8300"))
        with self.assertRaises(ValueError):
            summarize(GOOD.replace("parameter_levels=19", "parameter_levels=20"))

    def test_rejects_missing_breakdown_or_wrong_call_count(self):
        with self.assertRaises(ValueError):
            summarize(GOOD.replace(
                "GPU_ACTIVITY_BREAKDOWN category=head gpu_ms=30 calls=1\n", ""))
        with self.assertRaises(ValueError):
            summarize(GOOD.replace("category=relu gpu_ms=1400 calls=19",
                                   "category=relu gpu_ms=1400 calls=18"))


if __name__ == "__main__":
    unittest.main()
