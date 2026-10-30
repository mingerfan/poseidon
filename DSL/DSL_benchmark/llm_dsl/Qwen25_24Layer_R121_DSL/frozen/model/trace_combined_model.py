"""Explicit opt-in 24-layer model adapter; --plan does not execute a model."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'package'))
from qwen24 import spec

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prefill','decode','decode-pair'),default='decode-pair')
    p.add_argument('--positions',type=int,nargs='+',default=[0,1])
    p.add_argument('--cached-positions',type=int,nargs='+',default=[])
    p.add_argument('--weights',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--rms-degree-strategy',choices=('degree10p2_pair13', 'degree10p2_pair13_silu_canonical_fold2', 'degree10p2_pair13_silu_pre_fold2'),default='degree10p2_pair13')
    p.add_argument('--linear-budget-mode',choices=('baseline','compact','compact_head','wide_bsgs','wide_head32','wide_only','wide_baby16','wide_baby8'),default='baseline')
    p.add_argument('--layout-fusion',action='store_true')
    p.add_argument('--early-input-mask',action='store_true')
    p.add_argument('--clean-kv-output',action='store_true')
    p.add_argument('--mlp-pair',action='store_true')
    p.add_argument('--pair-batch',choices=('none','both_except_last'),default='none')
    p.add_argument('--rms-batch-strategy',default='tiled16')
    p.add_argument('--lm-head-baby',type=int,choices=(16,32),default=32)
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    # combined_runtime imports Hecate; signature can instead be inspected from
    # the frozen interface.json without requiring a native frontend locally.
    import hecate as hc
    import combined_runtime as combined
    sig=combined.signature(spec.config(),a.mode,a.positions,a.cached_positions)
    if a.plan:
        print(json.dumps(sig,indent=2));return
    if a.weights is None or a.output_dir is None:
        p.error('--weights and --output-dir required')
    if a.output_dir.exists():
        p.error('Use a new output directory')
    model=combined.load_model(a.weights,rms_degree_strategy=a.rms_degree_strategy,linear_budget_mode=a.linear_budget_mode,layout_fusion=a.layout_fusion,pair_batch=a.pair_batch,lm_head_baby_count=a.lm_head_baby,rms_batch_strategy=a.rms_batch_strategy,early_input_mask=a.early_input_mask,clean_kv_output=a.clean_kv_output,mlp_pair=a.mlp_pair)
    sig['optimization']['linear_budget_mode']=a.linear_budget_mode
    sig['optimization']['rms_degree_strategy']=a.rms_degree_strategy
    combined.register(hc,model,sig)
    a.output_dir.mkdir(parents=True)
    print(hc.save(str(a.output_dir),str(a.output_dir),bytecode=True))
    (a.output_dir/'signature.json').write_text(json.dumps(sig,indent=2))

if __name__=='__main__':main()
