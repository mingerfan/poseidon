"""Small branched native-plan fixture using an already saved training vector.

Only metadata and float64 algebra are prepared here. The existing native adapter
executes the resulting plan separately, with one final decryption.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct

from .r10_paper_metrics import write_json
from .r31_lattigo_runtime_plan import content_filename, plan_inventory


def branch_plan(target):
    values=[];execution=[]
    def value(name,level,scale=43,components=2,kind='ciphertext'):
        values.append(dict(id=name,kind=kind,context='qwen-bootstrap48',level=level,
            scale_log2=scale,components=components,ntt=True,place=dict(kind='host',rank=0)))
    def op(name,kind,inputs,level,scale=43,components=2,**attrs):
        value(name,level,scale,components)
        execution.append(dict(kind='compute',ordinal=2+len(execution),op=kind,inputs=inputs,
            output=name,place=dict(kind='host',rank=0),**({'attrs':attrs} if attrs else {})))
    def drop(name,source):op(name,'mod_switch',[source],3,target_level=3)
    value('x',5);value('offset',4,components=1,kind='plaintext');value('half',4,components=1,kind='plaintext')
    op('r','rotate',['x'],5,steps=1)
    op('after_rotate','add_cc',['x','x'],5)
    op('q','mul_cc',['r','r'],5,86,3)
    op('after_mul_cc','rotate',['r'],5,steps=-1)
    op('ql','relinearize',['q'],5,86)
    op('after_relinearize','rescale',['q'],4,43,3,target_level=4,target_scale_log2=43)
    op('reused_quadratic','relinearize',['after_relinearize'],4)
    op('s','rescale',['ql'],4,target_level=4,target_scale_log2=43)
    op('after_rescale','rescale',['ql'],4,target_level=4,target_scale_log2=43)
    op('xd','mod_switch',['x'],4,target_level=4)
    op('after_drop','add_cc',['x','x'],5)
    op('a','add_cc',['s','xd'],4)
    op('after_add_cc','negate',['s'],4)
    op('cp','add_cp',['a','offset'],4)
    op('after_add_cp','negate',['a'],4)
    op('negative','negate',['cp'],4)
    op('after_negate','add_cc',['cp','cp'],4)
    op('product','mul_cp',['after_negate','half'],4,86)
    op('after_mul_cp','add_cc',['after_negate','after_negate'],4)
    op('scaled_product','rescale',['product'],3,target_level=3,target_scale_log2=43)
    leaves=['scaled_product']
    for name in ('negative','after_mul_cp','reused_quadratic','after_rescale',
                 'after_drop','after_add_cc','after_add_cp','after_mul_cc','after_rotate'):
        lowered='down_'+name;drop(lowered,name);leaves.append(lowered)
    # Count negative twice: the independently expanded final expression is
    # 3*roll(x,-1)^2 + 7*x + 3/8, with no exact cancelling output branch.
    leaves.insert(2,'down_negative')
    current=leaves[0]
    for index,right in enumerate(leaves[1:]):
        output='y' if index==len(leaves)-2 else 'sum'+str(index)
        op(output,'add_cc',[current,right],3);current=output
    initialization=[dict(kind='encode',ordinal=i,output=name,
        payload=dict(kind='bundle',content='ssa-fixture:'+name))
        for i,name in enumerate(('offset','half'))]
    return dict(format_version=1,plan_id=1,target=target,values=values,external_inputs=['x'],
        initialization=initialization,execution=execution,finalization=[],final_outputs=['y'],
        plaintext_bundle=dict(id='r31-branched-ssa-training-fixture',version=1))


def expected_values(numbers):
    if len(numbers)!=(1<<14) or not all(math.isfinite(v) for v in numbers):
        raise ValueError('Expected one finite original 16384-slot training vector')
    return [3*numbers[(i+1)%len(numbers)]**2+7*x+.375 for i,x in enumerate(numbers)]


def prepare(input_path,source_math,work):
    source=json.loads(Path(source_math).read_text())
    rows=source.get('samples',[])
    matching=[r for r in rows if r.get('sample_id')=='train:0' and
        Path(r.get('input','')).resolve()==Path(input_path).resolve()]
    if (source.get('status')!='ok' or source.get('mathematical_equivalence_verified') is not True
            or len(matching)!=1 or matching[0].get('status')!='ok'):
        raise ValueError('Original train:0 packed input provenance is missing')
    raw=Path(input_path).read_bytes()
    if len(raw)!=8*(1<<14):raise ValueError('Original packed vector length differs')
    numbers=list(struct.unpack('<16384d',raw));expected=expected_values(numbers)
    work=Path(work);work.mkdir(parents=True,exist_ok=False)
    bundle=work/'fixture.bundle';(bundle/'data').mkdir(parents=True)
    plan=branch_plan(dict(target_id='poseidon-ckks-cpu',adapter_probe='branched-ssa',
        source_math=str(Path(source_math).resolve()),source_input=str(Path(input_path).resolve())))
    blobs=[]
    for name,number in (('offset',.125),('half',.5)):
        content='ssa-fixture:'+name
        (bundle/'data'/content_filename(content)).write_bytes(struct.pack('<d',number)*(1<<14))
        blobs.append(dict(content=content,byte_length=8*(1<<14)))
    manifest=dict(bundle_format_version=1,version=1,bundle_id=plan['plaintext_bundle']['id'],blobs=blobs)
    inventory=plan_inventory(plan,manifest)
    write_json(work/'plan.json',plan);write_json(bundle/'manifest.json',manifest)
    write_json(work/'expected.json',dict(indices=list(range(1<<14)),values=expected,
        scope='Original saved normalized train:0 mux/gap1 vector; independent branched algebra '
              '3*roll(x,-1)^2+7*x+3/8; final comparison only, no model-quality selection'))
    result=dict(status='ok',plan=str(work/'plan.json'),bundle=str(bundle),input=str(Path(input_path).resolve()),
        expected=str(work/'expected.json'),inventory=inventory,source_math=str(Path(source_math).resolve()),
        source_sample_id='train:0',normalization_reapplied=False,scalar_multiplier=1,
        independent_expression='3*roll(x,-1)^2+7*x+3/8',
        reused_after_target={kind:next(op['ordinal'] for i,op in enumerate(plan['execution'])
            if op['op']==kind and any(set(op['inputs'])&set(later['inputs']) for later in plan['execution'][i+1:]))
            for kind in inventory['op_counts']},
        keygen_runs=0,encryptions=0,bootstraps=0,decryptions=0,extra_integrity_checks=0)
    write_json(work/'fixture-result.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('input','source-math','work-dir'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(argv);result=prepare(args.input,args.source_math,args.work_dir)
    print(json.dumps(dict(status=result['status'],execution_count=result['inventory']['execution_count'])));return 0


if __name__=='__main__':raise SystemExit(main())
