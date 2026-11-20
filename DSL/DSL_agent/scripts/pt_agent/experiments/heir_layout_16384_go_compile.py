"""Accepted original-fusion layout16384 target IR to the original offline Go build.

Only the new layout-native identity is accepted. Original host mathematics,
native conversion clocks and this fresh split-source/library-build clock remain
separate. Generated code is compiled without running initializers or models.
"""
import ast
from datetime import datetime, timezone
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from . import heir_no_fusion_go_compile as base
from . import heir_saved_math_layout_16384 as layout

MODULE = 'pt_agent.experiments.heir_layout_16384_go_compile'
ROUND = 'HEIR-ORIGINAL-FUSION-LAYOUT-16384-GO-COMPILE'
MODE = 'accepted-original-fusion-layout16384-target-IR-split-offline-Go-library'
common = base.common
host_sources = base.host_sources
paths = base.paths
# The original gate needs source/scan helpers from its original native adapter.
# New layout identity and its real source gate override those helper exports.
native = SimpleNamespace(**dict(vars(layout.base), **vars(layout)))


def _gate_tree():
    """Retarget the trusted gate's six pass/lineage bookkeeping sites."""
    tree = ast.parse(inspect.getsource(base.model_source_gate))
    constants = {'pass_removed':('pass_added',1), 'removed_count':('added_count',2),
        'linalg-fuse-linear-ops':(layout.LAYOUT_PASS,1)}
    counts = {name:0 for name in constants};helpers=0;lineage=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in constants:
            old=node.value;node.value=constants[old][0];counts[old]+=1
        if (isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name)
                and node.value.id=='native' and node.attr=='remove_linear_fusion'):
            node.attr='add_layout_optimization';helpers+=1
        if isinstance(node,ast.keyword) and node.arg=='no_fusion_source':
            node.arg='layout_16384_source';lineage+=1
    if counts!={name:value[1]for name,value in constants.items()} or helpers!=1 or lineage!=1:
        raise ValueError('The original source gate pass/lineage recording structure changed')
    return ast.fix_missing_locations(tree)


def model_source_gate(args):
    namespace=dict(vars(base),native=native)
    for name in ('_accepted_registry','_closed_native_processes'):
        namespace[name]=host_sources._clone(getattr(base,name),namespace)
    exec(compile(_gate_tree(), str(Path(__file__).resolve())+'::original-source-gate','exec'),namespace)
    checked=namespace['model_source_gate'](args)
    record=common.read(args.source_case/'result.json')
    request=record.get('native_request')or{}
    if (record.get('original_linear_fusion_retained')is not True
            or record.get('prior_default_layout_crash_risk_retained')is not True
            or set(request)!={'parameters','source_mlir','pipeline','pass_added','added_count','other_pass_order_unchanged'}):
        raise ValueError('The explicit original-fusion single-layout native request is required')
    observation=record['phases']['native_compile']['observation']
    if (any(type(observation.get(name))is not int for name in ('started_ns','ended_ns'))
            or observation['ended_ns']<=observation['started_ns']):
        raise ValueError('Ordered actual native conversion clock endpoints are required')
    lineage=checked['layout_16384_source']
    lineage.update(original_linear_fusion_retained=True,pass_added=layout.LAYOUT_PASS,added_count=1,
        prior_default_layout_crash_risk_retained=True,
        source_native_process_seconds=(observation['ended_ns']-observation['started_ns'])/1e9,
        native_target_IR_verified=True,model_specific_native_codegen_verified=False)
    return checked


def source_gate(args):
    return dict(model=model_source_gate(args),backend=common.backend_gate(args))


def durable_json(path,result):
    if Path(path).name=='result.json'and result.get('round')==ROUND:
        result['timing_variant']='fresh-Go-only-from-accepted-original-fusion-layout16384-native'
        model=(result.get('source')or{}).get('model')or{}
        lineage=model.get('layout_16384_source')or{}
        result['inherited_native_target_IR_verified']=lineage.get('native_target_IR_verified')is True
        result['inherited_native_process_seconds']=lineage.get('source_native_process_seconds')
        result['inherited_native_case_seconds']=lineage.get('complete_case_seconds')
        result['code_generation_verified']=bool(result.get('status')=='ok'
            and result.get('go_library_compile_accepted')is True
            and result.get('library_build_passed')is True and result.get('monitor_stopped')is True)
        result['model_specific_native_codegen_verified']=result['code_generation_verified']
        result['previous_native_clock_added']=False
        result['scope']='New accepted layout16384 target IR followed by split Go source emission and offline library build; all generated initializers and model functions remain unexecuted'
    common.durable_json(path,result)


def append_registry(out,result,registry):
    keys=('status','attempted','completed','go_sources_generated','library_build_passed',
        'go_library_compile_accepted','monitor_stopped','static_native_bootstraps',
        'inherited_native_target_IR_verified','code_generation_verified','model_specific_native_codegen_verified',
        *common.FORMAL_FIELDS)
    row=dict(run_id=out.name,round=ROUND,model=result['model'],mode=MODE,git_rev=result['git_rev'],
        metrics={key:result[key]for key in keys},artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with Path(registry).open('a+')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):
            raise ValueError('The new layout Go followup requires one eight-field registry row')
        stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def _namespace(args):
    # Clone only the original ordinary namespace factory. Its monitor/terminal
    # protocol and original contextmanager objects stay intact.
    proxy=SimpleNamespace(**dict(vars(common),durable_json=durable_json))
    globals_=dict(vars(base),MODULE=MODULE,ROUND=ROUND,MODE=MODE,native=native,common=proxy,
        source_gate=source_gate,append_registry=append_registry,__name__=__name__,__file__=__file__)
    namespace=host_sources._clone(base._namespace,globals_)(args)
    original_prepare=namespace['prepare']
    def prepare(out,checked):
        original_prepare(out,checked)
        listing=common.read(out/'source/snapshots.json');seen={row['source']for row in listing['files']}
        for module in (base,layout.base):
            source=Path(module.__file__).resolve()
            if str(source)in seen:continue
            seen.add(str(source));target=out/'source'/source.name;shutil.copyfile(source,target)
            listing['files'].append(dict(source=str(source),snapshot=str(target),size_bytes=target.stat().st_size))
        common.durable_json(out/'source/snapshots.json',listing)
    namespace['prepare']=prepare
    return namespace


def execute(args):
    return _namespace(args)['execute'](args)


def parser():
    command=base.parser();command.description=__doc__
    return command


def main(argv=None):
    args=parser().parse_args(argv)
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().resolve())
    result=execute(args)
    print(json.dumps({key:result.get(key)for key in
        ('status','error','inherited_native_target_IR_verified','library_build_passed','code_generation_verified')}))
    return int(result['status']not in('prepared','ok'))


if __name__=='__main__':raise SystemExit(main())
