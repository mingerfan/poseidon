"""Finite U3a acceptance, with result ownership kept in the outer process."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

from ..paths import (RESULTS_DIR, REPO, WORK_ROOT, PLATFORM_DIR, DACAPO_ROOT,
                     DACAPO_UPSTREAM, HECATE_BUILD)

BASE = '5412df2b'
FROZEN = 'r8e7-reconfirm-20261022T122558'
U2 = 'u2-models-20261024T125911245307/COMPARISON.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def equal(left, right, description):
    if left != right:
        raise AssertionError(description)


def variants(args):
    return json.loads(args.u2.read_text())['variants']


def program_path(row):
    for key in ('source_report','artifacts_dir'):
        if row.get(key):
            source = Path(row[key])
            path = (source.parent if key == 'source_report' else source)/'program.ptdsl'
            if path.exists():
                return path
    raise ValueError('Retained U2 source program missing: '+row['model']+'/'+str(row['slots']))


COMPLETED = {
    'resnet20_relu': 'u3a-model-20261024T142804105549',
    'mobilenet_silu': 'u3a-model-20261024T142957161385',
    'mobilenet_relu': 'u3a-model-20261024T143323741670',
    'squeezenet_silu_avgpool': 'u3a-model-20261024T144305191394',
    'squeezenet_relu_maxpool': 'u3a-model-20261024T145117630480',
}


def memory_before(args):
    """Measure the retained failing implementation once under the same limit."""
    from .u2_multi_layout import baseline_package
    old = baseline_package(args.snapshot)
    previous = args.resume_from/'model'
    info = old('models.info').ModelInfo.load(previous)
    record = json.loads((previous/'ACCEPTANCE.json').read_text())
    module, _ = old('intake.probe').load_callable(record['source'])
    module = module.double().eval()
    arrays, _ = old('data.samples').StoredSplits(previous/'samples', info).get('heldout')
    source = (previous/'program.ptdsl').read_text()
    program = old('dsl.parser').parse(source)
    result = dict(model=record['model'], images_attempted=1, address_space_bytes=4*1024**3,
                  source=str(previous/'program.ptdsl'), snapshot=str(args.snapshot))
    try:
        backend = old('backends.mock').MockBackend(program.header.slots, batch=1)
        lower_module = old('lower')
        backend.bind(lower_module.pack_inputs(program, [arrays[info.graph.inputs[0]][:1]]))
        values, layouts = lower_module.lower(program, info.graph.weights, backend)
        lower_module.decode_outputs([value.value for value in values], layouts)
        result['mock_completed'] = True
    except MemoryError as exc:
        result.update(mock_completed=False, error=f'{type(exc).__name__}: {exc}')
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write(args.out/'MEMORY_BEFORE.json', result)
    return dict(memory_before=result)


def memory_proof(args):
    import numpy as np
    from .u2_multi_layout import baseline_package
    from ..backends import CountBackend
    from ..data.samples import StoredSplits
    from ..dsl.parser import parse
    from ..lower import lower
    from ..models.info import ModelInfo
    from ..models.zoo import get
    from ..planners import agent_tools
    from ..tests.test_p2_export import mock_reference
    old = baseline_package(args.snapshot)
    counts, validations, numeric = [], [], []
    for name, directory in COMPLETED.items():
        case = args.reference.parent/directory/'model'
        info = ModelInfo.load(case)
        source = (case/'program.ptdsl').read_text()
        a = old('backends.count').CountBackend(16384)
        b = CountBackend(16384)
        old('lower').lower(old('dsl.parser').parse(source), info.graph.weights, a)
        lower(parse(source), info.graph.weights, b)
        equal(a.stats(), b.stats(), name+' Count changed')
        counts.append(dict(model=name, count_equal=True, stats=b.stats()))
        write(args.out/'COUNT_EQUAL.json', counts)
        print('MEMORY Count '+name+' identical', flush=True)
        del info, a, b
        gc.collect()
    for name in ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp'):
        model = get(name, args.models_dir)
        graph = model.import_graph()
        source = (args.reference/'models'/name/'program.ptdsl').read_text()
        calibration = json.loads((args.reference/'models'/name/'calibration.json').read_text())
        first = old('planners.agent_tools').build_context(graph, calibration, n_val=20,
                                                         normalization=model.normalization)
        second = agent_tools.build_context(graph, calibration, n_val=20, normalization=model.normalization)
        before = old('planners.agent_tools').validate(old('dsl.parser').parse(source), first,
                                                     engine='mock', cache=False)
        gc.collect()
        after = agent_tools.validate(parse(source), second, engine='mock', cache=False)
        equal(before, after, name+' Mock validation changed')
        validations.append(dict(model=name, mock_validate_equal=True, validate=after, images=20))
        write(args.out/'VALIDATE_EQUAL.json', validations)
        print('MEMORY validate '+name+' identical', flush=True)
        del model, graph, first, second, before, after
        gc.collect()
    for name in ('resnet20_relu', 'squeezenet_relu_maxpool'):
        case = args.reference.parent/COMPLETED[name]/'model'
        info = ModelInfo.load(case)
        record = json.loads((case/'ACCEPTANCE.json').read_text())
        samples = Path(record.get('sample_source_dir', str(case/'samples')))
        arrays, _ = StoredSplits(samples, info).get('heldout', 20)
        program = parse((case/'program.ptdsl').read_text())
        with np.load(case/'numeric.npz') as archive:
            before = archive['mock']
        rows = []
        for i in range(20):
            rows.append(mock_reference(program, info.graph.weights, arrays[info.graph.inputs[0]][i:i+1]))
            gc.collect()
        after = np.concatenate(rows)
        if not np.array_equal(before, after):
            raise AssertionError(name+' retained Mock outputs changed')
        np.savez_compressed(args.out/(name+'.npz'), before=before, after=after)
        numeric.append(dict(model=name, images=20, array_equal=True, historical=str(case/'numeric.npz')))
        write(args.out/'MOCK_EQUAL.json', numeric)
        print('MEMORY numeric '+name+' 20/20 array_equal', flush=True)
        del info, arrays, program, before, after, rows
        gc.collect()
    return dict(counts=counts, validations=validations, numeric=numeric)


def compatibility(args):
    import numpy as np
    from . import u2_multi_layout as previous
    from ..backends import EmitBackend
    from ..dsl.parser import parse
    from ..lower import lower
    from ..models.zoo import get
    from ..planners import prompts, agent_tools

    previous.BASE = BASE  # Evidence label only; the source archive is supplied by the outer process.
    result = previous.single_proof(args)
    old = previous.baseline_package(args.snapshot)
    rows = result['programs']
    for row in variants(args):
        model = get(row['model'], args.models_dir)
        weights = model.import_graph().weights
        path = program_path(row); text = path.read_text()
        first, second = old('backends.emit').EmitBackend(row['slots']), EmitBackend(row['slots'])
        old('lower').lower(old('dsl.parser').parse(text), weights, first)
        lower(parse(text), weights, second)
        a, ac = first.render(); b, bc = second.render()
        equal(a,b,'U2 Emit changed: '+str(path))
        equal(list(ac),list(bc),'U2 constant order changed')
        for name in ac:
            if not np.array_equal(ac[name],bc[name]):
                raise AssertionError('U2 constant changed: '+name)
        rows.append(dict(program=row['model']+'-'+str(row['slots']),source=str(path),
                         baseline_rev=BASE,emit_text_equal=True,constants_equal=True,constants=len(ac)))
        write(args.out/'SINGLE_EMIT.json',rows)
        print('COMPAT '+rows[-1]['program']+' identical',flush=True)
        del first,second,a,b,ac,bc,weights,model;gc.collect()
    prompt_rows=[]
    for name in ('resnet20','resnet20_silu','cnn_plain','mlp'):
        model=get(name,args.models_dir); graph=model.import_graph()
        calibration=json.loads((args.reference/'models'/name/'calibration.json').read_text())
        before=old('planners.agent_tools').build_context(graph,calibration,n_val=20,normalization=model.normalization)
        after=agent_tools.build_context(graph,calibration,n_val=20,normalization=model.normalization)
        a=old('planners.prompts').build_system_prompt(before);b=prompts.build_system_prompt(after)
        equal(a,b,name+' default system prompt changed')
        initial_a=old('planners.prompts').initial_message(before);initial_b=prompts.initial_message(after)
        equal(initial_a,initial_b,name+' default initial message changed')
        directory=args.out/'prompts'/name;directory.mkdir(parents=True)
        (directory/'r8-v4.system.txt').write_text(b)
        (directory/'r8-v4.initial.json').write_text(initial_b)
        prompt_rows.append(dict(model=name,version=prompts.PROMPT_VERSION,system_equal=True,initial_equal=True))
        del before,after,graph,model;gc.collect()
    write(args.out/'PROMPTS.json',prompt_rows)
    result.update(programs=rows,prompts=prompt_rows,program_count=len(rows))
    equal(len(rows),17,'Expected all 17 retained programs')
    return result


def latency_diagnosis(args):
    from ..backends import DagBackend
    from ..cost import estimate, load_latency, plan_latency, PROFILES
    from ..dsl.parser import parse
    from ..lower import lower
    from ..models.zoo import get
    from ..placement import depth_dp

    rows=[]
    model_keys={'mul_single':'mul_cp','mul_double':'mul_cc','rescale_single':'rescale',
                'add_double':'add_cc','add_single':'add_cp','modswitch_single':'mod_switch',
                'rotate_single':'rotate','negate_single':'negate','bootstrap_single':'boot'}
    for row in variants(args):
        graph=get(row['model'],args.models_dir).import_graph()
        program=parse(program_path(row).read_text())
        backend=DagBackend(row['slots']);lower(program,graph.weights,backend)
        counted=defaultdict(Counter)
        class TableRow:
            def __init__(self,key,values): self.key,self.values=key,values
            def __getitem__(self,level):
                counted[model_keys.get(self.key,self.key)][level]+=1
                return self.values[level]
        latency=load_latency('qwen48')
        table={key:TableRow(key,values) for key,values in latency.items()}
        placement=depth_dp(backend,last_boot_level=4,boot_in_level=1,boot_out_level=5,latency_table=table)
        cost=estimate(program,graph.weights)
        plan_path=Path(row['runtime_plan']);plan=json.loads(plan_path.read_text())
        levels={value['id']:value['level'] for value in plan['values']}
        actual=defaultdict(Counter)
        for step in plan['execution']:
            name=step['op']
            level=levels[step['inputs'][0]] if step.get('inputs') else None
            actual[name][str(level)]+=1
        actual_latency=plan_latency(plan_path)
        equal(placement.boots,row['boots_compiled'],'U2 retained bootstrap count changed')
        equal(cost['boots_pred'],row['boots_compiled'],'U2 current bootstrap count changed')
        equal(cost['latency_pred_us'],placement.latency_us,'Instrumented latency differs')
        model_hist={name:{str(level):count for level,count in sorted(hist.items())} for name,hist in counted.items()}
        for name in ('mod_switch','relinearize'):
            model_hist.setdefault(name,{})
        record=dict(model=row['model'],slots=row['slots'],boots_pred=cost['boots_pred'],
            boots_compiled=row['boots_compiled'],rotations_pred=cost['counts']['rot'],
            latency_pred_us=cost['latency_pred_us'],latency_plan_us=actual_latency['latency_us'],
            latency_error_pct=100*abs(cost['latency_pred_us']-actual_latency['latency_us'])/actual_latency['latency_us'],
            model_level_hist=model_hist,plan_level_hist=dict(actual),runtime_plan=str(plan_path),
            model_counts={name:sum(hist.values()) for name,hist in model_hist.items()},
            plan_counts={name:sum(hist.values()) for name,hist in actual.items()})
        rows.append(record);write(args.out/'LATENCY.json',rows)
        print('LATENCY '+record['model']+'/'+str(record['slots'])+' '+str(record['latency_error_pct']),flush=True)
        del backend,graph,program;gc.collect()
    return dict(variants=rows,boots_equal=True)


def fixed_cifar_import(source, holder):
    from ..data.cifar import calibration_set
    from ..data.samples import Samples, legacy_cifar
    from ..intake.probe import load_callable, import_torch, ImportedModel
    import numpy as np
    module,name=load_callable(source)
    images,labels=calibration_set()
    samples=Samples('tensor',[{'input':images[:1]}],['frozen-calibration-example'],labels[:1])
    imported=import_torch(module,samples,name=name,source=source,split=dict(calibration=[0]))
    info,original=legacy_cifar(imported.info.graph)
    info.source=imported.info.source
    holder['module']=module.double().eval()
    return ImportedModel(info,original,module)


def model_acceptance(args):
    if args.resume_from is not None:
        return resume_model(args)
    import numpy as np
    import torch
    from types import SimpleNamespace
    from .. import cli,toolchain
    from ..backends import CountBackend
    from ..cost import estimate,plan_latency
    from ..data.cifar import calibration_set
    from ..data.samples import StoredSplits
    from ..dsl.parser import parse
    from ..export import export_package
    from ..export.codegen import generate
    from ..intake.coverage import report,check
    from ..intake.graph_exec import forward
    from ..kernels.layout import layout_blocks
    from ..lower import lower
    from ..models.info import ModelInfo
    from ..models.dacapo_builders import VARIANTS
    from ..run import _server_job
    from ..tensor_sim import run_tensor,_metadata,clear_caches
    from ..tests.test_p2_export import mock_reference,package_modules
    from .u2_multi_layout import compile_case

    if args.only not in VARIANTS:
        raise ValueError('--only must name one of the nine DaCapo variants')
    case=args.out/'model';case.mkdir()
    images,labels=calibration_set()
    inputs=case/'original-example.npz'
    np.savez(inputs,input=images[:1],labels=labels[:1])
    options=SimpleNamespace(model='pt_agent/models/dacapo_builders.py:'+args.only,inputs=inputs,
        out=case,stages=['calibrate','plan','check'],deg=127,margin=1.5,
        intake_python=str(PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'))
    holder={}
    status=cli._prepare(options,sample_source=lambda source:fixed_cifar_import(source,holder))
    record=dict(model=args.only,source=options.model,slots=16384,complete=False,
        model_entry='U1 CLI --model, internal fixed legacy_cifar source',calibration='frozen training selection',
        heldout='test[9000:10000]',report_images_read=0,llm_calls=0,intake_exit=status)
    write(case/'ACCEPTANCE.json',record)
    if status!=0:raise AssertionError('DaCapo U1 intake failed; see model/REPORT.json')
    status=cli._main_stages(options)
    record['cli_exit']=status;write(case/'ACCEPTANCE.json',record)
    if status!=0:raise AssertionError('DaCapo U1 stages failed; see model/REPORT.json')
    info=ModelInfo.load(case);graph=info.graph;module=holder.pop('module')
    source=(case/'program.ptdsl').read_text();program=parse(source)
    record['coverage']=report(graph);record['graph_check']=check(graph)
    equal(record['coverage']['lowering_ratio'],1.0,'CNN lowering coverage is incomplete')
    generated=generate(program)
    bn_nodes=[node for node in graph.nodes if node.op=='batchnorm' and node.attrs['weight']=='bn_1.weight']
    record['bn_folds']=[]
    for node in bn_nodes:
        following=graph.consumers(node.outputs[0])
        if len(following)==1 and following[0].op=='linear':
            prefix=node.attrs['weight'].removesuffix('.weight')
            equal('pre_bn='+repr(prefix) in generated,True,'Expected BN folded into following linear')
            record['bn_folds'].append(dict(prefix=prefix,into=following[0].attrs['weight'],direction='following'))
    numeric_check(args,case,info,record,program,case/'samples')
    del holder;gc.collect()
    return finish_model(args,case,info,module,record,inputs,source,program)


def numeric_check(args,case,info,record,program,samples):
    import numpy as np
    from ..data.samples import StoredSplits
    from ..tensor_sim import run_tensor
    from ..tests.test_p2_export import mock_reference
    graph=info.graph
    arrays,_=StoredSplits(samples,info).get('heldout',20)
    images=arrays[graph.inputs[0]]
    mock_rows,tensor_rows=[],[]
    normalized=maximum=0.0
    for start in range(20):
        batch=images[start:start+1]
        actual=mock_reference(program,graph.weights,batch)
        if start==0:
            record['first_mock_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        # Mock constants hold backend cycles; release each completed image before the next.
        gc.collect()
        tensor=run_tensor(program,graph.weights,batch,cache=False)
        error=np.abs(actual-tensor)
        maximum=max(maximum,float(error.max()))
        normalized=max(normalized,float((error/(1+np.abs(tensor))).max()))
        if not np.all(error<=1e-9*(1+np.abs(tensor))):
            raise AssertionError('Mock/tensor numeric difference exceeds U3 bound at heldout '+str(start))
        if not np.array_equal(actual.argmax(axis=1),tensor.argmax(axis=1)):
            raise AssertionError('Mock/tensor class differs at heldout '+str(start))
        mock_rows.append(actual);tensor_rows.append(tensor)
        print('MODEL '+args.only+' Mock/tensor '+str(start+1)+'/20',flush=True)
    np.savez_compressed(case/'numeric.npz',mock=np.concatenate(mock_rows),tensor=np.concatenate(tensor_rows))
    record.update(images=20,numeric_equal=True,classes_equal=True,max_abs_diff=maximum,max_normalized_diff=normalized,
                  mock_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(case/'ACCEPTANCE.json',record)
    del arrays,images;gc.collect()


def resume_model(args):
    import shutil
    import numpy as np
    import torch
    from ..dsl.parser import parse
    from ..intake.probe import load_callable
    from ..intake.graph_exec import forward
    from ..models.info import ModelInfo
    previous=args.resume_from/'model'
    record=json.loads((previous/'ACCEPTANCE.json').read_text())
    equal(record['model'],args.only,'Resume model differs')
    if (record.get('complete') or record.get('cli_exit')!=0
            or record.get('estimate') or record.get('package') or record.get('trace')):
        raise ValueError('Resume requires completed CLI stages and unfinished numerical acceptance')
    case=args.out/'model';case.mkdir()
    source=(previous/'program.ptdsl').read_text()
    (case/'program.ptdsl').write_text(source)
    for name in ('original-example.npz','calibration.json'):
        shutil.copy2(previous/name,case/name)
    info=ModelInfo.load(previous);info.save(case)
    module,_=load_callable(record['source']);module=module.double().eval()
    samples=Path(record.get('sample_source_dir',str(previous/'samples')))
    reused=record.get('numeric_equal') and record.get('images')==20
    record.update(source_acceptance=str(previous/'ACCEPTANCE.json'),sample_source_dir=str(samples),
                  reused_numeric_images=20 if reused else 0,resumed_heldout_from=record.get('heldout_done',0))
    program=parse(source)
    if reused:
        shutil.copy2(previous/'numeric.npz',case/'numeric.npz')
    else:
        from ..intake.coverage import report,check
        record.update(coverage=report(info.graph),graph_check=check(info.graph))
        numeric_check(args,case,info,record,program,samples)
    # The old partial checkpoint predates persistence of the reference-error value.
    # Recheck only this independent import proof; approximate scores are resumed.
    from ..data.samples import StoredSplits
    arrays,_=StoredSplits(samples,info).get('heldout',20)
    batch=arrays[info.graph.inputs[0]]
    with torch.no_grad():reference=module(torch.from_numpy(batch)).numpy()
    error=np.abs(reference-forward(info.graph,arrays)[info.graph.outputs[0]])
    if not np.all(error<=1e-9*(1+np.abs(reference))):
        raise AssertionError('Resumed imported graph differs from original module')
    record['import_reference_max_diff']=float(error.max())
    write(case/'ACCEPTANCE.json',record)
    del arrays,batch,reference,error;gc.collect()
    return finish_model(args,case,info,module,record,case/'original-example.npz',source,program,samples_dir=samples)


def finish_model(args,case,info,module,record,inputs,source,program,*,samples_dir=None):
    import numpy as np
    import torch
    from ..backends import CountBackend
    from ..backends.mock import MockBackend
    from ..backends.base import BackendError
    from ..cost import estimate
    from ..data.samples import StoredSplits
    from ..export import export_package
    from ..intake.graph_exec import forward
    from ..kernels.layout import layout_blocks
    from ..lower import lower
    from ..run import _server_job
    from ..tensor_sim import run_tensor,_metadata,clear_caches
    from ..tests.test_p2_export import mock_reference,package_modules
    from .u2_multi_layout import compile_case
    graph=info.graph
    arrays,labels=StoredSplits(samples_dir or case/'samples',info).get('heldout')
    images=arrays[graph.inputs[0]]
    done=record.get('heldout_done',0)
    exact_correct=round(record.get('exact_top1',0)*done/100)
    correct=round(record.get('top1',0)*done/100)
    diverged=list(record.get('diverged_indices',[]))
    import_max=record.get('import_reference_max_diff',0.0)
    gc.collect()
    for start in range(done,len(images),20):
        batch=images[start:start+20]
        with torch.no_grad():reference=module(torch.from_numpy(batch)).numpy()
        if start==0:
            independently=forward(graph,{graph.inputs[0]:batch})[graph.outputs[0]]
            difference=np.abs(reference-independently)
            import_max=float(difference.max())
            if not np.all(difference<=1e-9*(1+np.abs(reference))):
                raise AssertionError('Imported graph differs from original module')
        exact_correct+=int(np.sum(reference.argmax(axis=1)==labels[start:start+len(batch)]))
        try:
            approximate=run_tensor(program,graph.weights,batch,cache=False)
            correct+=int(np.sum(approximate.argmax(axis=1)==labels[start:start+len(batch)]))
        except BackendError:
            for offset in range(len(batch)):
                try:
                    approximate=run_tensor(program,graph.weights,batch[offset:offset+1],cache=False)
                except BackendError:
                    diverged.append(9000+start+offset)
                else:correct+=int(approximate.argmax(axis=1)[0]==labels[start+offset])
        record.update(heldout_done=start+len(batch),top1=100*correct/(start+len(batch)),
                      correct=correct,exact_correct=exact_correct,import_reference_max_diff=import_max,
                      exact_top1=100*exact_correct/(start+len(batch)),n_diverged=len(diverged),diverged_indices=diverged)
        write(case/'ACCEPTANCE.json',record)
        gc.collect()
        print('MODEL '+args.only+' heldout '+str(start+len(batch))+'/1000',flush=True)
    record.update(acc_drop_pp=record['exact_top1']-record['top1'],import_reference_max_diff=import_max)
    del module,images,arrays;gc.collect();clear_caches()
    cost=estimate(program,graph.weights)
    count=CountBackend(program.header.slots)
    values,layouts=lower(program,graph.weights,count)
    metadata=_metadata(program,graph.weights)[0]
    maximum_ct=max(len(layout_blocks(lay)) for _,_,lay,_ in metadata if hasattr(lay,'n'))
    record.update(estimate=cost,boots_pred=cost['boots_pred'],rotations=cost['counts']['rot'],
        latency_pred_us=cost['latency_pred_us'],const_bytes=cost['const_bytes'],
        input_ciphertexts=len(count._inputs),output_ciphertexts=len(values),max_tensor_ciphertexts=maximum_ct)
    del count,values,metadata;gc.collect()
    write(case/'ACCEPTANCE.json',record)
    package=export_package(source,program,graph.weights,case/'export')
    weights_path=case/'export/weights'/(program.name+'.npz')
    record.update(package=str(package),weights=str(weights_path))
    with package_modules(package) as runtime:
        original_inputs=np.load(inputs)['input']
        expected=mock_reference(program,graph.weights,original_inputs)
        gc.collect()
        if cost['const_bytes']>1.5*1024**3:
            backend=MockBackend(program.header.slots,batch=len(original_inputs))
            backend.bind(runtime.pack_arrays([original_inputs]))
            runtime.invoke(backend,graph.weights)
            actual=runtime.unpack_arrays(backend.outputs)[0]
            record['package_stream_mock_exact']=True
        else:
            actual=runtime.run_numpy(graph.weights,[original_inputs])[0]
            record['package_numpy_exact']=True
        if not np.array_equal(actual,expected):raise AssertionError('P2/Mock outputs differ')
    del graph,info;gc.collect()
    if cost['const_bytes']>1.5*1024**3:
        record.update(native_compiled=False,compile_reason='constants exceed local 1.5 GiB limit',boots_compiled=None,
                      server_job=str(_server_job(case,package,weights_path)))
    else:
        native=case/'native';native.mkdir()
        compile_case(native,record,package,weights_path,program.name)
        record.update(native_compiled=True,compile_reason='completed within trusted limits')
    record['complete']=True
    write(case/'ACCEPTANCE.json',record)
    return dict(model=record)


def coverage(args):
    from ..models.info import ModelInfo
    from ..models.dacapo_builders import VARIANTS
    from ..intake.coverage import report,check
    original=json.loads((args.reference.parent/'u1-section12-complete-20261024T111952470042/REPORT.json').read_text())
    rows=[]
    for row in original['coverage']:
        info=ModelInfo.load(row['artifacts_dir'])
        rows.append(dict(model=row['model'],coverage=report(info.graph),check=check(info.graph),source=row['artifacts_dir']))
        del info;gc.collect()
    for name in VARIANTS:
        completed=[]
        for path in sorted(args.reference.parent.glob('u3a-model-*/model/ACCEPTANCE.json')):
            record=json.loads(path.read_text())
            if record.get('complete') and record.get('model')==name:
                completed.append(path.parent)
        if not completed:
            raise AssertionError('Completed U3a acceptance missing: '+name)
        directory=completed[-1]
        info=ModelInfo.load(directory)
        rows.append(dict(model=name,coverage=report(info.graph),check=check(info.graph),source=str(directory)))
        del info;gc.collect()
    write(args.out/'COVERAGE.json',rows)
    return dict(models=rows)


def deferred_package(args):
    from .. import toolchain
    from ..cost import estimate
    from ..dsl.parser import parse
    from ..export import export_package
    from ..models.info import ModelInfo
    from ..run import _server_job
    if args.resume_from is None:
        raise ValueError('Deferred packaging requires the retained source run')
    previous=args.resume_from/'model'
    info=ModelInfo.load(previous)
    source=(previous/'program.ptdsl').read_text();program=parse(source)
    toolchain.setup_env()
    cost=estimate(program,info.graph.weights)
    gc.collect()
    package=export_package(source,program,info.graph.weights,args.out/'export')
    weights=args.out/'export/weights'/(program.name+'.npz')
    job=_server_job(args.out,package,weights)
    record=dict(model=args.only,source_program=str(previous/'program.ptdsl'),
        source_failure=str(args.resume_from/'REPORT.json'),estimate=cost,package=str(package),
        weights=str(weights),server_job=str(job),native_compiled=False,
        numeric_complete=False,heldout_complete=False,
        reason='Local Mock run hit the unchanged 4 GiB address-space limit. Only the server trace/compile package is prepared; numerical acceptance remains pending.')
    write(args.out/'DEFERRED.json',record)
    return dict(deferred=record)


def pool_calibration(args):
    import numpy as np
    import torch
    from ..analyzer import calibrate
    from ..data.cifar import calibration_set
    from ..dsl.parser import parse
    from ..intake import import_module
    from ..planners.rule import plan
    from ..tensor_sim import run_tensor
    from ..tests.test_u3_kernels import mock
    images,_=calibration_set()
    rows=[]
    for k,pad in ((2,0),(3,1)):
        module=torch.nn.MaxPool2d(k,2,pad).double().eval()
        graph=import_module(module,(torch.from_numpy(images[:1]),),'calibrated_pool_'+str(k),frontend='aten')
        calibration=calibrate(graph,{graph.inputs[0]:images})
        source=plan(graph,calibration,deg=127,margin=1.5)
        program=parse(source)
        maximum=max_u=consistency=0.0
        for start in range(0,len(images),20):
            batch=images[start:start+20]
            observations=[]
            def observer(name,value,layout,cfg):
                lo,hi=cfg[-2:]
                observations.append(float(np.max(np.abs(value.value*(2/(hi-lo))-(hi+lo)/(hi-lo)))))
            actual=mock(program,graph.weights,batch,observer=observer)
            gc.collect()
            tensor=run_tensor(program,graph.weights,batch,cache=False)
            with torch.no_grad():reference=module(torch.from_numpy(batch)).numpy()
            error=np.abs(actual-tensor)
            if not np.all(error<=1e-9*(1+np.abs(tensor))):raise AssertionError('Calibration pool Mock/tensor mismatch')
            consistency=max(consistency,float(error.max()))
            maximum=max(maximum,float(np.max(np.abs(actual-reference))))
            max_u=max(max_u,max(observations))
        op=next(op for op in program.ops if op.kind=='maxpool')
        rows.append(dict(kernel=k,stride=2,padding=pad,images=len(images),source='frozen training calibration selection',
            degree=op.attrs['deg'],domain=op.attrs['domain'],max_abs_error=maximum,max_u=max_u,
            mock_tensor_max_diff=consistency,relu_passes=2*(k-1)))
        write(args.out/'POOL_CALIBRATION.json',rows)
        (args.out/('pool-'+str(k)+'.ptdsl')).write_text(source)
    return dict(pools=rows)


def efficiency(args):
    import numpy as np
    from ..backends import CountBackend
    from ..dsl.parser import parse
    from ..kernels import cnn
    from ..kernels.conv import conv2d_mux
    from ..kernels.layout import mux_layout
    from ..lower import lower
    from ..tests.test_u3_kernels import source
    rng=np.random.default_rng(20261024)
    rows=[]
    for channels in (4,8):
        layout=mux_layout(1024,channels,4,4,1)
        weight=rng.normal(size=(channels,1,3,3))
        full=np.zeros((channels,channels,3,3))
        for channel in range(channels):full[channel,channel]=weight[channel,0]
        new,old=CountBackend(1024),CountBackend(1024)
        cnn.depthwise_blocks(new,new.input(0),layout,weight)
        conv2d_mux(old,old.input(0),layout,full,stride=1)
        rows.append(dict(channels=channels,depthwise=new.stats(),ordinary=old.stats()))
    pools=[]
    for slots in (4096,1024):
        for kernel,pad in ((2,0),(3,1)):
            text=source(slots,8,f'    y = maxpool(p,k={kernel},stride=2,pad={pad},method="cheb",deg=127,domain=(-12.,12.))\n    return y')
            backend=CountBackend(slots)
            lower(parse(text),{},backend)
            pools.append(dict(slots=slots,kernel=kernel,padding=pad,relu_passes=2*(kernel-1),stats=backend.stats()))
    write(args.out/'EFFICIENCY.json',dict(depthwise=rows,maxpool=pools))
    return dict(depthwise=rows,maxpool=pools)


def tests(args):
    import unittest
    if args.part=='unit':
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem)
            for p in sorted(Path('pt_agent/tests').glob('test_u3_*.py')))
    elif args.part=='intake-tests':
        intake=PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'
        if Path(sys.executable).absolute()!=intake.absolute():
            os.execv(str(intake),[str(intake),'-B','-m','pt_agent.experiments.u3_cnn_operators',*sys.argv[1:]])
        suite=unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.test_u1_onnx')
    else:
        packaging={'test_p2_export.py','test_toolchain_decoupled.py'}
        tensor={'test_r8f_tensor_validate.py'}
        files=[p for p in sorted(Path('pt_agent/tests').glob('test_*.py')) if not p.name.startswith('test_u3_')]
        files=[p for p in files if (p.name in packaging if args.only=='packaging' else
               p.name in tensor if args.only=='tensor' else p.name not in packaging|tensor)]
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
        equal(suite.countTestCases(),{'main':401,'packaging':17,'tensor':17}[args.only],'435-test inventory changed')
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    record=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
    write(args.out/'TESTS.json',record)
    if not result.wasSuccessful():raise AssertionError('U3a tests failed; see run.log')
    return record


def inside(args):
    record=dict(complete=False,part=args.part,llm_calls=0,report_images_read=0)
    try:
        import torch
        torch.set_num_threads(1)
        if args.part=='compat':record.update(compatibility(args))
        elif args.part=='latency':record.update(latency_diagnosis(args))
        elif args.part=='model':record.update(model_acceptance(args))
        elif args.part=='coverage':record.update(coverage(args))
        elif args.part=='pool':record.update(pool_calibration(args))
        elif args.part=='efficiency':record.update(efficiency(args))
        elif args.part=='defer':record.update(deferred_package(args))
        elif args.part=='memory-before':record.update(memory_before(args))
        elif args.part=='memory-proof':record.update(memory_proof(args))
        else:record.update(tests(args))
        record['complete']=True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        record['error']=f'{type(exc).__name__}: {exc}'
    record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write(args.out/'REPORT.json',record)
    return 0 if record['complete'] else 1


def limits():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('unit','compat','latency','model','regression','intake-tests','coverage','pool','efficiency','defer','memory-before','memory-proof'),required=True)
    parser.add_argument('--only')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--models-dir',type=Path)
    parser.add_argument('--u2',type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    if args.inside:return inside(args)
    args.out=(args.out or RESULTS_DIR/('u3a-'+args.part+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))).resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    args.reference=(args.reference or RESULTS_DIR/FROZEN).resolve()
    args.models_dir=(args.models_dir or RESULTS_DIR/'models-v2').resolve()
    args.u2=(args.u2 or RESULTS_DIR/U2).resolve()
    if args.part=='compat':
        archive=args.out/'baseline.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git','archive',BASE,'DSL/DSL_agent/scripts/pt_agent'],cwd=REPO,stdout=stream,check=True)
        args.snapshot=args.out/'baseline-tree';args.snapshot.mkdir()
        with tarfile.open(archive) as stream:stream.extractall(args.snapshot)
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
             '-m','pt_agent.experiments.u3_cnn_operators','--inside','--part',args.part,
             '--out',str(args.out),'--reference',str(args.reference),'--models-dir',str(args.models_dir),'--u2',str(args.u2)]
    if args.only:command+=['--only',args.only]
    if args.snapshot:command+=['--snapshot',str(args.snapshot)]
    if args.resume_from:command+=['--resume-from',str(args.resume_from.resolve())]
    write(args.out/'argv.json',command)
    env=dict(os.environ,POSEIDON_WORK_ROOT=str(WORK_ROOT),PT_DACAPO_ROOT=str(DACAPO_ROOT),
             PT_DACAPO_UPSTREAM=str(DACAPO_UPSTREAM),PT_HECATE_BUILD=str(HECATE_BUILD),PT_AGENT_RESULTS_ROOT=str(RESULTS_DIR))
    begin=time.monotonic();print('RUN='+str(args.out),flush=True)
    with (args.out/'run.log').open('w') as stream:
        try:
            status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,preexec_fn=limits,timeout=3720).returncode
        except subprocess.TimeoutExpired:status=124
    path=args.out/'REPORT.json'
    record=json.loads(path.read_text()) if path.exists() else dict(complete=False)
    record.update(exit=status,seconds=time.monotonic()-begin)
    write(path,record)
    from ..registry import append,new_run_id,git_rev
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=new_run_id('U3a',args.only or args.part,'acceptance'),round='U3a',model=args.only or args.part,
                mode='acceptance',git_rev=git_rev(),metrics=record,artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(path),flush=True)
    return status


if __name__=='__main__':
    raise SystemExit(main())
