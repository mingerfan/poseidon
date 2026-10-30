"""A1 section 6.16: observe unchanged Mock batches before changing retention."""

import gc
import json
from pathlib import Path
import resource
import sys
import types
from collections import Counter
from unittest.mock import patch


def retained_size(value, seen=None):
    """Count each retained container and ndarray once without keeping objects."""
    import numpy as np
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (value.nbytes if not value.flags.owndata else 0)
    if isinstance(value, dict):
        return size + sum(retained_size(k, seen)+retained_size(v, seen)
                          for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(retained_size(v, seen) for v in value)
    return size


def observation_state(observer):
    from ..domain_criteria import ViolationOrder
    function = getattr(observer, 'sample_event', None)
    for cell in getattr(function, '__closure__', ()) or ():
        value = cell.cell_contents
        if isinstance(value, ViolationOrder):
            return {key: getattr(value, key) for key in
                    ('inverse_outputs', 'relative_errors', 'chains', 'observed', 'rows')}
    return {}


def snapshot(observer=None):
    from ..backends.mock import MockBackend, MockExpr
    from ..kernels.slots import _current
    fields = {}
    backends = nodes = node_bytes = operations = constants = constant_bytes = 0
    for obj in gc.get_objects():
        if isinstance(obj, MockBackend):
            backends += 1
            operations += getattr(obj, '_nid', 0)
            constants += len(getattr(obj, '_constants', {}))
            constant_bytes += retained_size((getattr(obj, '_constants', {}),
                                             getattr(obj, '_constant_objects', {})))
        elif isinstance(obj, MockExpr):
            nodes += 1
            node_bytes += getattr(getattr(obj, 'value', None), 'nbytes', 0)
    current = _current.get()
    vector_bytes = retained_size(current[1:]) if current is not None else 0
    observed = observation_state(observer)
    fields.update(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  mock_backends=backends, mock_recorded_operations=operations,
                  live_mock_nodes=nodes, live_mock_node_array_bytes=node_bytes,
                  constant_entries=constants, backend_constant_cache_bytes=constant_bytes,
                  public_vector_cache_bytes=vector_bytes,
                  observer_cache_bytes=retained_size(observed),
                  observer_entries={key: len(value) for key, value in observed.items()},
                  gc_count=list(gc.get_count()))
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:')):
            key, value, *_ = line.split()
            fields[key[:-1]+'_bytes'] = int(value)*1024
    return fields


def unit(args):
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.test_a1_mock_memory')
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    return dict(complete=result.wasSuccessful(),tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),llm_calls=0,report_images=0)


def retention_references():
    """Describe actual referrers and recursive closures without saving tensors."""
    from ..backends.mock import MockBackend, MockExpr
    objects = gc.get_objects()
    backends = [x for x in objects if isinstance(x, MockBackend)]
    ignored = {id(objects), id(backends)}
    rows = []
    for backend in backends:
        refs = gc.get_referrers(backend)
        direct = Counter(type(x).__name__ for x in refs if id(x) not in ignored)
        closures = []
        for obj in objects:
            if not isinstance(obj, types.FunctionType) or obj.__qualname__ != 'cheb_eval.<locals>.rec':
                continue
            captures = dict(zip(obj.__code__.co_freevars,
                                (cell.cell_contents for cell in obj.__closure__)))
            base = captures.get('base')
            base_captures = (dict(zip(base.__code__.co_freevars,
                                     (cell.cell_contents for cell in base.__closure__)))
                             if isinstance(base, types.FunctionType) else {})
            tables = {key: value for key, value in {**captures, **base_captures}.items()
                      if isinstance(value, dict)}
            related = {key: [x for x in value.values()
                            if isinstance(x, MockExpr) and x.backend is backend]
                       for key, value in tables.items()}
            if any(related.values()):
                closures.append(dict(function=obj.__qualname__,
                    source=obj.__code__.co_filename,recursive_self_reference=captures.get('rec') is obj,
                    retained_tables={key: dict(nodes=len(value),array_bytes=sum(x.value.nbytes for x in value))
                                     for key,value in related.items()}))
        rows.append(dict(backend_class=type(backend).__name__,direct_referrer_types=dict(direct),
            input_endpoint_cycle=any(x.backend is backend for x in backend._inputs),
            output_endpoint_cycle=any(x.backend is backend for x in backend._outputs or ()),
            identity_cache_entries=len(backend._constant_objects),constant_entries=len(backend._constants),
            recursive_closures=closures))
    return dict(backends=rows,method='gc.get_referrers plus closure cell inspection')


def heldout(args):
    """Retain before/after measurements; confirmation and report stay unused."""
    import numpy as np
    from ..dsl.parser import parse
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    from ..planners.transformer_session import json_safe
    from .. import transformer_validation as validation
    from .a1_transformer_agent import read,samples_for
    info=ModelInfo.load(args.model_dir)
    ctx=build_context(info,read(args.model_dir/'calibration.json'),samples=samples_for(args.model_dir,info))
    ctx.prompt_version=args.prompt_version
    program=parse(args.program.read_text())
    outputs=[];diagnostics=[]
    original=validation.execute
    def execute(*values,**options):
        result=original(*values,**options)
        if args.part=='mock-before':
            (args.out/'REFERRERS.json').write_text(json.dumps(retention_references(),indent=2)+'\n')
        return result
    with patch.object(validation,'execute',execute):
        measured=validation.validate(program,ctx,n=20,engine='mock',stop_on_divergence=False,
            output_callback=outputs.append,diagnostic_callback=diagnostics.append)
    actual=outputs[0]
    np.savez_compressed(args.out/'outputs.npz',mock=actual)
    measured['logit_rel_err']=float(np.max(np.abs(actual-ctx.val_exact_logits[:20])))/float(np.max(np.abs(ctx.val_exact_logits[:20])))
    result=dict(complete=True,program=str(args.program),split='heldout',samples=20,engine='mock',
                validation=json_safe(measured),diagnostics=json_safe(diagnostics),llm_calls=0,report_images=0)
    if args.part=='mock-equivalence':
        before=read(args.root/'REPORT.json')
        if not before['complete'] or before['program']!=str(args.program):
            raise ValueError('Equivalence requires the same completed before program')
        with np.load(args.root/'outputs.npz') as data:expected=data['mock']
        first=lambda reports:[s['first_violation'] for r in reports for s in r['samples']]
        u=lambda v:([r['max_u'] for r in v['rows']],v['max_u'],v.get('max_u_calibrated'),v.get('max_u_analytic'))
        checks=dict(agree=measured['agree']==before['validation']['agree'],
            first_violation=first(json_safe(diagnostics))==first(before['diagnostics']),
            u=u(measured)==u(before['validation']),
            logits=bool(np.all(np.abs(actual-expected)<=1e-12+1e-12*np.abs(expected))))
        result.update(complete=all(checks.values()),comparison=checks,
                      max_abs_diff=float(np.max(np.abs(actual-expected))),before=str(args.root/'REPORT.json'))
        if not result['complete']:
            (args.out/'COMPARISON.json').write_text(json.dumps(json_safe(result),indent=2)+'\n')
            raise RuntimeError('A1 section 6.16 heldout equivalence differs')
    return result


def frozen_metrics(args):
    """Compute missing frozen statistics without candidate selection."""
    import numpy as np
    from ..dsl.parser import parse
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    from ..planners.transformer_session import json_safe
    from .. import transformer_validation as validation
    from .a1_transformer_agent import read,samples_for,split_context
    marker=read(args.frozen)
    if Path(marker['program']).read_text()!=args.program.read_text():
        raise ValueError('Post-freeze metrics require the retained unchanged program')
    info=ModelInfo.load(args.model_dir)
    ctx=build_context(info,read(args.model_dir/'calibration.json'),samples=samples_for(args.model_dir,info))
    ctx.prompt_version=args.prompt_version
    program=parse(args.program.read_text())
    rows={}
    for split in ('heldout','confirm'):
        selected=ctx if split=='heldout' else split_context(ctx,split,n=20)
        captured=[]
        measured=validation.validate(program,selected,n=20,engine='tensor',
            stop_on_divergence=False,output_callback=captured.append)
        expected=selected.val_exact_logits[:20]
        measured['logit_rel_err']=float(np.max(np.abs(captured[0]-expected)))/float(np.max(np.abs(expected)))
        rows[split]=json_safe(measured)
    return dict(complete=True,program=str(args.program),frozen=str(args.frozen),
                engine='tensor',affects_selection=False,rows=rows,llm_calls=0,report_images=0)


def diagnostic(args):
    import numpy as np
    from ..dsl.parser import parse
    from ..dsl.interface import require_interface
    from ..models.info import ModelInfo
    from ..planners.agent_tools import build_context
    from ..planners.transformer_session import json_safe
    from ..tensor_sim import clear_caches
    from .. import transformer_validation as validation
    from .a1_transformer_agent import read, samples_for, split_context
    info = ModelInfo.load(args.model_dir)
    program = parse(args.program.read_text())
    require_interface(program, info)
    ctx = build_context(info, read(args.model_dir/'calibration.json'),
                        samples=samples_for(args.model_dir, info))
    ctx.prompt_version = args.prompt_version
    ctx = split_context(ctx, 'report', n=100)
    if len(ctx.val_images) != 100:
        raise ValueError('A1 section 6.16 requires the same report100 partition')
    clear_caches()
    original = validation.execute
    batch_number = 0
    log_path = args.out/'batches.jsonl'

    def record(phase, observer, **extra):
        row = dict(batch=batch_number, phase=phase, **snapshot(observer), **extra)
        with log_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
        print('MOCK_BATCH='+json.dumps(row, allow_nan=False), flush=True)

    def execute(*values, **options):
        nonlocal batch_number
        batch_number += 1
        observer = options.get('observer')
        record('start', observer)
        try:
            result = original(*values, **options)
        except BaseException as exc:
            record('failed', observer, error=type(exc).__name__+': '+str(exc))
            raise
        return result

    captured = []
    with patch.object(validation, 'execute', execute):
        measured = validation.validate(program, ctx, engine='mock',
                                       stop_on_divergence=False, output_callback=captured.append,
                                       batch_callback=lambda observer:record('end',observer))
    outputs = captured[0]
    np.savez_compressed(args.out/'outputs.npz', mock=outputs)
    denominator = float(np.max(np.abs(ctx.val_exact_logits)))
    measured['logit_rel_err'] = float(np.max(np.abs(outputs-ctx.val_exact_logits)))/denominator
    result=dict(complete=True,program=str(args.program),model_dir=str(args.model_dir),
                split='report',report_images=100,engine='mock',batch_size=20,
                validation=json_safe(measured),batches=str(log_path),llm_calls=0)
    if args.part=='mock-release-check':
        ends=[json.loads(x) for x in log_path.read_text().splitlines() if json.loads(x)['phase']=='end']
        checks=dict(batch_count=len(ends)==5,live_backends=all(x['mock_backends']<=1 for x in ends),
                    rss_growth=all(x['VmRSS_bytes']<=ends[0]['VmRSS_bytes']+0.3*1024**3 for x in ends[1:]))
        result.update(complete=all(checks.values()),release_checks=checks,
            batch_ends=ends,needs_batch10=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>3.8*1024**3)
        if not result['complete']:
            (args.out/'RELEASE_CHECK.json').write_text(json.dumps(json_safe(result),indent=2)+'\n')
            raise RuntimeError('A1 section 6.16 batch retention checks failed')
    return result
