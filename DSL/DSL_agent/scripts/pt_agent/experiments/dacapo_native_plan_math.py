"""Ideal decoded arithmetic of the unchanged original DaCapo RuntimePlan.

No CKKS, key, compiler, Torch model or API is invoked. Bootstrap, relinearize,
rescale and modulus switching are mathematical identities solely in this oracle.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import sys
import threading
import time

from . import dacapo_native_binding as binding
from . import r25_orbit_packed_worker as reference
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'DACAPO-NATIVE-MATH'
MODEL = 'dacapo-original-resnet20-silu'
MODE = 'original-dacapo-decoded-ideal-three-probe-math'
SCOPE = ('Unchanged original DaCapo RuntimePlan on train:0,train:1,train:2; '
         'decoded ideal identities for bootstrap/rescale/mod_switch/relinearize; '
         'original Poly physical32 reference and original strict mathematical criterion')
GIB = 1024**3
SLOTS = 65536
IDENTITIES = ('mod_switch', 'rescale', 'relinearize', 'boot')
EXPECTED = dict(execution_count=12661, value_count=14364, initialization_count=1702,
    inline_scalar_count=122, bundle_encode_count=1580, unique_blob_count=1497,
    bootstrap_count=19, last_use_ciphertext_peak_count=23,
    op_counts=dict(mod_switch=477, rotate=1340, mul_cp=4304, add_cc=2869,
        rescale=2701, add_cp=153, boot=19, mul_cc=361, relinearize=361, negate=76))
SOURCE_ARGS = ('plan', 'bundle', 'scalar_source_proof', 'poly_source_alignment',
               'compile_result', 'r13_result', 'r24_result', 'probes', 'probe_source', 'archive_manifest')
ZERO_FIELDS = ('api_calls', 'llm_calls', 'model_forwards', 'torch_model_builds',
    'torch_model_forwards', 'checkpoint_reads', 'raw_cifar_reads', 'heldout_samples',
    'confirm_samples', 'report_samples', 'compiler_runs', 'trace_runs', 'keygen_runs',
    'ckks_executions', 'native_bootstrap_executions', 'intermediate_decryptions',
    'new_downloads', 'new_installations', 'integrity_checks', 'archive_payload_reads', 'archive_restores')
FORMAL_FIELDS = ('predicted_latency_seconds', 'actual_ckks_latency_seconds',
                 'accuracy_acceptance', 'security_acceptance')
COUNTERS = ('plan_load_attempts', 'plan_load_completed', 'manifest_load_attempts',
    'manifest_load_completed', 'probe_load_attempts', 'probe_load_completed',
    'packed_forward_attempts', 'packed_forward_completed', 'math_check_attempts',
    'math_check_completed', 'op_attempts', 'op_completed')


def _np():
    return reference._numeric_modules()


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Source JSON must be an object')
    return value


def _same(a, b):
    return isinstance(a, (str, Path)) and Path(a).resolve() == Path(b).resolve()


def _file(path):
    path = Path(path)
    st = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(st.st_mode):
        raise ValueError('A regular explicit source file is required: '+str(path))
    return dict(path=str(path.resolve()), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def _clean(process):
    return (process.get('exit_code') == 0 and not process.get('timed_out')
        and process.get('signal') is None and not process.get('signals_sent')
        and not process.get('remaining_pids'))


def check_sources(args):
    """Check saved semantic identities; never open weights, samples or payloads."""
    compiled, r13, r24 = (_read(getattr(args, name)) for name in
                          ('compile_result', 'r13_result', 'r24_result'))
    if (compiled.get('round') != 'DACAPO-ORIGINAL' or compiled.get('model') != MODEL
            or compiled.get('status') != 'ok' or compiled.get('original_source_variant') is not True
            or compiled.get('r29_cheb63_variant') is not False
            or any(compiled.get(name) != 1 for name in ('trace_attempts','trace_completed',
                'compile_attempts','compile_completed','native_compile_attempts','native_compile_completed'))
            or compiled.get('bootstrap_count') != EXPECTED['bootstrap_count']
            or compiled.get('export_contract_compatible') is not True
            or not _same((compiled.get('plan') or {}).get('path'), args.plan)
            or not all(_clean(compiled.get(name) or {}) for name in ('trace_process','compile_process'))
            or not _clean((compiled.get('compile_worker') or {}).get('process') or {})):
        raise ValueError('A clean unchanged original DaCapo compilation and its actual plan are required')
    if (r13.get('round') != 'R13' or r13.get('status') != 'ok'
            or r13.get('model') != binding.SOURCE_FACTORY
            or r24.get('round') != 'R24' or r24.get('status') != 'ok'
            or r24.get('model') != binding.SOURCE_FACTORY
            or r24.get('mode') != 'orbit-native-polynomial-physical-scale-check'):
        raise ValueError('Original R13/R24 source identity differs')
    original = (compiled.get('sources') or {}).get('original_model') or {}
    source_model = (r13.get('worker') or {}).get('model_source') or {}
    if (original.get('factory') != binding.SOURCE_FACTORY
            or not _same(original.get('checkpoint'), source_model.get('checkpoint'))
            or not _same(original.get('original_model_source'), source_model.get('original_source'))
            or not _same((compiled.get('sources') or {}).get('saved_r13_result',{}).get('path'), args.r13_result)):
        raise ValueError('Original checkpoint/definition/R13 paths differ')
    artifacts = (r13.get('worker') or {}).get('artifacts') or {}
    if (not _same(artifacts.get('training_probes'), args.probes)
            or not _same(artifacts.get('training_probe_source'), args.probe_source)):
        raise ValueError('Only the three original saved training probes may be read')
    worker = r24.get('worker') or {}
    for name in ('original_reference_passed','native_polynomial_math_passed','physical_scaled_forward_completed'):
        if worker.get(name) is not True:
            raise ValueError('R24 independent physical32 source did not pass: '+name)
    expected_counts = dict(original_forward_attempts=3,original_forward_completed=3,
        approx_forward_attempts=3,approx_forward_completed=3,activation_attempts=57,
        activation_completed=57,math_check_attempts=57,math_check_completed=57)
    if (any(worker.get(k) != v for k,v in expected_counts.items())
            or worker.get('probe_source') != (r13.get('worker') or {}).get('source')
            or not _same((r24.get('source_records') or {}).get('records',{}).get('r13'), args.r13_result)):
        raise ValueError('R24 reference counters or training provenance differ')
    probes = worker.get('approximate_probes')
    if (not isinstance(probes,list) or len(probes)!=3
            or [row.get('probe_index') for row in probes] != [0,1,2]
            or [row.get('sample_id') for row in probes] != ['train:0','train:1','train:2']):
        raise ValueError('Exactly three ordered R24 physical32 references are required')
    alignment = _read(args.poly_source_alignment)
    checks = alignment.get('checks') or {}
    if (alignment.get('status') != 'source_arithmetic_aligned'
            or any(checks.get(name) is not True for name in ('MPCB.py:GenPoly','MPCB.py:fint','Poly.py:GenPoly'))
            or alignment.get('coefficient_count') != 96
            or alignment.get('parsed_public_coefficient_values_equal') is not True
            or alignment.get('parsed_integer_tree_equal') is not True
            or alignment.get('all_coefficients_finite') is not True
            or alignment.get('silu_semantics',{}).get('verified_from_source') is not True):
        raise ValueError('Independent original-Poly arithmetic source alignment is missing')
    roots = alignment.get('source_paths') or []
    script = Path((compiled.get('sources') or {}).get('script',{}).get('path',''))
    native_sources = {str(Path(row['path']).resolve()) for row in (worker.get('native_metadata') or {}).get('sources',[])}
    if (len(roots)!=2 or Path(roots[0]).resolve()!=script.parents[2]/'python/poly/poly'
            or str((Path(roots[1])/'MPCB.py').resolve()) not in native_sources
            or not _same(alignment.get('source_reference',{}).get('r24_result'), args.r24_result)):
        raise ValueError('Poly alignment is attached to different original or R24 sources')
    archive = _read(args.archive_manifest)
    bundles = [row for row in archive.get('files',[]) if row.get('kind')=='runtime_bundle']
    original_plan=Path(args.plan).resolve()
    if (not original_plan.name.endswith('.runtime-plan.json') or len(bundles)!=1
            or not _same(args.bundle,bundles[0].get('path',bundles[0].get('source')))
            or not _same(bundles[0].get('path',bundles[0].get('source')),
                original_plan.with_name(original_plan.name.removesuffix('.runtime-plan.json')+'.bundle'))):
        raise ValueError('Closed archive metadata belongs to a different original plan bundle')
    proof = _read(args.scalar_source_proof)
    if not _same(proof.get('source_plan'), args.plan):
        raise ValueError('Scalar source evidence belongs to a different original plan')
    summary = dict(paths={name:(_file(getattr(args,name)) if name!='bundle' else
        dict(path=str(Path(args.bundle).resolve()), kind='directory')) for name in SOURCE_ARGS},
        source_factory=binding.SOURCE_FACTORY, original_checkpoint=original.get('checkpoint'),
        checkpoint_contents_read=False, original_definition=original.get('original_model_source'),
        reference_source='R24 worker.approximate_probes[].approximate; source-aligned original Poly and physical32 units',
        original_reference_times_included=False, original_plan_modified=False,
        original_poly_preserved=True, original_pack_bb32_preserved=True,
        training_preprocessing_verified=False, scalar_semantics='Original HEAAN period1 constant expansion',
        source_only_arithmetic_alignment=True, concrete_native_primes_verified=False,
        packing_provenance=dict(
            benchmark_source=str(script),benchmark_lines=[54,56,58,59,60],
            primitive_source=str(script.parents[2]/'python/poly/poly/MPCB.py'),
            primitive_lines=[86,87,88,89,90,91,629,633,668,676,681,682,683],
            official_io_source=str(script.parents[2]/'examples/tests/ResNet.py'),
            official_io_lines=[47,58,59,65,66,67,75,79],
            input_formula='CHW float64/32; pad3072 to4096; tile16 to65536',
            output_formula='first10 decoded slots times32, shape[1,10]',
            actual_input_schema_inferred_from_saved_probes=True))
    return summary,r13,r24,proof


def load_program(plan_path, manifest_path, source_proof):
    plan, manifest = _read(plan_path), _read(manifest_path)
    metadata = binding.plan_inventory(plan, manifest, source_proof=source_proof, plan_path=plan_path)
    if metadata['source_gates']['per_site_provenance_verified'] is not True:
        raise ValueError('All singleton inline sites require the original periodic encoding source evidence')
    values = {row['id']:row for row in plan['values']}
    encodes = {row['output']:row['payload'] for row in plan['initialization']}
    last = {}
    for index, op in enumerate(plan['execution']):
        for name in op['inputs']:
            if values[name]['kind']=='ciphertext':
                last[name]=index
    return dict(plan=plan,manifest=manifest,values=values,encodes=encodes,last_use=last,
                input=plan['external_inputs'][0],output=plan['final_outputs'][0],metadata=metadata)


def check_restored_bundle(program, bundle, archive_manifest):
    """Check plain path/size inventory, without opening archives or blob bytes."""
    bundle=Path(bundle)
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError('An explicitly restored original runtime bundle is required')
    archive=_read(archive_manifest)
    choices=[r for r in archive.get('files',[]) if r.get('kind')=='runtime_bundle']
    if len(choices)!=1:
        raise ValueError('One original closed bundle inventory is required')
    record=choices[0]
    basename=Path(record.get('path',record.get('source',''))).name
    inventory=record.get('inventory') or []
    expected={}
    for row in inventory:
        relative=Path(row.get('path',''))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!=basename:
            raise ValueError('Archive metadata paths must remain beneath the original bundle name')
        if row.get('kind')=='file':
            expected[str(Path(*relative.parts[1:]))]=row.get('bytes')
        elif row.get('kind')!='directory':
            raise ValueError('Unsupported original bundle metadata item')
    actual={}
    for parent,directories,files in os.walk(bundle,followlinks=False):
        for name in directories+files:
            path=Path(parent)/name; st=path.lstat()
            if path.is_symlink() or not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
                raise ValueError('Restored bundle links/special files are unsupported')
            if stat.S_ISREG(st.st_mode):
                if st.st_nlink!=1:raise ValueError('Restored hard-linked payload is unsupported')
                actual[str(path.relative_to(bundle))]=st.st_size
    if actual!=expected:
        raise ValueError('Restored bundle file/size inventory differs from the closed original record')
    blobs={row['content']:row for row in program['manifest']['blobs']}
    required={'manifest.json'}|{'data/'+binding.content_filename(name) for name in blobs}
    if set(actual)!=required or any(actual['data/'+binding.content_filename(name)]!=row['byte_length'] for name,row in blobs.items()):
        raise ValueError('Restored manifest/blob paths or byte lengths differ')
    return dict(status='metadata_checked',restored_path=str(bundle.resolve()),
        archive_source_path=record.get('path',record.get('source')),file_count=len(actual),blob_count=len(blobs),
        file_bytes=sum(actual.values()),archive_payload_read=False,blob_payload_values_read=0,
        inventory_scope='Regular relative file names and sizes; no byte, digest, CRC or content comparison')


class LazyPlaintexts:
    """One read-only decoded vector cached; no eager RNS/plaintext construction."""
    def __init__(self,bundle,program,slots=SLOTS):
        self.bundle,self.program,self.slots=Path(bundle),program,slots
        self.name=self.current=None
        self.hits=self.loads=self.blob_reads=self.inline_expansions=self.bytes_read=self.evictions=0
        self.peak=0
    def get(self,name):
        if self.current is not None and self.name==name:
            self.hits+=1;return self.current
        if self.current is not None:self.evictions+=1
        self.name=self.current=None
        np=_np(); payload=self.program['encodes'][name]
        if payload['kind']=='inline':
            data=np.full(self.slots,payload['values'][0],dtype=np.float64)
            self.inline_expansions+=1
        else:
            path=self.bundle/'data'/binding.content_filename(payload['content'])
            if path.is_symlink() or path.stat().st_size!=self.slots*8:
                raise ValueError('Lazy original bundle vector size/path differs')
            with path.open('rb') as stream:
                data=np.fromfile(stream,dtype=np.dtype('<f8'),count=self.slots)
            self.blob_reads+=1;self.bytes_read+=int(data.nbytes)
        if data.dtype!=np.dtype('float64') or data.shape!=(self.slots,) or not bool(np.isfinite(data).all()):
            raise ValueError('A constant must be a finite original float64 vector')
        data.flags.writeable=False
        self.name,self.current=name,data
        self.loads+=1;self.peak=max(self.peak,int(data.nbytes))
        return data
    def summary(self):
        return dict(capacity=1,loads=self.loads,hits=self.hits,evictions=self.evictions,
            blob_reads=self.blob_reads,bytes_read=self.bytes_read,inline_expansions=self.inline_expansions,
            resident_vectors=int(self.current is not None),peak_decoded_bytes=self.peak,encoded_plaintexts=0)
    def clear(self):self.name=self.current=None


def _vector(value,slots):
    np=_np()
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype('float64') or value.shape!=(slots,) or not bool(np.isfinite(value).all()):
        raise ValueError('Every ideal decoded value must be finite float64 with the original slot count')
    return value


def execute_program(program,constants,packed,result,progress=None):
    """Pure decoded arithmetic; original instruction order/SSA identifiers stay intact."""
    np=_np();slots=constants.slots
    pool={program['input']:_vector(packed,slots)}
    metadata=program['values'];last=program['last_use'];final=program['output']
    def observe():
        arrays={id(value):value for value in pool.values()}
        result['maximum_live_ciphertext_values']=max(result['maximum_live_ciphertext_values'],len(pool))
        result['maximum_live_numeric_arrays']=max(result['maximum_live_numeric_arrays'],len(arrays))
        result['maximum_live_numeric_bytes']=max(result['maximum_live_numeric_bytes'],sum(x.nbytes for x in arrays.values()))
    observe()
    for index,op in enumerate(program['plan']['execution']):
        kind=op['op'];result['op_attempts']+=1;result['op_attempted_by_kind'][kind]+=1
        a=pool[op['inputs'][0]]
        b=None
        if len(op['inputs'])==2:
            name=op['inputs'][1]
            b=constants.get(name) if metadata[name]['kind']=='plaintext' else pool[name]
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            if kind in IDENTITIES:
                output=a
                result['ideal_identity_nodes'][kind]+=1
                if kind=='boot':
                    result['bootstrap_input_rows'].append(dict(sample_id=result.get('active_sample_id'),
                        ordinal=op['ordinal'],input_id=op['inputs'][0],output_id=op['output'],
                        maximum_absolute=float(np.max(np.abs(a))),mean=float(np.mean(a)),
                        mathematical_identity_only=True,domain_acceptance=None))
            elif kind=='rotate':output=np.roll(a,-op['attrs']['steps'])
            elif kind in ('add_cc','add_cp'):output=np.add(a,b)
            elif kind in ('mul_cc','mul_cp'):output=np.multiply(a,b)
            elif kind=='negate':output=np.negative(a)
            else:raise ValueError('Unsupported ideal operation: '+kind)
        pool[op['output']]=_vector(output,slots);observe()
        result['op_completed']+=1;result['op_completed_by_kind'][kind]+=1
        for name in set(op['inputs']):
            if name in pool and last.get(name)==index and name!=final:del pool[name]
        if op['output'] not in last and op['output']!=final:del pool[op['output']]
        if progress is not None and ((index+1)%256==0 or index+1==len(program['plan']['execution'])):progress()
    if set(pool)!={final}:raise ValueError('Only the original final ciphertext value may remain live')
    return pool[final]


def initial_result():
    result=dict(round=ROUND,model=MODEL,source_factory=binding.SOURCE_FACTORY,mode=MODE,status='running',scope=SCOPE,
        source_checked=False,original_training_math_verified=False,full_execution_scope_ready=False,
        original_plan_modified=False,probe_rows=[],bootstrap_input_rows=[],constant_reads=None,
        maximum_live_ciphertext_values=0,maximum_live_numeric_arrays=0,maximum_live_numeric_bytes=0,
        op_attempted_by_kind=dict.fromkeys(binding.OPS,0),op_completed_by_kind=dict.fromkeys(binding.OPS,0),
        ideal_identity_nodes=dict.fromkeys(IDENTITIES,0),bootstrap_domain_acceptance=None,
        backend_ready=False,security_validated=False,accuracy_validated=False,formal_matrix_complete=False)
    result.update(dict.fromkeys(COUNTERS,0));result.update(dict.fromkeys(ZERO_FIELDS,0))
    result.update(dict.fromkeys(FORMAL_FIELDS,None))
    return result


def _check_budget(args,started):
    if time.perf_counter_ns()-started>=args.seconds*10**9:raise TimeoutError('Ideal-math worker wall allowance exhausted')
    total=artifact_bytes(args.work_dir.parent)
    if total>args.case_gib*GIB:raise RuntimeError('Ideal-math case artifact allowance exhausted')
    if shutil.disk_usage(args.work_dir).free<10*GIB:raise RuntimeError('Disk reserve below10GiB')


def run_checks(args,recorder,result):
    started=time.perf_counter_ns();args.work_dir=Path(args.work_dir)
    def checkpoint():
        _check_budget(args,started)
        write_json(args.work_dir/'worker-result.json',result)
    with recorder.span('source.saved_records'):
        summary,r13,r24,proof=check_sources(args)
        result.update(source_summary=summary,source_checked=True)
    with recorder.span('source.original_plan_and_manifest'):
        result['plan_load_attempts']+=1;result['manifest_load_attempts']+=1;checkpoint()
        program=load_program(args.plan,Path(args.bundle)/'manifest.json',proof)
        for name,expected in EXPECTED.items():
            if program['metadata'].get(name)!=expected:raise ValueError('Original plan inventory differs: '+name)
        result['program_metadata']=program['metadata']
        result['plan_load_completed']+=1;result['manifest_load_completed']+=1
        result['restored_bundle_metadata']=check_restored_bundle(program,args.bundle,args.archive_manifest)
    with recorder.span('source.saved_three_training_probes'):
        result['probe_load_attempts']+=1;checkpoint()
        images,labels,source=reference._load_probes(args,r13,result)
        references=reference._cached_references(r24)
        result['probe_source']=source;result['probe_load_completed']+=1
    constants=LazyPlaintexts(args.bundle,program)
    try:
        for index in range(3):
            result.update(active_probe_index=index,active_sample_id=source['sample_ids'][index])
            with recorder.span('original.decoded_ideal_forward',sample_id=source['sample_ids'][index]):
                result['packed_forward_attempts']+=1;checkpoint()
                packed=reference.pack_input(images[index])
                input_path=args.work_dir/('packed-input-train-'+str(index)+'.f64')
                packed.astype('<f8',copy=False).tofile(input_path)
                output=execute_program(program,constants,packed,result,progress=checkpoint)
                decoded=reference.decode_output(output)
                result['packed_forward_completed']+=1
            with recorder.span('original.strict_math',sample_id=source['sample_ids'][index]):
                result['math_check_attempts']+=1
                compared=reference._check_math(references[index],decoded)
                result['math_check_completed']+=1
                result['probe_rows'].append(dict(probe_index=index,sample_id=source['sample_ids'][index],
                    label=int(labels[index]),decoded=decoded.tolist(),reference=references[index].tolist(),
                    packed_input=_file(input_path),math=compared))
                result['constant_reads']=constants.summary();checkpoint()
                if not compared['passed']:raise ValueError('Original RuntimePlan decoded arithmetic fails the original strict mathematical criterion')
                del output,decoded,packed
        expected={k:3*v for k,v in EXPECTED['op_counts'].items()}
        if result['op_completed']!=3*EXPECTED['execution_count'] or result['op_attempted_by_kind']!=expected or result['op_completed_by_kind']!=expected:
            raise ValueError('Every original operation must run exactly once on each of the three probes')
        math_proof=dict(status='strict_math_passed',source_factory=binding.SOURCE_FACTORY,
            source_plan=str(Path(args.plan).resolve()),sample_ids=['train:0','train:1','train:2'],
            original_poly_preserved=True,original_pack_bb32_preserved=True,original_plan_modified=False,
            absolute_tolerance=1e-12,relative_tolerance=1e-12,all_three_passed=True,
            reference_result=str(Path(args.r24_result).resolve()),scope=SCOPE,
            crypto_executions=0,bootstrap_domain_acceptance=None)
        write_json(args.work_dir/'source-math-proof.json',math_proof)
        result.update(status='ok',original_training_math_verified=True,
            full_execution_scope_ready=True,source_math_proof=str(args.work_dir/'source-math-proof.json'))
    finally:
        result['constant_reads']=constants.summary();constants.clear()


def worker(args):
    args.work_dir=Path(args.work_dir);args.work_dir.mkdir(parents=True,exist_ok=False)
    result=initial_result();recorder=StageRecorder(args.work_dir/'events.jsonl');started=time.perf_counter_ns()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    original_handler=signal.getsignal(signal.SIGALRM);original_alarm=signal.getitimer(signal.ITIMER_REAL)
    def expired(signum,frame):raise TimeoutError('Ideal-math worker wall allowance exhausted')
    try:
        memory=int(args.memory_gib*GIB);_,hard=resource.getrlimit(resource.RLIMIT_AS)
        if hard!=resource.RLIM_INFINITY:memory=min(memory,hard)
        resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,args.seconds)
        with recorder.span('dacapo.original_plan_ideal_math'):run_checks(args,recorder,result)
    except BaseException as exc:
        result.update(status='error',error_type=type(exc).__name__,error=str(exc)[:3000],
            original_training_math_verified=False,full_execution_scope_ready=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,original_handler)
        if original_alarm[0]:signal.setitimer(signal.ITIMER_REAL,*original_alarm)
        result.update(elapsed_ns=time.perf_counter_ns()-started,
            worker_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(args.work_dir/'worker-result.json',result)
    return result


def _worker_env():
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
         and k not in ('HECATE','PYTHONHOME','PYTHONPATH','PYTHONSTARTUP','PYTHONUSERBASE','PYTHONINSPECT')}
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2]);env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONNOUSERSITE']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    return env


@contextmanager
def _artifact_guard(out,case_gib):
    stop=threading.Event();errors=[]
    old=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise RuntimeError('Ideal-math outer interrupted by resource monitor')
    signal.signal(signal.SIGTERM,interrupted)
    def monitor():
        while not stop.wait(.5):
            try:
                size=artifact_bytes(out);free=shutil.disk_usage(out).free
                with (out/'artifact-resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(ts_ns=time.perf_counter_ns(),artifact_bytes=size,disk_free_bytes=free))+'\n')
                if size>case_gib*GIB or free<10*GIB:raise RuntimeError('Case artifact allowance or disk reserve exhausted')
            except BaseException as exc:
                errors.append(str(exc));os.kill(os.getpid(),signal.SIGTERM);return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:yield errors
    finally:stop.set();thread.join();signal.signal(signal.SIGTERM,old)


def execute(args):
    from .. import paths
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter_ns();recorder=StageRecorder(out/'events.jsonl');revision='unknown'
    result=dict(round=ROUND,model=MODEL,mode=MODE,status='running',worker=None,worker_attempts=0,worker_completed=0,
        actual_source_reads=None,original_training_math_verified=False,full_execution_scope_ready=False,
        **dict.fromkeys(FORMAL_FIELDS,None))
    try:
        with recorder.span('outer.explicit_source_paths'):
            result['selected_paths']={name:str(Path(getattr(args,name)).resolve()) for name in SOURCE_ARGS}
            for name in SOURCE_ARGS:
                p=Path(getattr(args,name))
                if not (p.is_dir() if name=='bundle' else p.is_file()):raise FileNotFoundError(str(p))
            if not Path(args.python).is_file():raise FileNotFoundError(str(args.python))
            if shutil.disk_usage(out).free<10*GIB:raise RuntimeError('Disk reserve below10GiB')
            snapshots=out/'source';snapshots.mkdir()
            for source in (Path(__file__),Path(binding.__file__),Path(reference.__file__)):
                shutil.copyfile(source,snapshots/source.name)
        with recorder.span('outer.git_revision'):
            observed=run_process(['git','rev-parse','HEAD'],cwd=paths.REPO,log=out/'git.log',
                resources=out/'git-resources.jsonl',timeout=15,mem_gib=.5,fsize_gib=.1)
            if _clean(observed):revision=(out/'git.log').read_text().strip()
        argv=[str(args.python),'-B','-m','pt_agent.experiments.dacapo_native_plan_math','--worker',
            '--work-dir',str(out/'worker'),'--seconds',str(args.seconds),'--memory-gib',str(args.memory_gib),
            '--case-gib',str(args.case_gib)]
        for name in SOURCE_ARGS:argv.extend(['--'+name.replace('_','-'),str(Path(getattr(args,name)).resolve())])
        result['worker_argv']=argv
        with _artifact_guard(out,args.case_gib) as violations:
            with recorder.span('outer.original_ideal_math_worker'):
                result['worker_attempts']=1;write_json(out/'result.json',result)
                process=run_process(argv,cwd=Path(__file__).resolve().parents[2],log=out/'worker.log',
                    resources=out/'resources.jsonl',timeout=args.seconds+30,mem_gib=args.memory_gib,
                    fsize_gib=1,env=_worker_env())
                result['process']=process
                target=out/'worker/worker-result.json'
                result['worker']=_read(target) if target.is_file() else None
                worker_result=result['worker'] or {}
                if violations or not _clean(process) or worker_result.get('status')!='ok':
                    raise RuntimeError('Original DaCapo ideal-math worker failed; partial evidence retained')
                if (worker_result.get('round')!=ROUND or worker_result.get('mode')!=MODE
                        or worker_result.get('source_checked') is not True
                        or worker_result.get('original_training_math_verified') is not True
                        or worker_result.get('full_execution_scope_ready') is not True
                        or any(worker_result.get(k)!=0 for k in ZERO_FIELDS)
                        or any(worker_result.get(k) is not None for k in FORMAL_FIELDS)
                        or worker_result.get('packed_forward_completed')!=3
                        or worker_result.get('math_check_completed')!=3
                        or worker_result.get('op_completed')!=3*EXPECTED['execution_count']):
                    raise RuntimeError('Ideal-math terminal protocol is incomplete')
                result.update(status='ok',worker_completed=1,original_training_math_verified=True,full_execution_scope_ready=True)
    except BaseException as exc:result.update(status='error',error_type=type(exc).__name__,error=str(exc)[:3000])
    finally:
        if result['worker'] is None and (out/'worker/worker-result.json').is_file():
            try:result['worker']=_read(out/'worker/worker-result.json')
            except (OSError,ValueError):pass
        worker_result=result['worker'] or {}
        terminal_worker=(worker_result.get('status') in ('ok','error')
            and type(worker_result.get('elapsed_ns')) is int)
        result['partial_checkpoint_counts']={k:worker_result.get(k) for k in COUNTERS}
        result['actual_source_reads']={k:(worker_result.get(k) if terminal_worker else None) for k in COUNTERS}
        result['worker_counter_scope']=('Terminal worker counters' if terminal_worker else
            'No terminal worker record; checkpoint counts are observed lower bounds and actual totals remain null')
        timing=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns());result['timing']=timing
        result['timing_scope']='recorded_elapsed_seconds and phase summary end before terminal persistence; completion.json records the full endpoint after required artifacts and registry; saved source preparation times excluded'
        result['recorded_elapsed_seconds']=(time.perf_counter_ns()-started)/1e9
        result['git_rev']=revision;write_json(out/'result.json',result)
        metrics={k:result.get(k) for k in ('status','worker_attempts','worker_completed','original_training_math_verified','full_execution_scope_ready',*FORMAL_FIELDS)}
        metrics.update(result['actual_source_reads'])
        row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=revision,metrics=metrics,
            artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        registry=Path(args.registry);registry.parent.mkdir(parents=True,exist_ok=True)
        with registry.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):raise RuntimeError('Run id already exists')
            stream.seek(0,os.SEEK_END);stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=timing,required_artifact_bytes=artifact_bytes(out))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--registry',type=Path);parser.add_argument('--python',type=Path)
    for name in SOURCE_ARGS:parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=900);parser.add_argument('--memory-gib',type=float,default=4)
    parser.add_argument('--case-gib',type=float,default=4)
    args=parser.parse_args(argv)
    if not 0<args.seconds<=900 or not 0<args.memory_gib<=4 or not 0<args.case_gib<=4:
        parser.error('This ideal-math batch is bounded by900seconds/4GiB memory/4GiB artifacts')
    if args.worker:
        if args.work_dir is None:parser.error('Worker requires explicit --work-dir')
        result=worker(args)
    else:
        from .. import paths
        if args.out is None:parser.error('--out is required')
        args.registry=args.registry or paths.RESULTS_DIR/'registry.jsonl'
        args.python=args.python or paths.ORBIT_ROOT.parent/'orbit-venv/bin/python'
        result=execute(args)
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
