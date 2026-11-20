"""Prepare the explicit P57 independent39-cell native profile and references.

Preparation uses a deterministic public microbenchmark, protocol metadata and
two clear arithmetic paths. It creates no keys or ciphertexts and never runs
the native driver. Original36-cell/P61 entries and artifacts remain intact.
"""
import argparse
from array import array
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time

from .r10_paper_metrics import write_json
from .r31_lattigo_binding import NativeBinding
from .r31_lattigo_profile import make_profile_plan
from .r31_lattigo_profile_reference import compare_references, direct_profile_sum, replay_profile
from .r31_lattigo_profile_subset import EXPECTED_CELLS, deterministic_fixture
from .r31_lattigo_runtime_plan import (
    auxiliary_arguments, c2s_arguments, checkpoint_arguments, content_filename, plan_inventory,
)

MODE = 'independent-p57-native39-profile-preparation'
SLOTS = 16384
CACHE_CAPACITY = 32
CASE_BYTES = 32*1024**2
SECONDS = 60
EXPECTED_OPS = dict(mod_switch=42,boot=1,mul_cp=8,rescale=16,add_cc=42,
    add_cp=4,rotate=5,mul_cc=8,relinearize=8,negate=1)


def binding(c2s_bits=None):
    return NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,
        bootstrap_normalization_log2=3,auxiliary_prime_bits=57 if c2s_bits is None else 61,
        c2s_bits=c2s_bits)


def mode_for(c2s_bits=None):
    return MODE if c2s_bits is None else 'independent-c2s52-p61-native39-profile-preparation'


def _key(cell):
    return json.dumps({key:cell[key] for key in ('op','inputs','output','attrs')},
        sort_keys=True,separators=(',',':'))


def _operand(level,scale=50,components=2,kind='ciphertext'):
    return dict(kind=kind,level=level,scale_log2=scale,components=components)


def _canonical_base():
    """Accepted36 protocol identities, independent of full-model values."""
    rows=[]
    def add(op,inputs,output,attrs=None,steps=None):
        attrs=attrs or {};representative=dict(attrs)
        if steps is not None:representative['steps']=steps
        rows.append(dict(op=op,inputs=inputs,output=output,attrs=attrs,representative_attrs=representative))
    add('boot',[_operand(1)],_operand(5),dict(implementation='native',target_level=5,
        target_scale_log2=50,target_components=2))
    for level in range(2,6):
        add('rescale',[_operand(level,100)],_operand(level-1),dict(target_level=level-1,target_scale_log2=50))
        add('relinearize',[_operand(level,100,3)],_operand(level,100))
        add('mul_cc',[_operand(level),_operand(level)],_operand(level,100,3))
    for level,target in ((2,1),(3,2),(4,2),(5,2),(4,3),(5,3),(5,4)):
        add('mod_switch',[_operand(level)],_operand(target),dict(target_level=target))
    for level,steps in ((1,4),(2,1024),(3,1024),(4,16351)):
        add('add_cc',[_operand(level),_operand(level)],_operand(level))
        add('add_cp',[_operand(level),_operand(level,components=1,kind='plaintext')],_operand(level))
        add('rotate',[_operand(level)],_operand(level),steps=steps)
    for level in (2,3,4):
        add('mul_cp',[_operand(level),_operand(level,components=1,kind='plaintext')],_operand(level,100))
    add('negate',[_operand(3)],_operand(3))
    return rows


def select_cells(base,missing):
    if base.get('cell_count')!=36 or len(base.get('cells',[]))!=36:
        raise ValueError('Retain the accepted36-cell metadata source')
    if missing.get('cell_count')!=3 or len(missing.get('cells',[]))!=3:
        raise ValueError('Retain exactly the three independently measured missing cells')
    rows=copy.deepcopy(base['cells']+missing['cells'])
    actual={_key(row):row for row in rows}
    expected_rows=_canonical_base()+[dict(row,representative_attrs=(dict(steps=1023)
        if row['op']=='rotate' else copy.deepcopy(row['attrs']))) for row in EXPECTED_CELLS]
    expected={_key(row):row for row in expected_rows}
    if len(actual)!=39 or set(actual)!=set(expected):
        raise ValueError('The new profile must preserve the exact accepted39 metadata identities')
    for key,row in actual.items():
        wanted=expected[key];attrs=row.get('representative_attrs',{})
        if any(type(value['level']) is not int or type(value['scale_log2']) is not int
               or type(value['components']) is not int for value in row['inputs']+[row['output']]):
            raise ValueError('Cell dimensions must retain integer protocol metadata')
        if any(attrs.get(k)!=v for k,v in row['attrs'].items()):
            raise ValueError('Representative target attrs differ from the measured cell')
        if row['op']=='rotate' and (type(attrs.get('steps')) is not int
                or attrs['steps']!=wanted['representative_attrs']['steps']):
            raise ValueError('Preserve the recorded representative rotation per cell')
    return [actual[key] for key in sorted(actual)]


def make_catalog_source(cells,*,c2s_bits=None):
    values=[];execution=[]
    for index,cell in enumerate(cells):
        inputs=[]
        for j,meta in enumerate(cell['inputs']):
            name=f'catalog-{index}-input-{j}';inputs.append(name)
            values.append(dict(id=name,context='qwen-bootstrap48',**copy.deepcopy(meta)))
        output=f'catalog-{index}-output'
        values.append(dict(id=output,context='qwen-bootstrap48',**copy.deepcopy(cell['output'])))
        execution.append(dict(op=cell['op'],ordinal=index,inputs=inputs,output=output,
            attrs=copy.deepcopy(cell['representative_attrs'])))
    return dict(target=dict(target_id='pt-agent-lattigo-native-adapter',adapter_variant=binding(c2s_bits).variant(CACHE_CAPACITY),
        precision_bits=50,independent_profile_catalog=True,source_model_execution=False),
        values=values,execution=execution)


def build_profile(base,missing,*,c2s_bits=None):
    cells=select_cells(base,missing)
    selected=binding(c2s_bits)
    plan,manifest,metadata=make_profile_plan(make_catalog_source(cells,c2s_bits=c2s_bits),binding=selected)
    manifest['bundle_id']=('r31-independent-native39-p57-public-fixture' if c2s_bits is None
                           else 'r31-independent-native39-c2s52-p61-public-fixture')
    plan['plaintext_bundle']['id']=manifest['bundle_id']
    inventory=plan_inventory(plan,manifest,binding=selected)
    if (metadata['cell_count']!=39 or metadata['setup_and_aggregation_count']!=96
            or inventory['execution_count']!=135 or inventory['encode_count']!=9
            or inventory['op_counts']!=EXPECTED_OPS or inventory['bootstrap_count']!=1):
        raise ValueError('The accepted39 cells must produce135 operations and9 plaintexts')
    metadata.update(mode=mode_for(c2s_bits),native_profile_group=selected.variant(CACHE_CAPACITY),
        bootstrap_cache_capacity=CACHE_CAPACITY,residual_cache_capacity=1,plaintext_cache_capacity=1,
        source_model_execution=False,old_target_durations_reused=False,
        fixture_scope='Public deterministic(i%17-8)/32 across16384 slots; no model/calibration/quality data',
        target_observation_count=1,per_cell_repeated_variance=None,
        actual_native_executions=0,security_acceptance=None)
    return plan,manifest,metadata,inventory


def validate_prepared(record,*,c2s_bits=None):
    wanted=binding(c2s_bits);bits=wanted.auxiliary_prime_bits
    client=record.get('client') or {};requested=record.get('requested_native_parameters') or {}
    if (record.get('status')!='ok' or record.get('phase')!='prepare' or record.get('client_cache_protocol',0)<9
            or bits not in record.get('supported_auxiliary_prime_bits',[])
            or client.get('status')!='ok' or client.get('mode')!='parameters' or client.get('precision_bits')!=50
            or client.get('bootstrap_normalization_log2')!=3 or client.get('bootstrap_input_divisor')!=8
            or requested.get('log_p')!=[bits]*3 or requested.get('bootstrap_log_p')!=[bits]*4
            or requested.get('actual_constructor_primes_pending') is not False
            or any(client.get(key)!=0 for key in ('keygen_api_attempts','encryption_attempts','compute_attempts','final_decryption_attempts'))):
        raise ValueError('An actual isolated selected-group parameter-only preparation with zero crypto is required')
    constructor=client.get('bootstrap_parameters') or {}
    residual=constructor.get('ResidualParameters') or {};bootstrap=constructor.get('BootstrappingParameters') or {}
    mod1=constructor.get('Mod1ParametersLiteral') or {}
    if (residual.get('LogN')!=16 or bootstrap.get('LogN')!=16
            or residual.get('LogDefaultScale')!=50 or bootstrap.get('LogDefaultScale')!=50
            or len(residual.get('P',[]))!=3 or len(bootstrap.get('P',[]))!=4
            or mod1.get('Mod1InvDegree')!=15 or mod1.get('LogMessageRatio')!=2
            or mod1.get('Mod1Degree')!=30 or mod1.get('DoubleAngle')!=3 or mod1.get('K')!=16):
        raise ValueError('Actual P57 constructor policy or prime dimensions differ')
    for primes in (residual['P'],bootstrap['P']):
        if (len(set(primes))!=len(primes) or any(type(x) is not int or x<3 or x%2!=1
                or not bits-.1<math.log2(x)<bits+.1 for x in primes)):
            raise ValueError('Retain actual auxiliary primes from the explicit selected constructor')
    if c2s_bits is None:
        if residual['P']!=bootstrap['P'][:3]:
            raise ValueError('The original P57 group retains its first three bootstrap P primes')
    elif (not set(residual['P']).issubset(bootstrap['P']) or residual['P'][0]!=bootstrap['P'][0]):
        raise ValueError('C2S52 residual auxiliary basis must be a subset with the same low-modulus P0')
    # The native wrapper extends both bases from sk.Value.Q. Its independently
    # constructed residual P basis may occur at bootstrap indices0,1,3.

    if c2s_bits is not None:
        c2s_arguments(record,c2s_bits)
        if (record.get('variant')!=wanted.variant(CACHE_CAPACITY)
                or client.get('variant')!=wanted.variant(CACHE_CAPACITY)
                or client.get('c2s_bits')!=52 or client.get('bootstrap_cache_capacity')!=32
                or requested.get('bootstrap_coeffs_to_slots_factorization_depth_and_log_scales')!=[[52]]*4
                or constructor.get('CoeffsToSlotsParameters',{}).get('Levels')!=[1]*4
                or constructor.get('SlotsToCoeffsParameters',{}).get('Levels')!=[1]*3
                or len(residual.get('Q',[]))!=9 or len(bootstrap.get('Q',[]))!=28
                or residual['Q']!=bootstrap['Q'][:9]
                or any(abs(math.log2(x)-52)>.01 for x in bootstrap['Q'][-4:])
                or math.fsum(math.log2(x) for x in bootstrap['Q']+bootstrap['P'])>1747):
            raise ValueError('Actual C2S52x4/P61 public constructor dimensions or complete modulus differ')
    binary=Path(record['ready_binary'])
    if not binary.is_file():raise ValueError('The accepted prepared native binary must exist')
    auxiliary_arguments(record,bits);checkpoint_arguments(record,1000)
    return dict(binary=str(binary),variant=wanted.variant(CACHE_CAPACITY),
        residual_P=residual['P'],bootstrap_P=bootstrap['P'],actual_mod1=mod1,
        source_preparation_variant=record.get('variant'),actual_constructor_validated=True,
        secret_distribution=bootstrap.get('Xs'),error_distribution=bootstrap.get('Xe'),
        security_acceptance=None,scope='Actual constructor metadata and explicit policy only; no primality test, hash or security certificate')


def execution_command(preparation,*,out,prepared,c2s_bits=None):
    command=[sys.executable,'-B','-m','pt_agent.experiments.r31_lattigo_runtime_plan','execute',
        '--out',str(out),'--prepared',str(prepared),'--plan',preparation['output_plan'],
        '--bundle',preparation['output_bundle'],'--input',preparation['input'],'--expected',preparation['expected'],
        '--allow-bootstrap','--bootstrap-cache','32','--precision-bits','50','--mod1-inv-degree','15',
        '--log-message-ratio','2','--bootstrap-normalization-log2','3','--auxiliary-prime-bits',str(binding(c2s_bits).auxiliary_prime_bits),
        '--check-input-ssa','--trace-key-requests','--checkpoint-interval','1000','--seconds','900',
        '--native-address-space-gib','24','--native-rss-gib','14','--minimum-available-gib','1.25',
        '--native-go-memory-limit','11GiB','--case-gib','4']
    if c2s_bits is not None:command+=['--c2s-bits',str(c2s_bits)]
    return command


def prepare(args):
    began=time.perf_counter_ns();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    c2s_bits=getattr(args,'c2s_bits',None)
    result=dict(mode=mode_for(c2s_bits),status='running',error=None,native_executions=0,keygen_runs=0,
        encryption_runs=0,bootstrap_runs=0,decryption_runs=0,model_forwards=0,calibration_reads=0,
        heldout_reads=0,confirm_reads=0,report_reads=0,llm_calls=0,downloads=0,installations=0,
        extra_integrity_checks=0,security_acceptance=None,formal_latency=None,artifact_bytes=0)
    old=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):raise TimeoutError('Public39-cell reference preparation wall allowance exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,SECONDS)
    def check():
        if time.perf_counter_ns()-began>SECONDS*10**9:raise TimeoutError('Public profile preparation expired')
        size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if size>CASE_BYTES or shutil.disk_usage(out).free<10*1024**3:raise RuntimeError('Public profile preparation disk allowance exhausted')
    try:
        source=validate_prepared(json.loads((args.prepared/'result.json').read_text()),c2s_bits=c2s_bits)
        base=json.loads(args.base_metadata.read_text());missing=json.loads(args.missing_metadata.read_text())
        plan,manifest,metadata,inventory=build_profile(base,missing,c2s_bits=c2s_bits);check()
        packed=deterministic_fixture();constants={row['content']:array('d',[row['value']])*SLOTS for row in metadata['constant_vectors']}
        replay,summary=replay_profile(plan,constants,packed);check()
        direct=direct_profile_sum(metadata,packed,precision_bits=50)
        comparison=compare_references(direct,replay)
        if comparison['passed'] is not True:raise ValueError('Direct39-cell and instruction arithmetic differ')
        bundle=out/'profile.bundle';(bundle/'data').mkdir(parents=True)
        for content,values in constants.items():
            data=array('d',values)
            if sys.byteorder!='little':data.byteswap()
            (bundle/'data'/content_filename(content)).write_bytes(data.tobytes());check()
        raw=array('d',packed)
        if sys.byteorder!='little':raw.byteswap()
        (out/'input.bin').write_bytes(raw.tobytes())
        write_json(out/'profile-plan.json',plan);write_json(bundle/'manifest.json',manifest)
        write_json(out/'profile-metadata.json',metadata);write_json(out/'reference-summary.json',summary)
        write_json(out/'expected.json',dict(indices=list(range(SLOTS)),values=list(direct),
            scope='New '+('P57' if c2s_bits is None else 'C2S52x4/P61')+' independent39-cell public reference; all16384 final slots; clear bootstrap identity',
            mathematical_comparison=comparison,model_quality_measurement=False))
        result.update(status='prepared',prepared_parameter_source=source,
            base_metadata=str(args.base_metadata.resolve()),missing_metadata=str(args.missing_metadata.resolve()),
            cell_count=39,execution_count=135,setup_and_aggregation=96,plaintext_count=9,
            bootstrap_cache_capacity=32,residual_cache_capacity=1,plaintext_cache_capacity=1,
            inventory=inventory,reference_comparison=comparison,input=str(out/'input.bin'),
            output_plan=str(out/'profile-plan.json'),output_bundle=str(bundle),expected=str(out/'expected.json'),
            metadata=str(out/'profile-metadata.json'),source_group=('P57 residual3/bootstrap4' if c2s_bits is None
                else 'C2S52x4/P61 residual3/bootstrap4')+', precision50/Inv15/MR2/div8',
            fixture_scope=metadata['fixture_scope'],per_cell_repeated_variance=None)
        result['future_native_argv']=execution_command(result,out=args.native_out,prepared=args.prepared,c2s_bits=c2s_bits)
        check()
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        result['elapsed_ns']=time.perf_counter_ns()-began
        for _ in range(12):
            write_json(out/'preparation.json',result)
            size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            if size>CASE_BYTES:result.update(status='error',error='Terminal public profile preparation exceeded disk allowance')
            if size==result['artifact_bytes']:break
            result['artifact_bytes']=size
        else:raise RuntimeError('Public profile preparation artifact size did not converge')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base-metadata','missing-metadata','prepared','out','native-out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--c2s-bits',type=int,choices=(52,))
    args=p.parse_args(argv);result=prepare(args)
    print(json.dumps(dict(status=result['status'],error=result['error'],cell_count=result.get('cell_count'))))
    return 0 if result['status']=='prepared' else 1


if __name__=='__main__':raise SystemExit(main())
