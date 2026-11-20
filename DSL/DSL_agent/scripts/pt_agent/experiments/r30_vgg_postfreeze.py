"""Post-freeze compilation and one independent report for audited hybrid VGG.

Private function globals reuse the trusted R30 phase implementation. Original
modules, the failed cold case and its successful worker evidence stay intact.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from types import FunctionType

from .. import paths
from . import r30_alex_postfreeze as shared
from . import r30_continue_common_session as continuation
from . import r30_frozen_cnn_report as report
from . import vgg_continuation_terminal_audit as terminal_audit
from .r30_continue_alex_session import same_state

ROUND='R30-VGG-POSTFREEZE'
MODE='audited-hybrid-vgg-trusted-postfreeze-compile'
MODEL=continuation.MODEL
FACTORY=continuation.FACTORY
MODULE='pt_agent.experiments.r30_vgg_postfreeze'


def _clone(module):
    namespace=dict(vars(module))
    for key,value in tuple(namespace.items()):
        if isinstance(value,FunctionType)and value.__module__==module.__name__:
            clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__;namespace[key]=clone
    return namespace


def frozen_source(source_case,audit_case):
    """Only saved JSON/source/metadata; no arrays, Provider or confirmation run."""
    from ..planners.transformer_session import load_state
    from .r13_common_models import FACTORIES
    case=Path(source_case).resolve();audit_case=Path(audit_case).resolve()
    if case==audit_case:raise ValueError('The independent audit and retained continuation must differ')
    checked=shared.read(audit_case/'result.json');closed=shared.read(audit_case/'completion.json')
    proof=checked.get('proof')or{}
    if (checked.get('round')!=terminal_audit.ROUND or checked.get('mode')!=terminal_audit.MODE
            or checked.get('model')!=MODEL or checked.get('status')!='ok' or checked.get('error')is not None
            or closed.get('status')!='ok' or closed.get('registry_persisted')is not True
            or closed.get('required_artifacts_persisted')is not True
            or any(checked.get(key)is not True for key in ('formal_frozen','quality_acceptance_passed','strict_math_passed'))
            or any(checked.get(key)!=0 for key in ('model_runs','api_calls','new_confirmations','report_samples','compiler_runs','real_ckks_runs'))
            or checked.get('source_case_modified')is not False
            or not shared.same(checked.get('source_case'),case)
            or not shared.same(proof.get('source_result'),case/'result.json')
            or proof.get('strict_math_checks')!=6 or proof.get('confirmation_runs_verified')!=1
            or proof.get('original_api_calls')!=4 or proof.get('continued_api_calls')!=2
            or proof.get('total_session_calls')!=6 or proof.get('selected_AST_verified')is not True
            or proof.get('JSON_confirmation_semantics_verified')is not True):
        raise ValueError('Require the completed independent frozen VGG readback audit')
    original=shared.read(case/'result.json');marker=shared.read(case/'completion.json')
    if (original.get('round')!=continuation.ROUND or original.get('mode')!=continuation.MODE
            or original.get('model')!=MODEL or original.get('status')!='error'
            or original.get('error')!=terminal_audit.ERROR or not shared.clean(original.get('process')or{})
            or original.get('resource_monitor_error')is not None or original.get('worker_attempts')!=1
            or original.get('worker_completed')!=0 or original.get('actual_new_api_calls')!=2
            or original.get('original_api_calls')!=4 or marker.get('status')!='error'
            or marker.get('required_artifacts_persisted')is not True or marker.get('registry_persisted')is not True):
        raise ValueError('Retain the exact original closed JSON readback failure and clean worker')
    source=original['source'];cold=Path(source['original_case']).resolve();agent=case/'work/agent'
    worker=shared.read(agent/'worker-result.json');state=load_state(agent/'SESSION.json')
    if (not same_state(original.get('worker'),worker) or state.get('numeric_lock',False)is not False
            or worker.get('error')is not None
            or continuation.check_worker(worker,state,source,2,agent_dir=agent)!='ok'):
        raise ValueError('The retained frozen worker and complete SESSION differ')
    program=Path(worker['selected_program']).resolve()
    info_path=cold/'work/intake/model_info/model_info.json';calibration=cold/'work/calibration/calibration.json'
    if (program!=agent/'program.ptdsl' or not shared.same(source.get('model_info'),info_path)
            or not shared.same(source.get('calibration'),calibration)
            or any(not shared.same(proof.get(k),v)for k,v in
                [('source_case',case),('original_case',cold),('worker_result',agent/'worker-result.json'),
                 ('selected_program',program),('model_info',info_path),('calibration',calibration)])):
        raise ValueError('Bind the independent audit to this exact original intake and frozen source')
    info=shared.read(info_path);identity=info.get('source')or{};expected=FACTORIES[FACTORY]
    if (identity.get('factory')!=FACTORY or identity.get('type')!='torch'
            or info.get('task')!='classification' or info.get('metric')!='classification_top1'
            or info.get('threshold')!={'acc_drop_pp':.5} or info.get('split',{}).get('kind')!='legacy_cifar'
            or any(info['split'].get(k)!=list(range(a,b))for k,a,b in
                [('heldout',9000,10000),('confirm',8000,9000),('report',0,8000)])):
        raise ValueError('Preserve the original VGG task, policy and fixed CIFAR partitions')
    for field,key in [('original_source','source'),('checkpoint','checkpoint'),
                      ('inference_preprocessing_source','inference_preprocessing_source')]:
        if not shared.same(identity.get(field),expected[key]):raise ValueError('Original VGG source changed: '+field)
        shared._regular(expected[key])
    normalization=report.official_normalization(expected['inference_preprocessing_source'])
    if identity.get('normalization')!=normalization or identity.get('preprocessing_provenance')!='official inference test':
        raise ValueError('Retain the documented original inference normalization')
    intake=shared.read(cold/'work/intake/phase_result.json')
    if (intake.get('model')!=MODEL or intake.get('status')!='ok'
            or any(intake.get(k)is not True for k in ('import_completed','real_probes_passed','graph_math_passed','roundtrip_passed','static_coverage'))
            or intake.get('graph_math',{}).get('passed')is not True
            or len(intake.get('graph_math',{}).get('samples')or[])!=3
            or any(row.get('passed')is not True or row.get('finite')is not True for row in intake['graph_math']['samples'])):
        raise ValueError('Retain successful original intake and its three-probe proof')
    cal=shared.read(cold/'work/calibration/phase_result.json');provenance=shared.read(cold/'work/calibration/calibration-source.json')
    indices=provenance.get('indices')or[]
    if (cal.get('calibration_completed')is not True or cal.get('calibration_samples')!=1000
            or cal.get('calibration_batch')!=1 or provenance.get('split')!='train' or provenance.get('count')!=1000
            or len(indices)!=1000 or len(set(indices))!=1000
            or any(type(x)is not int or not 0<=x<50000 for x in indices)
            or provenance.get('normalization')!=normalization):
        raise ValueError('Preserve the one original1000-training calibration')
    probes=cold/'work/intake/training_probes.npz';probe_source=cold/'work/intake/training_probe_source.json'
    probe=shared.read(probe_source)
    if probe.get('split')!='train' or probe.get('indices')!=[0,1,2]or probe.get('sample_ids')!=['train:0','train:1','train:2']:
        raise ValueError('Retain the original three saved training probes')
    historical=shared.read(cold/'result.json');old_marker=shared.read(cold/'completion.json')
    if (historical.get('model')!=MODEL or historical.get('round')!='R30' or historical.get('status')!='error'
            or old_marker.get('status')!='error' or old_marker.get('registry_persisted')is not True
            or old_marker.get('required_artifacts_persisted')is not True):
        raise ValueError('Keep the original closed cold failure and its clock')
    metadata=historical['source'];profile=Path(metadata['compiler_profile']);template=cold/'source/compile-metadata-template.json'
    if shared.read(profile)!=shared.read(cold/'source/compiler-profile.json'):
        raise ValueError('The saved original compiler profile changed')
    records=Path(metadata['records']).resolve()
    if records!=(paths.WORK_RESULTS/'agent-cifar-offline-records-r1804/cifar10-records').resolve():
        raise ValueError('Use the original CIFAR source')
    fronts={str(Path(row['argv'][row['argv'].index('--frontend')+1]).resolve())
            for row in historical.get('processes',[])if'--frontend'in row.get('argv',[])}
    if len(fronts)!=1:raise ValueError('One original trusted frontend path is required')
    frontend=Path(next(iter(fronts)))
    for path in (program,info_path,calibration,probes,probe_source,profile,template,frontend):shared._regular(path)
    selected=worker['agent']['confirm']['selected']
    return dict(source_case=str(case),original_case=str(cold),quality_audit_case=str(audit_case),
        model=MODEL,factory=FACTORY,model_info=str(info_path),calibration=str(calibration),program=str(program),
        agent_dir=str(agent),metadata_template=str(template),profile=str(profile),frontend=str(frontend),
        records=str(records),normalization=normalization,probes=str(probes),probe_source=str(probe_source),report_count=8000,
        selected_iteration=selected['iteration'],selected_origin=selected['origin'],formal_frozen=True,strict_math_passed=True,
        confirmation=worker['agent']['confirm']['evals'],original_api_calls=4,continuation_api_calls=2,total_api_calls=6,
        tuner=True,tuner_budget_s=600,confirm_rule='c1',confirmation_repeated=False,cold_start=False,
        original_complete_seconds=old_marker['complete_case_elapsed_seconds'],
        continuation_complete_seconds=marker['complete_case_elapsed_seconds'],
        audit_complete_seconds=closed['complete_case_elapsed_seconds'],historical_boots_pred=worker['boots_pred'],
        diagnostic_elide_final_ir=False,original_r13_math_checks_reused=3,source_program_ast_matches=True,
        reference_criterion=report.CRITERION,original_sources={k:str(expected[k])for k in
            ('source','checkpoint','inference_preprocessing_source')})


def postfreeze_namespace(audit_case):
    audit_case=Path(audit_case).resolve();namespace=_clone(shared)
    namespace.update(ROUND=ROUND,MODE=MODE,MODEL=MODEL,FACTORY=FACTORY,MODULE=MODULE,__file__=__file__)
    namespace['frozen_source']=lambda case:frozen_source(case,audit_case)
    command=namespace['phase_command']
    def phase_command(args,gate,phase,work,remaining):
        argv,seconds=command(args,dict(gate,diagnostic_elide_final_ir=
            bool(getattr(args,'diagnostic_elide_final_ir',False))),phase,work,remaining)
        return argv+['--quality-audit-case',str(audit_case)],seconds
    namespace['phase_command']=phase_command
    return namespace


def compiled_source(compile_case,source_case,audit_case):
    gate=postfreeze_namespace(audit_case)['compiled_source'](compile_case,source_case)
    case=Path(compile_case).resolve();record=shared.read(case/'result.json')
    plan=Path(record['phases']['compile']['runtime_plan']).resolve()
    if not plan.is_relative_to(case/'post-freeze/emit')or not plan.name.endswith('.runtime-plan.json'):
        raise ValueError('Retain this current compiled RuntimePlan under its own emitted artifact directory')
    shared._regular(plan)
    gate.update(runtime_plan=str(plan),compiled_bootstraps=record['compiled_bootstraps'])
    return gate


def report_namespace(compile_case,audit_case):
    compile_case=Path(compile_case).resolve();audit_case=Path(audit_case).resolve();namespace=_clone(report)
    namespace['__spec__']=__spec__
    def source_gate(case,model):
        if model!=MODEL:raise ValueError('Only the audited hybrid VGG program is supported')
        gate=compiled_source(compile_case,case,audit_case)
        gate.update(continuation_report_adapter=True,report_outcome_used_for_selection=False)
        return gate
    run=namespace['run_process'];trusted_worker=namespace['worker']
    def launch(argv,**kwargs):
        argv=list(argv)
        if any(flag in argv for flag in ('--compile-case','--quality-audit-case','--phase')):
            raise ValueError('Pass each explicit report source path once')
        argv+=['--phase','report','--compile-case',str(compile_case),'--quality-audit-case',str(audit_case)]
        out=Path(argv[argv.index('--out')+1]);snapshot=out/'source';snapshot.mkdir(exist_ok=False)
        for module in (sys.modules[__name__],shared,report,continuation,terminal_audit):
            shutil.copyfile(module.__file__,snapshot/Path(module.__file__).name)
        return run(argv,**kwargs)
    def worker(args):
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        from .r29_local_compile_matrix import native_metadata_commands
        gate=source_gate(args.source_case,args.model)
        with native_metadata_commands(json.loads(Path(gate['metadata_template']).read_text())):code=trusted_worker(args)
        target=args.out/'worker-result.json'
        if target.is_file():
            result=shared.read(target);result.update(continuation_report_adapter=True,compile_case=str(compile_case),
                quality_audit_case=str(audit_case),report_outcome_used_for_selection=False,new_operator_identifier_digests=0)
            report.write_json(target,result)
        return code
    namespace.update(source_gate=source_gate,run_process=launch,worker=worker)
    return namespace


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase',choices=('compile','report'),default='compile')
    parser.add_argument('--quality-audit-case',type=Path,required=True)
    parser.add_argument('--compile-case',type=Path)
    args,remaining=parser.parse_known_args(sys.argv[1:]if argv is None else argv)
    if args.phase=='report':
        if args.compile_case is None:parser.error('--compile-case is required for the one report')
        return report_namespace(args.compile_case,args.quality_audit_case)['main'](remaining)
    if args.compile_case is not None:parser.error('--compile-case belongs only to report')
    return postfreeze_namespace(args.quality_audit_case)['main'](remaining)


if __name__=='__main__':raise SystemExit(main())
