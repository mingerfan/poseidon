"""Frozen external --dacapo evaluation, separate from one-shot DSL delivery.

Only the compiler owns ciphertext-state scheduling. Configuration is interface
evidence, never numerical certification. No library or security value is changed.
"""
import copy
import json
from pathlib import Path

from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha,write_once
import compiler_configuration as configurations
import poseidon_runtime_compile as compilation

FORMAT='poseidon-compiler-managed-external-evaluation-v1'
RESOURCE_INVENTORY_ROOT=Path('/home/lhohy/Code/Poseidon/DSL/DSL_agent/docs/whole_models/proposals/bootstrap-evalmod-output-diagnostic-r1929/cold-host-inventory-r3206')


def descriptor(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=sha(read_bytes(path,128*1024**2)))


def _read(record):
    require(type(record)is dict and set(record)=={'path','sha256'},'Frozen external descriptor')
    raw=read_bytes(record['path'],128*1024**2)
    require(sha(raw)==record['sha256'],'External compiler dependency bytes changed')
    return json.loads(raw)


def freeze_profile(selector,canary,help_evidence,help_text):
    """Freeze actual compiler/configuration/caller requirements before generation."""
    sdk=checked_bound(_read(selector))
    require(sdk['format']=='poseidon-runtime-toolchain-v2','Original ordinary CPU SDK selector')
    cfg=configurations.configuration(configurations.COMPILER_MANAGED_CONFIGURATION)
    require(sdk['source_lock_sha256']==cfg['source_lock_sha256'],'Original registered source lock')
    base=Path(__file__).absolute().parent
    # The configuration files live in the installed agent root, even in inert controls.
    from hecate_python_env import ROOT
    profile=descriptor(ROOT/'scripts/baseline/poseidon-compiler-profile-v1.json')
    spec=descriptor(ROOT/'scripts/baseline/poseidon-operator-spec-v1.json')
    require(profile['sha256']==cfg['ckks_config_sha256']and spec['sha256']==cfg['operator_spec_sha256'],
            'Original profile and security/operator parameters only')
    compiler=descriptor(Path(sdk['compiler_build'])/'bin/hecate-opt')
    require(compiler['sha256']==sdk['artifacts']['compiler'],'Actual installed compiler image')
    evidence=checked_bound(_read(help_evidence))
    require(evidence['actual_live_capture_validated']is True and evidence['original_argv']==['/hecate-opt','--help']
        and evidence['original_executable_sha256']==compiler['sha256'],'Actual selected compiler interface evidence')
    raw=read_bytes(help_text['path'],16*1024**2)
    require(sha(raw)==help_text['sha256']and all(token in raw for token in
        (b'--dacapo',b'--boot-placement',b'=optimized',b'--runtime-plan-boot-implementation')),
        'Observed automatic pipeline and original native export interface')
    require(type(canary)is dict and set(canary)=={'overlay','plan','result'},'Original scoped ordinary CPU admission descriptors')
    for record in canary.values():
        require(sha(read_bytes(record['path'],16*1024**2))==record['sha256'],'Original scoped admission changed')
        checked_bound(json.loads(read_bytes(record['path'],16*1024**2)),record['binding'])
    lock=descriptor(sdk['source_lock']);source_lock=_read(lock)
    require(lock['sha256']==cfg['source_lock_sha256']and source_lock['dacapo_commit']==cfg['dacapo_commit'],
            'Exact compiler revision and approved patch ancestry')
    source=Path(sdk['runtime_source'])/'third_party/dacapo'
    backend_sources={str(source/name):sha(read_bytes(source/name))for name in [
        'tools/optimizer.cpp','lib/Dialect/Earth/Transforms/DaCapoPlanner.cpp',
        'lib/Dialect/Earth/Transforms/BootstrapPlacement.cpp','lib/Dialect/Earth/Transforms/ProactiveRescaling.cpp',
        'lib/Dialect/Earth/Transforms/EarlyModswitch.cpp']}
    sources={str(Path(module.__file__).absolute()):sha(read_bytes(module.__file__))
             for module in [configurations,compilation]}
    sources[str(Path(__file__).absolute())]=sha(read_bytes(__file__))
    from hecate_python_env import WORK
    catalog=(ROOT/'scripts/baseline/application-source-catalog-v1.json').resolve()
    sources[str(catalog)]=sha(read_bytes(catalog))
    for name in ['migration_core.py','controlled_apply.py','manifest.json']:
        controller=WORK/'results/agent-evidence64-peer-proof-revision-r2812'/name
        sources[str(controller)]=sha(read_bytes(controller))
    for name in ['inventory_current_boot.py','HOST_IDENTITY_ACCEPTANCE.json','SOURCE_EQUIVALENCE_AND_SCOPE.json']:
        path=RESOURCE_INVENTORY_ROOT/name
        sources[str(path)]=sha(read_bytes(path))
    for name in ['compiler_managed_cli.py','compiler_managed_delivery.py','first_output_assessment.py','first_output_backend.py','first_output_numerical_evidence.py','normal_cpu_overlay_backend.py','qualification_session.py','validation_adapter.py','compiler_managed_layout.py','public_geometry_basis.py','public_basis_packing.py','public_constant_loads.py','public_basis_storage.py','decorated_functions.py','candidate_trace.py','first_output_dsl_math.py','unified_graph_contract.py','unified_capacity_constants.py','unified_request_storage.py','public_weight_context.py','frozen_evaluation_inputs.py','unified_graph_prepare.py','public_weight_references.py','public_origin_columns.py','strict_json_files.py','audit_unified_candidate.py','poseidon_sandbox_policy.py','candidate_bundle.py','poseidon_input_groups.py','poseidon_validation.py','normal_cpu_overlay_sandbox.py','normal_cpu_overlay_package.py','ordinary_complete_output.py']:
        path=Path(__file__).with_name(name)
        sources[str(path.absolute())]=sha(read_bytes(path))
    optional_reader=Path(__file__).with_name('ordinary_complete_numerical_evidence.py')
    sources[str(optional_reader.absolute())]=sha(read_bytes(optional_reader))
    return bound(dict(format=FORMAT,selector=copy.deepcopy(selector),ordinary_CPU_canary=copy.deepcopy(canary),
        compiler_help_evidence=copy.deepcopy(help_evidence),compiler_help_text=copy.deepcopy(help_text),
        compiler=compiler,compiler_profile=profile,operator_spec=spec,source_lock=lock,
        configuration=cfg,command=compilation.command_for_configuration(cfg),
        compiler_sources=backend_sources,adapter_sources=sources,
        compiler_revision=source_lock['dacapo_commit'],LLVM_MLIR_version='18.1.2',
        approved_dependency_patch_sha256=source_lock['approved_patches'],
        Agent_responsibility=['computational implementation','fusion','layout','blocking'],
        compiler_responsibility=['rescale','modswitch','bootstrap placement','ciphertext state'],
        default_explicit_DSL_state_operations=False,compiler_interface_observed=True,
        bootstrap_numerically_certified=False,full_SDK_or_first_delivery_certified=False,
        mathematical_tolerance=dict(atol=1e-12,rtol=1e-12),CKKS_tolerance=dict(atol=1e-5,rtol=1e-4),
        reference_domain_application_whole_model_requirements_unchanged=True,
        no_library_or_security_parameter_change=True))


def verify_profile(value):
    checked_bound(value)
    require(value.get('format')==FORMAT,'Independent external compiler profile')
    actual=freeze_profile(value['selector'],value['ordinary_CPU_canary'],
                          value['compiler_help_evidence'],value['compiler_help_text'])
    require(value==actual,'Frozen external compiler configuration/source changed')
    return value


def backend(profile, *, evaluation_inputs=None):
    verify_profile(profile)
    from normal_cpu_overlay_backend import NormalCpuOverlayBackend
    evidence=profile['ordinary_CPU_canary']
    return NormalCpuOverlayBackend(evidence['overlay'],evidence['plan'],evidence['result'],
                                  compiler_evaluation_profile=profile,evaluation_inputs=evaluation_inputs)


def freeze_complete_output_profile(compiler_profile,client_source_profile,protocol_requirements):
    """Optional saved-output reader, independent of the default native launcher."""
    from ordinary_complete_output import freeze_profile
    return freeze_profile(compiler_profile,client_source_profile,protocol_requirements)


def read_complete_outputs(profile,output,request):
    """Read saved full-complex groups; native/first-output authority stays external."""
    from ordinary_complete_output import read_groups
    return read_groups(profile,output,request)


def select_complete_output_policy(policy,output_profile):
    """Freeze optional output selection in the pre-generation numerical policy."""
    from ordinary_complete_numerical_evidence import select_policy
    return select_policy(policy,output_profile)


def _numerical_reader(policy,mathematical_evidence,*,compiler_profile=None):
    """Only a separately produced math record; no crypto output becomes a reference."""
    def reader(request,source,proof):
        from first_output_backend import _json
        from first_output_numerical_evidence import native_input_samples
        descriptors=mathematical_evidence
        prepared=None
        if type(descriptors) is dict:
            prepared=checked_bound(_read(descriptors))
            if prepared.get('format')=='poseidon-independent-math-record-set-v1':
                require(prepared['source_sha256']==sha(source) and
                        prepared['policy_sha256']==sha(canonical(policy)), 'Independent mathematical record-set binding')
                descriptors=prepared['records'];prepared=None
            else:descriptors=[descriptors]
        require(type(descriptors) is list and 1<=len(descriptors)<=1024, 'Finite independent mathematical records')
        rows=[]
        for descriptor in descriptors:
            record=prepared if prepared is not None else checked_bound(_read(descriptor))
            require(record['source_sha256']==sha(source) and record['policy_sha256']==sha(canonical(policy)),
                    'Independent mathematical source/policy binding')
            require(type(record.get('inputs')) is list, 'Independent mathematical input rows')
            rows.extend(copy.deepcopy(record['inputs']))
        if 'native_output_profile' in policy:
            selected=policy['native_output_profile']
            require(type(compiler_profile)is dict and selected['compiler_profile_binding']==compiler_profile['binding'],
                    'Complete-output profile selects the frozen external compiler')
            from ordinary_complete_numerical_evidence import read_rows
            result=read_rows(request,source,proof,policy,rows)
            result['mathematical_evidence']=descriptors[0] if len(descriptors)==1 else descriptors
            return result
        root=Path(proof['evidence']);execution=_json(root/'attempt-00/output/execution.json')
        inputs=native_input_samples(request,_json(root/'attempt-00/output/inputs.json'))
        require(len(inputs)==len(execution['outputs']),'Complete actual native input/output scope')
        observed={sha(canonical(sample)):output for sample,output in zip(inputs,execution['outputs'])}
        require(len(observed)==len(inputs)and set(observed)==set(policy['inputs']),'No missing or duplicate frozen native inputs')
        require(len(rows)==len(policy['inputs']) and
                {r['input_sha256']for r in rows}==set(policy['inputs']),'Every independent mathematical row required once')
        for row in rows:
            output=observed[row['input_sha256']]
            row['CKKS_real']=[output[c][s]for c,s in request['layout']['output_selectors']]
        return dict(inputs=rows,intermediate_decryptions=0,reference_or_tolerance_changed=False,
                    mathematical_evidence=descriptors[0] if len(descriptors)==1 else descriptors)
    return reader


def assess_frozen(delivery,delivery_binding,profile,plan,output,*,batch,checkpoint,mathematical_evidence):
    """Real external numerical evaluation; never generation or feedback repair."""
    from delivery_component import verify_delivery
    from first_output_assessment import assess_once
    from first_output_backend import FrozenSourceBackendEvaluator
    root=Path(output).absolute();root.mkdir(mode=0o700)
    captured={}
    receipt,request,source,mechanism=verify_delivery(delivery,delivery_binding)
    verify_profile(profile);checked_bound(plan)
    require(mechanism.get('compiler_evaluation_profile')==profile,'External configuration must freeze before first output')
    require(request['compiler_configuration']==profile['configuration'],'Frozen request/evaluator configuration differs')
    slot=plan['slots'][request['request_id']];policy=checked_bound(slot['policy'])
    service=backend(profile,evaluation_inputs=policy.get("evaluation_inputs"))
    evaluator=FrozenSourceBackendEvaluator(service,output=Path(output)/'native',frozen_identity=service.identity_manifest(),
        batch=batch,checkpoint=checkpoint,numerical_reader=_numerical_reader(policy,mathematical_evidence,compiler_profile=profile))
    def evaluate(request,source):
        witness=evaluator(request,source)
        # Unknown SDK/adapter/evidence availability is not proof of a compiler bug.
        if witness.get('failure_category')=='backend_blocked':
            proof=witness.get('backend_binding')
            diagnosis=dict(status='evaluation_failed',dependency_confirmed=False)
            if proof:
                folder=Path(proof['evidence'])
                diagnosis=classify_dependency(json.loads(read_bytes(folder/'report.json')),folder,profile)
            witness['failure_category']=diagnosis['status']
            witness['dependency_failure_confirmed']=diagnosis['dependency_confirmed']
            witness['dependency_evidence']=diagnosis
        captured['witness']=witness
        return witness
    original=assess_once(delivery,delivery_binding,plan,root/'original-chain',evaluate)
    result=_dependency_outcome(original,captured.get('witness',{}))
    result.pop('binding')
    result['original_external_chain']=descriptor(root/'original-chain/assessment.json')
    saved=bound(result)
    write_once(root/'assessment.json',canonical(saved))
    return saved


def _dependency_outcome(original,witness):
    # The legacy evaluator also uses backend_blocked for missing evidence.
    # Preserve its complete receipt; the current mode requires an actual cause.
    checked_bound(original)
    result=dict(original);result.pop('binding')
    evidence=witness.get('dependency_evidence',{})
    confirmed=(witness.get('dependency_failure_confirmed')is True
               and evidence.get('dependency_confirmed')is True)
    result['dependency_failure_confirmed']=confirmed
    if result.get('status')=='backend_blocked'and not confirmed:
        result.update(status='evaluation_failed',failure_category='evaluation_failed',
                      dependency_classification='missing_or_unverified_evidence_is_not_a_backend_cause')
    return bound(result)


def classify_dependency(report,folder,profile):
    """Only concrete internal compiler evidence grants backend_blocked."""
    verify_profile(profile)
    attempt=report.get('attempts',[])
    if len(attempt)!=1:return dict(status='evaluation_failed',dependency_confirmed=False)
    attempt=attempt[0]
    if attempt.get('checked')is True and attempt.get('trace',{}).get('frontend')=='real_Hecate'and attempt.get('compile_command')==profile['command']:
        plan=Path(folder)/'attempt-00/output/lowered._hecate_golden.runtime-plan.json'
        spec=json.loads(read_bytes(profile['operator_spec']['path']))
        if plan.is_file()and spec['operators']['boot']['supported']is False:
            body=json.loads(read_bytes(plan,16*1024**2))
            if any(op.get('op')=='boot'for op in body.get('execution',[])):
                return dict(status='backend_blocked',dependency_confirmed=True,kind='selected_runtime_boot_unsupported',
                    compiler=profile['compiler'],plan=descriptor(plan),operator_spec=profile['operator_spec'])
        path=Path(folder)/'attempt-00/compile.log'
        raw=read_bytes(path,16*1024**2).decode('utf-8',errors='replace')
        hits=[p for p in profile['compiler_sources']if Path(p).name in raw]
        if hits and ('Assertion' in raw or 'LLVM ERROR:' in raw):
            return dict(status='backend_blocked',dependency_confirmed=True,kind='compiler_internal_failure',
                compiler=profile['compiler'],log=descriptor(path),source_files={p:profile['compiler_sources'][p]for p in hits})
    category='validation_failed'if attempt.get('executed')else'evaluation_failed'
    return dict(status=category,dependency_confirmed=False,failure_layer=attempt.get('failure_layer'))
