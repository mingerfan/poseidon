"""One original ordinary-CPU client over one retained compiled public program.

This is a B-owned backend acceptance primitive, not Agent generation or repair.
The unchanged client creates one process-local ephemeral key set and only decodes
the program's final outputs. No persistent private assets are created or read.
"""
import copy
from pathlib import Path
import shutil

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
from first_output_backend import _json, verify_qualification_binding
import normal_cpu_overlay_sdk as sdk

FORMAT = 'poseidon-normal-cpu-overlay-retained-program-probe-v1'


def _proof(descriptor):
    require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'}, 'Retained compilation descriptor')
    path = Path(descriptor['path']).absolute()
    require(sha(read_bytes(path)) == descriptor['sha256'], 'Retained compilation binding changed')
    proof = checked_bound(_json(path)); root = Path(proof['evidence'])
    request = _json(root/'request.json'); source = read_bytes(root/'attempt-00/candidate.py')
    verify_qualification_binding(proof, request, source)
    return proof, root, request, source


def prepare(overlay, retained_compilation, output, *, work, platform, agent_root):
    context = sdk.verify(overlay, work=work, platform=platform, agent_root=agent_root)
    proof, root, request, source = _proof(retained_compilation)
    require(request['compiler_configuration'] == context['outer']['ordinary_compiler_configuration'],
            'Original registered ordinary CPU configuration required')
    output = Path(output).absolute()
    require(output.parent.resolve() == output.parent and output.parent == Path(work)/'results'
            and not output.exists(), 'New separate accounted probe output')
    original_inputs = _json(root/'attempt-00/output/inputs.json')
    require(type(original_inputs) is list and 1 <= len(original_inputs) <= 16, 'Original fixed ordinary input groups')
    import normal_cpu_overlay_sandbox
    sources={str(Path(p).absolute()):sha(read_bytes(p)) for p in (__file__,sdk.__file__,normal_cpu_overlay_sandbox.__file__)}
    return bound(dict(format=FORMAT, overlay=copy.deepcopy(overlay), retained_compilation=copy.deepcopy(retained_compilation),
        original_compilation_scope='retained IR/source and references only; historical zero-key runtime is not current evidence',
        request_id=request['request_id'], source_sha256=sha(source), source_files=sources, output=str(output),
        original_inputs_sha256=sha(read_bytes(root/'attempt-00/output/inputs.json')),
        original_reference_sha256=sha(canonical(_json(root/'report.json')['attempts'][0]['comparison']['reference'])),
        selected_inner_sdk=context['outer']['inner_selector']['path'],
        limits=dict(runtime_client_invocations=1, ephemeral_key_sets_maximum=1, persistent_key_epochs=0,
            API=0, compilation=0, reference_model_forwards=0, downloads=0,
            CPU=[150,155],AS=4*1024**3,FSIZE=16*1024**2,FD=128,CORE=0),
        CKKS=dict(atol=1e-5,rtol=1e-4),bootstrap=False,Agent_generated=False,first_output_acceptance=False))


def execute(plan, batch, checkpoint):
    """The trusted B worker registers/finishes its own frozen shared batch."""
    checked_bound(plan);require(plan['format']==FORMAT and callable(checkpoint), 'Frozen backend probe plan')
    from shared_execution import Batch
    require(isinstance(batch,Batch), 'Original shared coordinator batch')
    row=batch.record()
    require(row['plan']['source_binding']==plan['binding'] and row['plan']['stages']['encrypted']==1
            and all(v==0 for k,v in row['plan']['stages'].items()if k!='encrypted'), 'Exactly one original CPU client, no compile/API/repair')
    output=Path(plan['output']);require(any(Path(a['path'])==output for a in row['artifacts']), 'Probe output is the accounted batch directory')
    require(output.is_dir() and not (output/'probe-start.json').exists(), 'A consumed probe cannot be restarted')
    from hecate_python_env import WORK,ROOT
    from platform_config import configuration
    from poseidon_runtime_paths import MANIFEST
    require(str(MANIFEST)==plan['selected_inner_sdk'], 'Select the frozen ordinary SDK before importing the worker')
    context=sdk.verify(plan['overlay'],work=WORK,platform=configuration()['id'],agent_root=ROOT)
    proof,root,request,source=_proof(plan['retained_compilation'])
    require(request['request_id']==plan['request_id'] and sha(source)==plan['source_sha256']
            and all(sha(read_bytes(p))==h for p,h in plan['source_files'].items()), 'Probe source or retained request changed')
    from application_backend import PoseidonBackend
    PoseidonBackend().check_request(request,'numerical')
    write_once(output/'probe-start.json',canonical(bound(dict(format=FORMAT,plan_binding=plan['binding'],
        SDK_binding=context['outer']['binding'],runtime_client_attempts=1,ephemeral_key_sets_may_be_created=1,
        persistent_key_epochs=0,Agent_API=0,compilation=0,bootstrap=False))))
    client=output/'client';client.mkdir(mode=0o700)
    shutil.copytree(root/'attempt-00/output/bound',client/'bound',symlinks=False)
    raw_inputs=read_bytes(root/'attempt-00/output/inputs.json');require(sha(raw_inputs)==plan['original_inputs_sha256'], 'Original inputs changed')
    write_once(client/'inputs.json',raw_inputs);write_once(client/'payload.json',canonical(dict(format=FORMAT,plan_binding=plan['binding'])))
    argv=['/poseidon-agent-cpu','/out/bound/plan.json','/operator-spec.json','/out/bound/bundle','/out/inputs.json','/out/execution.json']
    from normal_cpu_overlay_sandbox import run
    checkpoint()
    with batch.stage('encrypted'):
        code=run(client/'payload.json',client,argv,client/'execute.log',150,selected_library=context['library'])
    write_once(output/'native-terminal.json',canonical(bound(dict(exit_code=code,runtime_client_attempts=1,
        ephemeral_key_material_created='one set expected on successful original client; unknown on failure',persistent_key_epochs=0))))
    require(code==0,'Original CPU client failed; preserve consumed attempt')
    execution=_json(client/'execution.json')
    from poseidon_artifact_gate import PRIMES
    require(execution['backend']=='poseidon_cpu' and execution['encrypted_execution'] is True
            and execution['executor_secret_key'] is False and execution['bootstrap'] is False
            and execution['security']=='tc128' and execution['degree']==32768
            and execution['q']==list(PRIMES[:13]) and execution['p']==list(PRIMES[13:]), 'Original Q13/P1/tc128 client boundary')
    inputs=_json(client/'inputs.json');outputs=execution['outputs']
    actual=[[sample[c][s] for c,s in request['layout']['output_selectors']]for sample in outputs]
    reference=_json(root/'report.json')['attempts'][0]['comparison']['reference']
    require(sha(canonical(reference))==plan['original_reference_sha256'] and len(actual)==len(inputs)==len(reference), 'Original reference/sample mapping')
    from first_output_assessment import compare
    comparisons=[compare(r,a,atol=1e-5,rtol=1e-4)for r,a in zip(reference,actual)]
    dso=checked_bound(_json(client/'normal-cpu-overlay-dso.json'))
    require(dso['selected_library_sha256']==context['inner']['artifacts']['poseidon_library'], 'Actual selected repaired DSO binding')
    checkpoint();sdk.verify(plan['overlay'],work=WORK,platform=configuration()['id'],agent_root=ROOT)
    result=bound(dict(format='poseidon-normal-cpu-overlay-original-client-result-v1',status='original_client_completed',
        plan_binding=plan['binding'],SDK_binding=context['outer']['binding'],request_id=request['request_id'],source_sha256=sha(source),
        original_compilation_binding=proof['binding'],actual_DSO_binding=dso['binding'],
        input_groups=len(inputs),compared_values=sum(x['compared_values']for x in comparisons),
        failed_values=sum(x['failed_values']for x in comparisons),max_absolute_error=max(x['max_absolute_error']for x in comparisons),
        comparisons=comparisons,real_encrypted_execution=True,API=0,compilation=0,reference_model_forwards=0,
        runtime_client_invocations=1,ephemeral_key_sets=1,persistent_key_epochs=0,bootstrap=False,
        ordinary_CPU_canary_passed=sum(x['failed_values']for x in comparisons)==0,
        complete_normal_SDK_qualification=False,
        Agent_generated=False,first_output_acceptance=False,complete_network_acceptance=False,
        historical_response_source_package_and_certificate_unchanged=True))
    write_once(output/'report.json',canonical(result));return result
