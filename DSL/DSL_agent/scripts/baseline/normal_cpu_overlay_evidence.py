"""Read the actual B ordinary-CPU canary; never grant whole-SDK correctness.

All recorded numbers are recomputed from retained original references and actual
final decode records. This reader does not compile, execute HE, read keys or
run the original model. Its output can admit further independent qualification.
"""
from pathlib import Path

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from first_output_backend import _json
from first_output_assessment import compare
import normal_cpu_overlay_sdk as sdk


def _descriptor(record):
    require(type(record) is dict and set(record)=={'path','sha256','binding'}, 'Bound canary descriptor')
    require(sha(read_bytes(record['path']))==record['sha256'], 'Canary record changed')
    return checked_bound(_json(record['path']),record['binding'])


def _tree(root):
    root=Path(root)
    require(root.is_dir() and root.resolve()==root, 'Actual retained compiled bundle')
    return {str(p.relative_to(root)):sha(read_bytes(p,16*1024**2))
            for p in sorted(root.rglob('*')) if p.is_file()}


def verify(overlay, plan_record, result_record, *, work, platform, agent_root):
    context=sdk.verify(overlay,work=work,platform=platform,agent_root=agent_root)
    plan=_descriptor(plan_record);result=_descriptor(result_record)
    import normal_cpu_overlay_probe as probe
    import normal_cpu_overlay_sandbox as sandbox
    from normal_cpu_overlay_probe import FORMAT,_proof
    from normal_cpu_overlay_sandbox import loaded_library
    require(plan['format']==FORMAT and plan['overlay']==overlay
            and result['format']=='poseidon-normal-cpu-overlay-original-client-result-v1'
            and result['plan_binding']==plan['binding'] and result['SDK_binding']==context['outer']['binding'], 'Actual canary SDK/plan binding')
    root=Path(plan['output']);client=root/'client'
    require(root.parent==Path(work)/'results' and root.resolve()==root
            and Path(result_record['path'])==root/'report.json', 'Exact accounted canary result')
    expected_sources={str(Path(p).absolute()):sha(read_bytes(p))for p in
                      (probe.__file__,sdk.__file__,sandbox.__file__)}
    require(plan['source_files']==expected_sources, 'Exact original canary adapter sources changed')
    require(plan['CKKS']==dict(atol=1e-5,rtol=1e-4)
            and plan['bootstrap'] is plan['Agent_generated'] is plan['first_output_acceptance'] is False
            and plan['limits']==dict(runtime_client_invocations=1,ephemeral_key_sets_maximum=1,persistent_key_epochs=0,
                API=0,compilation=0,reference_model_forwards=0,downloads=0,CPU=[150,155],AS=4*1024**3,FSIZE=16*1024**2,FD=128,CORE=0),
            'Unchanged ordinary client budget, numerical threshold and scope')
    proof,original,request,source=_proof(plan['retained_compilation'])
    require(request['compiler_configuration']==context['outer']['ordinary_compiler_configuration']
            and plan['selected_inner_sdk']==context['outer']['inner_selector']['path']
            and result['request_id']==plan['request_id']==request['request_id']
            and result['source_sha256']==plan['source_sha256']==sha(source)
            and result['original_compilation_binding']==proof['binding'], 'Original compilation/request ancestry')
    require(_tree(client/'bound')==_tree(original/'attempt-00/output/bound'), 'Client used a changed retained RuntimePlan/bundle')
    raw_inputs=read_bytes(client/'inputs.json',16*1024**2)
    require(sha(raw_inputs)==plan['original_inputs_sha256']==sha(read_bytes(original/'attempt-00/output/inputs.json',16*1024**2)), 'Original fixed canary inputs')
    start=checked_bound(_json(root/'probe-start.json'));terminal=checked_bound(_json(root/'native-terminal.json'))
    require(start['plan_binding']==plan['binding'] and start['SDK_binding']==context['outer']['binding']
            and start['runtime_client_attempts']==1 and start['persistent_key_epochs']==0
            and terminal['exit_code']==0 and terminal['runtime_client_attempts']==1
            and terminal['persistent_key_epochs']==0, 'Actual one-client terminal')
    dso=loaded_library(client/'execute.log',context['library'])
    require(dso==checked_bound(_json(client/'normal-cpu-overlay-dso.json'))
            and result['actual_DSO_binding']==dso['binding']
            and dso['selected_library_sha256']==context['inner']['artifacts']['poseidon_library'], 'Actual repaired DSO loader evidence')
    execution=_json(client/'execution.json')
    from poseidon_artifact_gate import PRIMES
    require(execution['backend']=='poseidon_cpu' and execution['encrypted_execution'] is True
            and execution['executor_secret_key'] is False and execution['bootstrap'] is False
            and execution['security']=='tc128' and execution['degree']==32768
            and execution['q']==list(PRIMES[:13]) and execution['p']==list(PRIMES[13:]), 'Actual original ordinary CPU parameters')
    inputs=_json(client/'inputs.json');outputs=execution['outputs']
    require(type(outputs)is list and len(outputs)==len(inputs), 'Actual final output sample count')
    actual=[[sample[c][s]for c,s in request['layout']['output_selectors']]for sample in outputs]
    reference=_json(original/'report.json')['attempts'][0]['comparison']['reference']
    require(sha(canonical(reference))==plan['original_reference_sha256'] and len(reference)==len(inputs), 'Original reference mapping')
    comparisons=[compare(r,a,atol=1e-5,rtol=1e-4)for r,a in zip(reference,actual)]
    counts=dict(input_groups=len(inputs),compared_values=sum(x['compared_values']for x in comparisons),
        failed_values=sum(x['failed_values']for x in comparisons),max_absolute_error=max(x['max_absolute_error']for x in comparisons))
    require(all(result[k]==v for k,v in counts.items()) and result['comparisons']==comparisons
            and result['ordinary_CPU_canary_passed']==(counts['failed_values']==0), 'Claimed canary numbers differ from actual final outputs')
    require(result['real_encrypted_execution'] is True and result['runtime_client_invocations']==1
            and result['ephemeral_key_sets']==1 and result['persistent_key_epochs']==0
            and result['API']==result['compilation']==result['reference_model_forwards']==0
            and result['Agent_generated'] is result['first_output_acceptance'] is result['complete_network_acceptance'] is False
            and result['complete_normal_SDK_qualification'] is False, 'Canary cannot promote unrelated deliverables')
    from shared_execution import Coordinator
    journal,_=Coordinator(Path(work).parents[1]/'coordination/v1',[Path(work)/'results']).snapshot()
    rows=[row for row in journal['batches'].values() if row['plan']['source_binding']==plan['binding']
          and any(Path(a['path'])==root for a in row['artifacts'])]
    require(len(rows)==1 and rows[0]['status']=='succeeded' and rows[0]['active'] is None
            and rows[0]['used']['encrypted']==1
            and all(v==0 for k,v in rows[0]['used'].items()if k!='encrypted')
            and rows[0]['plan']['stages']['encrypted']==1
            and all(v==0 for k,v in rows[0]['plan']['stages'].items()if k!='encrypted'), 'One terminal original shared execution batch required')
    import normal_cpu_overlay_host_evidence
    host=normal_cpu_overlay_host_evidence.verify(root,context,plan,result,rows[0])
    files={str(p):sha(read_bytes(p,16*1024**2))for p in sorted(root.rglob('*'))if p.is_file()}
    runtime_plan=_json(client/'bound/plan.json')
    ops=sorted({row['op']for row in runtime_plan['execution']if row.get('kind')=='compute'})
    require(not any('boot' in op.lower()for op in ops), 'Ordinary canary cannot certify refresh')
    require(sha(read_bytes(result_record['path']))==result_record['sha256'] and all(sha(read_bytes(p,16*1024**2))==h for p,h in files.items()), 'Canary evidence changed while inspected')
    return bound(dict(format='poseidon-normal-CPU-scoped-canary-evidence-v1',overlay=overlay,plan=plan_record,result=result_record,
        original_compilation_binding=proof['binding'],terminal_batch_sha256=sha(canonical(rows[0])),actual_DSO_binding=dso['binding'],
        ordinary_configuration=context['outer']['ordinary_compiler_configuration'],counts=counts,actual_covered_runtime_ops=ops,
        evidence_sha256=files,host_witness_binding=host['binding'],host_witness_evidence_sha256=host['evidence_sha256'],
        ordinary_CPU_test_admission=counts['failed_values']==0,
        complete_normal_SDK_qualification=False,bootstrap_certified=False,Agent_first_output_or_complete_network_certified=False,
        scope='One original compiled program with its fixed inputs; admission for independent ordinary-CPU trials, not correctness of other models'))
