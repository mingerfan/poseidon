"""External startup admission from immutable delivery and original shared ledger.

This reader never retries generation, changes a terminal ledger, admits a DSL
to the compiler, or certifies numerical results. The original external assessor
still verifies the full host request and the exact source before qualification.
"""
import math
from pathlib import Path

from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha,verify_delivery
import lossless_host_bound_delivery as host_entry
import dual_reference_first_output_assessment as external
import shared_execution

VERSION='poseidon-completed-delivery-external-startup-v1'


def source_manifest():
    import delivery_component
    modules=[host_entry,external,shared_execution,delivery_component]
    paths=[Path(__file__).absolute(),*[Path(m.__file__).absolute()for m in modules]]
    return {str(p):sha(read_bytes(p))for p in paths}


def bind_startup(mechanism):
    external.verify_selection(mechanism)
    require('external_startup_contract'not in mechanism,'Select startup once before generation')
    sources=source_manifest()
    require(all(mechanism['source_sha256'].get(p)==value for p,value in sources.items()),'Startup absent from generator freeze')
    return bound(dict({k:v for k,v in mechanism.items()if k!='binding'},external_startup_contract=VERSION,external_startup_sources=sources))


def verify_selection(mechanism):
    external.verify_selection(mechanism)
    sources=source_manifest()
    require(mechanism.get('external_startup_contract')==VERSION and mechanism.get('external_startup_sources')==sources,
        'External startup selection changed or was not frozen')
    require(all(mechanism['source_sha256'].get(p)==value for p,value in sources.items()),'Unfrozen startup source')
    return mechanism


def _state(folder,receipt,mechanism,source,row,owner_observed,*,test_only,retained):
    """Pure fixture seam. Live entry obtains owner and row from original tools."""
    require(type(test_only)is bool and type(retained)is bool,'Explicit admission scope')
    checked_bound(receipt);checked_bound(mechanism)
    require(receipt['mechanism_binding']==mechanism['binding'],'Receipt generator differs')
    require(receipt.get('mode')=='delivery_generation'and receipt.get('origin')=='provider_first_output'
        and receipt.get('real_Agent')is True and receipt.get('first_output_eligible')is True
        and type(receipt.get('first_generation_index'))is int and receipt['first_generation_index']==0
        and type(receipt.get('post_output_repairs'))is int and receipt['post_output_repairs']==0,'Original single delivery required')
    require(receipt.get('external_evaluation_performed')is False,'Delivery must precede external evaluation')
    if source is None:
        require(receipt['source_sha256']is None and receipt['source_bytes']==0,'Missing original DSL')
    else:
        require(type(source)is bytes and receipt['status']=='dsl_delivered'and receipt['source_sha256']==sha(source)
            and receipt['source_bytes']==len(source),'Original source binding')
    require(type(row)is dict and row.get('status')in('succeeded','failed','interrupted')and row.get('active')is None,'Delivery stage remains active')
    require(type(row.get('finished'))in(int,float)and math.isfinite(row['finished'])and row['finished']>0,'Terminal timestamp required')
    plan=row.get('plan');require(type(plan)is dict and row.get('plan_binding')==sha(canonical(plan)),'Original ledger plan binding')
    require(plan.get('source_binding')==mechanism['binding'],'Ledger used another generator')
    require(type(plan.get('stages'))is dict and set(plan['stages'])==set(shared_execution.KINDS)
        and all(type(v)is int and v==int(k=='api')for k,v in plan['stages'].items()),'Original API-only reservation')
    used=row.get('used');require(type(used)is dict and set(used)==set(shared_execution.KINDS)
        and all(type(v)is int and v==int(k=='api')for k,v in used.items()),'Original single consumed API; no restored allowance')
    folder=Path(folder).absolute();require(folder.resolve()==folder,'Indirect delivery folder')
    accounted=False
    for item in row.get('artifacts',[]):
        if type(item)is not dict or type(item.get('path'))is not str:continue
        root=Path(item['path'])
        if not(root.is_absolute()and root.resolve()==root and folder!=root and folder.is_relative_to(root)):continue
        st=root.stat();require(st.st_dev==item.get('device')and st.st_ino==item.get('inode'),'Delivery artifact identity changed')
        accounted=True
    require(accounted,'Delivery outside original accounted artifacts')
    metrics=receipt.get('provider_metrics');require(type(metrics)is dict,'Original Provider metrics')
    require(all(type(metrics.get(k))is int and metrics[k]==1 for k in('agent_calls','generation_attempts','request_attempts'))
        and metrics.get('transport_retries')==0,'Original Provider counts changed')
    require(type(metrics.get('calls'))is list and len(metrics['calls'])==1,'Exactly one original HTTP record')
    config=mechanism.get('provider_configuration');require(type(config)is dict and config.get('max_calls')==1 and config.get('provider_retries')==0,'Frozen single call configuration')
    for key in ('model','reasoning_effort','max_tokens','timeout_seconds','service_provider','provider_retries','max_calls'):
        require(metrics.get(key)==config.get(key),'Provider configuration changed')
    interrupted=row['status']=='interrupted'
    if interrupted:
        require(owner_observed!=row.get('owner'),'Interrupted delivery owner still alive')
        require(row.get('uncertain_stage')is None,'Interrupted API may still be uncertain')
        stage=row.get('last_stage');require(type(stage)is dict and stage.get('kind')=='api'and stage.get('number')==1
            and stage.get('outcome')=='completed'and stage.get('execution_uncertain')is False,'Interrupted API did not complete')
        call=metrics['calls'][0];diag=call.get('transport_diagnostics')
        require(call.get('status')=='response_received'and type(diag)is dict and diag.get('stage')=='complete'
            and diag.get('status')==200 and call.get('finish_reason')in('stop','length'),'Original complete response required')
        require(receipt.get('response_sha256')is not None,'Interrupted delivery response not preserved')
    return bound(dict(format=VERSION,mode='retained_development_read'if retained else'external_startup_admission',
        request_id=receipt['request_id'],delivery_binding=receipt['binding'],mechanism_binding=mechanism['binding'],
        source_sha256=receipt['source_sha256'],ledger_row_sha256=sha(canonical(row)),parent_status=row['status'],
        completed_API_with_bookkeeping_interruption=interrupted,has_frozen_DSL=source is not None,
        eligible_for_external_assessment=not(retained or test_only),first_output_success_certified=False,
        full_host_verification_deferred_to_original_assessor=True,raw_source_bytes_changed=False,ledger_mutations=0,
        API=0,qualification=0,compiler=0,encrypted=0,retries=0,repairs=0,test_only=test_only))


def _read(folder,binding,coordinator,batch_name,*,retained):
    require(type(coordinator)is shared_execution.Coordinator,'Original shared coordinator required')
    receipt,wire,source,mechanism=verify_delivery(folder,binding)
    host_entry._mechanism(mechanism)
    if not retained:verify_selection(mechanism)
    snapshot,_=coordinator.snapshot()
    require(batch_name in snapshot['batches'],'Original delivery ledger row absent')
    row=snapshot['batches'][batch_name]
    owner=row.get('owner');require(type(owner)is dict and type(owner.get('pid'))is int,'Original owner required')
    observed=shared_execution.process(owner['pid'])
    result=_state(folder,receipt,mechanism,source,row,observed,test_only=False,retained=retained)
    return receipt,wire,source,mechanism,result


def admit(folder,binding,coordinator,batch_name):
    """Future frozen entry only. The original assessor still performs all checks."""
    return _read(folder,binding,coordinator,batch_name,retained=False)


def read_retained(folder,binding,coordinator,batch_name):
    """Read old evidence without retroactively changing its first-output result."""
    return _read(folder,binding,coordinator,batch_name,retained=True)
