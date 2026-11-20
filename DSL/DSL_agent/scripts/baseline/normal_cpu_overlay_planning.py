"""Scoped admission for independent trials on the attested ordinary CPU path.

A passed substrate canary permits testing new requests; every request still has
to pass its own original-source compilation, mathematics and actual CKKS check.
No refresh, whole-model or global SDK numerical certificate is inferred.
"""
from pathlib import Path

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
import normal_cpu_overlay_evidence as evidence
import normal_cpu_overlay_sdk as sdk

AUTHORITY='poseidon-normal-CPU-scoped-canary-admission-v1'


def _sources(agent_root):
    from application_identity import source_catalog,_catalog_values,source_closure
    root=(Path(agent_root)/'scripts/baseline').resolve()
    return source_closure(root,[Path(__file__).name],dynamic_authorities=_catalog_values(source_catalog(root))[2])['sources']


def constraints(overlay,canary_plan,canary_result,*,work,platform,agent_root):
    measured=evidence.verify(overlay,canary_plan,canary_result,work=work,platform=platform,agent_root=agent_root)
    context=sdk.verify(overlay,work=work,platform=platform,agent_root=agent_root)
    records=[overlay,canary_plan,canary_result,context['outer']['parent'],context['outer']['inner_selector']]
    return bound(dict(format='poseidon-frozen-generation-backend-constraints-v1',
        evidence_sha256={row['path']:row['sha256']for row in records},
        artifact_sha256={str(context['library']):context['inner']['artifacts']['poseidon_library']},
        SDK_binding=overlay['binding'],SDK_format=sdk.FORMAT,numeric_policy='strict independent request qualification',
        normal_Application_ready=measured['ordinary_CPU_test_admission'],metadata_attestation=True,
        native_completion_observed=True,certified_refresh_profiles=[],observed_failed_refresh_profiles=[],
        source_of_authority=AUTHORITY,normal_CPU_canary=dict(overlay=overlay,plan=canary_plan,result=canary_result),
        normal_CPU_canary_evidence_binding=measured['binding'],normal_CPU_measured_counts=measured['counts'],
        measured_runtime_ops=measured['actual_covered_runtime_ops'],
        complete_normal_SDK_qualification=False,Agent_first_output_or_complete_network_certified=False,
        normal_CPU_environment=dict(work=str(work),platform=platform,agent_root=str(agent_root)),
        normal_CPU_planning_source_sha256=_sources(agent_root),
        scope='admit further independent ordinary CPU trials only; canary correctness does not establish another model',test_only=False))


def verify(value):
    checked_bound(value)
    require(value.get('source_of_authority')==AUTHORITY and value.get('SDK_format')==sdk.FORMAT,
            'Explicit ordinary CPU scoped admission authority')
    canary=value['normal_CPU_canary'];environment=value['normal_CPU_environment']
    actual=constraints(canary['overlay'],canary['plan'],canary['result'],**environment)
    require(value==actual,'Frozen ordinary CPU admission evidence/source/scope changed')
    return value
