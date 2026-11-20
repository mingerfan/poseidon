"""Finite generation settings selected from public structural/resource facts.

This selects existing guidance and Provider hints before generation. It cannot
reserve a hidden token split, certify a route, generate DSL or repair an answer.
"""
from dataclasses import asdict
from pathlib import Path

from benchmark_graph import validate
from application_planning.resources import logical_graph
from delivery_component import bound, canonical, checked_bound, require, sha, read_bytes
from generation_protocol import Config
import unified_origin_guidance

VERSION = 'poseidon-bounded-first-output-generation-profile-v1'


def select(model, *, maximum_output_tokens=32768, api_timeout_seconds=1200):
    require(type(maximum_output_tokens) is int and 1<=maximum_output_tokens<=32768,
            'Existing finite first-output token cap')
    require(type(api_timeout_seconds) is int and 1<=api_timeout_seconds<=1200,
            'Existing finite first-output API timeout')
    checked=validate(model);features=logical_graph(model,checked)
    uses={}
    for node in model['nodes']:
        for ref in node['inputs']:
            if checked['kinds'][ref]=='cipher':uses[ref]=uses.get(ref,0)+1
    fanout=sum(n>1 for n in uses.values())
    larger=(features['node_count']>=8 or features['largest_cipher_tensor_elements']>256
            or fanout>0 or len(model['inputs'])>1 or len(model['outputs'])>1)
    effort='low'if larger else'high'
    config=Config(model='deepseek-flash',reasoning_effort=effort,max_tokens=maximum_output_tokens,
        max_calls=1,timeout_seconds=api_timeout_seconds,provider_retries=0,stream=False)
    guidance=unified_origin_guidance.VERSION
    require(guidance=='explicit-v21','Audited existing bounded-expression guidance version')
    facts=dict(node_count=features['node_count'],largest_cipher_tensor_elements=features['largest_cipher_tensor_elements'],
        tensor_definition_elements=features['tensor_definition_elements'],cipher_fanout_values=fanout,
        logical_inputs=len(model['inputs']),logical_outputs=len(model['outputs']))
    return bound(dict(format=VERSION,model_sha256=sha(canonical(model)),public_features=facts,
        preparation_options=dict(generation_guidance=guidance),provider_configuration=asdict(config),
        guidance_sha256=sha(read_bytes(Path(unified_origin_guidance.__file__).absolute())),
        selector_sha256=sha(read_bytes(Path(__file__).absolute())),
        decision='use existing bounded-expression context; lower reasoning hint for larger public DAGs to limit deliberation pressure on the shared completion budget'
            if larger else'use existing bounded-expression context with the original high hint on a small unbranched graph',
        no_model_name_dispatch=True,compiler_or_numeric_feedback_used=False,DSL_generated=False,
        actual_reasoning_or_DSL_token_partition=None,completion_guaranteed=False,
        backend_or_numeric_certificate=False,API_calls=0,resource_limits_increased=False))


def verify(profile,model):
    checked_bound(profile)
    config=profile['provider_configuration']
    expected=select(model,maximum_output_tokens=config['max_tokens'],api_timeout_seconds=config['timeout_seconds'])
    require(profile==expected,'Frozen generation profile/model/source changed')
    return profile
