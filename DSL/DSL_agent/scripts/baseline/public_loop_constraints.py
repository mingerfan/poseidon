"""Source-bound public loop facts for a future explicitly frozen prompt version.

This does not alter existing requests, Provider bodies, checker limits or DSLs.
"""
import ast
from pathlib import Path
import re

from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha
import native_public_loops

VERSION='poseidon-public-native-loop-constraints-v1'
NAMES=('MAX_RANGE','MAX_ITERATIONS','MAX_NODES','MAX_INTEGER')


def describe():
    path=Path(native_public_loops.__file__).absolute();raw=read_bytes(path);tree=ast.parse(raw)
    literal={n.targets[0].id:n.value.value for n in tree.body if type(n)is ast.Assign and len(n.targets)==1
        and type(n.targets[0])is ast.Name and n.targets[0].id in NAMES and type(n.value)is ast.Constant and type(n.value.value)is int}
    require(set(literal)==set(NAMES)and all(literal[k]==getattr(native_public_loops,k)for k in NAMES),'Original literal loop authority changed')
    return bound(dict(format=VERSION,limits=literal,source_sha256={str(path):sha(raw),str(Path(__file__).absolute()):sha(read_bytes(__file__))},
        meanings=dict(MAX_RANGE='Maximum number of iterations in each static range, including ranges with nonzero starts or steps.',
            MAX_ITERATIONS='Maximum total expanded loop iterations counted across nested ranges.',
            MAX_NODES='Maximum source/expanded AST nodes enforced by this loop expander.',
            MAX_INTEGER='Maximum absolute public integer operand, range argument and folded result.'),
        numeric_reference_or_DSL_included=False,limits_changed=False,Provider_body_changed=False,old_request_or_response_changed=False,
        selected_in_live_generation=False,whole_native_contract_completeness_claimed=False))


def prompt_text(descriptor):
    checked_bound(descriptor);require(descriptor==describe(),'Loop source/limits changed')
    n=descriptor['limits']
    return (f"Native static-loop limits: each range has at most {n['MAX_RANGE']} iterations; "
        f"all expanded loops together have at most {n['MAX_ITERATIONS']} iterations. "
        f"The loop expander limits source and expanded AST nodes to {n['MAX_NODES']}; "
        f"public integer operands, range arguments and folded results have absolute value at most {n['MAX_INTEGER']}. "
        "These are existing checker bounds, not new capabilities. They do not establish compiler or CKKS feasibility.")


def audit(wire):
    require(type(wire)is dict and type(wire.get('rules'))is str,'Actual public wire rules required')
    descriptor=describe();guidance=wire.get('generation_guidance',{})
    loops=guidance.get('native_syntax_rules',{}).get('loops','')if type(guidance)is dict else''
    text=wire['rules']+'\n'+loops
    # A number in model weights/IDs is not a public statement of this bound.
    statements=re.split(r'[.!?\n]',text)
    limit=str(descriptor['limits']['MAX_RANGE'])
    disclosed=any(re.search(r'\b'+re.escape(limit)+r'\b',s)and re.search(r'\b(range|loop|iterations?)\b',s,re.I)for s in statements)
    return bound(dict(format='poseidon-public-loop-disclosure-audit-v1',request_id=wire.get('request_id'),
        actual_wire_sha256=sha(canonical(wire)),original_rule_and_loop_guidance_sha256=sha(text.encode()),
        MAX_RANGE=descriptor['limits']['MAX_RANGE'],exact_per_range_limit_disclosed=bool(disclosed),
        audit_scope='Mandatory rules and the selected loop syntax explanation only; model constants are excluded.',
        not_a_complete_native_constraint_audit=True,new_descriptor_binding=descriptor['binding'],
        existing_request_edited=False,future_prompt_integration_required=not disclosed))
