"""Typed array arithmetic, alias and overlap witnesses; trusted finite probes."""
import ast
import copy
import hashlib
from benchmark_graph import digest
from seal_artifact_gate import require
from unified_public_contract import normalize

FEATURES=frozenset(('cipher_pair','empty_left','inplace.Add','inplace.Sub','inplace.Mult',
                    'object.overlap_mult','overlap','rank_broadcast','zero_dim'))


def check(source,request,reference,events,feature,*,max_spans,max_attempts):
    from unified_public_coverage import fingerprint,span
    candidates=[];seen=set()
    for event in events:
        position=tuple(event['span'])
        if (set(event['features']) & {'event.object_binary','event.object_inplace'} and
                feature in event['features'] and position not in seen):
            candidates.append(event);seen.add(position)
    attempts=0
    for event in candidates[:max_spans]:
        attempts+=1;require(attempts<=max_attempts,'Public influence work bound')
        probe=None;changed=source
        try:
            if feature=='inplace.Mult':
                tree=ast.parse(source)
                target=next(n for n in ast.walk(tree) if type(n) is ast.AugAssign and list(span(n))==event['span'])
                require(type(target.target) is ast.Name,'Alias witness requires a local name target')
                # Keep arithmetic and RHS evaluation, but bind a fresh array so
                # only actual alias observers can distinguish this intervention.
                replacement=ast.Assign(targets=[copy.deepcopy(target.target)],value=ast.BinOp(
                    left=ast.Name(id=target.target.id,ctx=ast.Load()),op=copy.deepcopy(target.op),right=copy.deepcopy(target.value)))
                class Rebind(ast.NodeTransformer):
                    def visit_AugAssign(self,node):
                        return ast.copy_location(replacement,node) if node is target else self.generic_visit(node)
                changed=ast.unparse(ast.fix_missing_locations(Rebind().visit(tree)))
                intervention=dict(kind='rebind_instead_of_alias_mutation')
            else:
                kind={'cipher_pair':'cipher_pair','empty_left':'empty_as_zero',
                      'inplace.Add':'sequential_overlap','overlap':'sequential_overlap',
                      'object.overlap_mult':'sequential_overlap'}.get(feature,'all_results')
                if kind=='sequential_overlap':
                    require('event.object_inplace' in event['features'] and event['facts']['overlapping'],
                            'Observable overlap requires actual in-place shared storage')
                probe=dict(span=event['span'],kind=kind)
                intervention=dict(kind=kind,probe=probe)
            expanded=normalize(changed,request,_object_probe=probe)
            actual=fingerprint(expanded,request)
        except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError,StopIteration):
            continue
        require(len(actual)==len(reference),'Changed influence ABI')
        delta=max((abs(a-b) for a,b in zip(actual,reference)),default=0.)
        if delta>1e-9:
            source_hash=hashlib.sha256(changed.encode()).hexdigest()
            return dict(span=event['span'],intervention=intervention,
                        changed_source_sha256=source_hash,
                        intervention_sha256=digest(dict(source_sha256=source_hash,intervention=intervention)),
                        changed_normalized_sha256=expanded['construction']['normalized_sha256'],
                        max_probe_delta=delta,
                        evidence='finite_typed_pair_alias_or_overlap_influence_not_all_input_proof'),attempts
    return None,attempts
