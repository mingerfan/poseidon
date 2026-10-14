"""Executed-expression intervention witnesses for the unified public profile.

No eval/exec and no private tests/reference. A span intervention affects all
reachable evaluations at that span; it does not prove individual invocations.
"""
import ast
import copy
import hashlib
import math
import warnings
from benchmark_graph import digest
from construction_evidence import ConstructionEvidenceError
from seal_artifact_gate import require
from unified_public_contract import normalize,event_record

MAX_SPANS=8
MAX_INTERVENTIONS=80

def span(node):
    return tuple(getattr(node,k,0) for k in ("lineno","col_offset","end_lineno","end_col_offset"))

def fingerprint(expanded,request):
    """Independent small evaluator of validated flat arithmetic at the real ABI."""
    period=request["layout"]["input_slot_period"]
    fn=ast.parse(expanded["source"]).body[0]
    def vector(v):
        raw=v if type(v) is list else [v]
        require(len(raw) in (1,period),"Influence constant period")
        return [float(raw[i%len(raw)]) for i in range(period)]
    answers=[]
    for probe in range(3):
        values={k:vector(v) for k,v in expanded["constants"].items()}
        for j,inp in enumerate(request["layout"]["inputs"]):
            size=inp.get("elements",math.prod(inp["shape"]))
            values[inp["dsl_name"]]=[(((i*i+3*i+7*probe+5*j+1)%23)-11)/32
                                      if i<size else 0. for i in range(period)]
        values["zero_ct"]=[0.]*period
        def expression(n):
            if type(n) is ast.Name:return values[n.id]
            if type(n) is ast.Constant:return vector(n.value)
            if type(n) is ast.UnaryOp and type(n.op) is ast.USub:return [-v for v in expression(n.operand)]
            if type(n) is ast.BinOp:
                a,b=expression(n.left),expression(n.right)
                if type(n.op) is ast.Add:return [x+y for x,y in zip(a,b)]
                if type(n.op) is ast.Sub:return [x-y for x,y in zip(a,b)]
                if type(n.op) is ast.Mult:return [x*y for x,y in zip(a,b)]
            if type(n) is ast.Call and type(n.func) is ast.Attribute and n.func.attr=="rotate":
                from hecate_contract import rotation_literal
                a=expression(n.func.value);k=rotation_literal(n.args[0])%period
                return a[k:]+a[:k]
            if type(n) in (ast.List,ast.Tuple):return [expression(v) for v in n.elts]
            raise ValueError("Unexpected normalized influence AST")
        for statement in fn.body:
            if type(statement) is ast.Assign:values[statement.targets[0].id]=expression(statement.value)
            elif type(statement) is ast.Return:
                out=expression(statement.value)
                # The normalizer also permits one complete Expr as a return.
                # Its flat slot vector is one ciphertext, never P outputs.
                if type(statement.value) not in (ast.List,ast.Tuple):out=[out]
                require(len(out)==request["layout"]["output_ciphertexts"] and
                        all(type(v) is list and len(v)==period for v in out),
                        "Influence output ciphertext/slot shape mismatch")
                answers.extend(out[c][s] for c,s in request["layout"]["output_selectors"])
            else:raise ValueError("Unexpected normalized influence statement")
    require(all(math.isfinite(v) for v in answers),"Nonfinite influence probe")
    return answers

def check_exercise(source,request):
    events=[];expanded=normalize(source,request,lambda n:events.append(event_record(n)))
    reference=fingerprint(expanded,request)
    tree=ast.parse(source);nodes={}
    for node in ast.walk(tree):
        if isinstance(node,(ast.expr,ast.stmt)):nodes.setdefault(span(node),node)
    from unified_public_exercises import STRUCTURAL
    from unified_public_interventions import replacements as control_replacements
    witnesses={};attempts=0
    for feature in request["construction_exercise"]["required_features"]:
        from unified_public_unary import FEATURES as UNARY_FEATURES,check as check_unary
        if feature in UNARY_FEATURES:
            witness,spent=check_unary(source,request,reference,events,feature,
                                     max_spans=MAX_SPANS,max_attempts=MAX_INTERVENTIONS-attempts)
            attempts+=spent
            require(witness is not None,"Missing contributing typed unary operation: "+feature)
            witnesses[feature]=witness
            continue
        from unified_public_arithmetic import FEATURES as ARITHMETIC_FEATURES,check as check_arithmetic
        if feature in ARITHMETIC_FEATURES:
            witness,spent=check_arithmetic(source,request,reference,events,feature,
                                          max_spans=MAX_SPANS,max_attempts=MAX_INTERVENTIONS-attempts)
            attempts+=spent
            require(witness is not None,"Missing contributing typed array arithmetic: "+feature)
            witnesses[feature]=witness
            continue
        from unified_public_storage import FEATURES as STORAGE_FEATURES,check as check_storage
        if feature in STORAGE_FEATURES:
            witness,spent=check_storage(source,request,reference,events,feature,
                                       max_spans=MAX_SPANS,max_attempts=MAX_INTERVENTIONS-attempts)
            attempts+=spent
            require(witness is not None,"Missing contributing typed storage context: "+feature)
            witnesses[feature]=witness
            continue
        from unified_public_views import FEATURES as VIEW_FEATURES,check as check_view
        if feature in VIEW_FEATURES:
            witness,spent=check_view(source,request,reference,events,feature,
                                    max_spans=MAX_SPANS,max_attempts=MAX_INTERVENTIONS-attempts)
            attempts+=spent
            require(witness is not None,'Missing observable object storage distinction: '+feature)
            witnesses[feature]=witness
            continue
        positions=sorted({tuple(e["span"]) for e in events if feature in e["features"]})[:MAX_SPANS]
        for position in positions:
            target=nodes.get(position)
            if target is None:continue
            if feature in STRUCTURAL:
                witnesses[feature]=dict(span=list(position),evidence="reachable_structure_only_no_numeric_credit")
                break
            special=control_replacements(target,feature,tree)
            replacements=[ast.Constant(0),ast.Constant(1),ast.Constant(-1),
                ast.List(elts=[ast.Constant(2),ast.Constant(3),ast.Constant(4)],ctx=ast.Load()),
                ast.List(elts=[ast.Constant("2"),ast.Constant("3"),ast.Constant("4")],ctx=ast.Load()),
                ast.Constant("0"),ast.Constant("1"),
                ast.Call(func=ast.Attribute(value=ast.Name(id="np",ctx=ast.Load()),attr="array",ctx=ast.Load()),
                         args=[ast.List(elts=[ast.Constant(2),ast.Constant(3)],ctx=ast.Load())],keywords=[])]
            # Evaluate the original expression exactly once before substituting its
            # result. Removing the call itself would falsely credit side effects in
            # its arguments/receiver even when its result is unused.
            if feature in ("call.enumerate","call.zip","call.items"):
                replacements.insert(0,ast.parse("[(0,2),(1,3)]",mode="eval").body)
            if feature=="call.iter":
                replacements.insert(0,ast.parse("iter([2,3])",mode="eval").body)
            if feature=="call.Chebyshev" or type(target) is ast.Call and type(target.func) is ast.Attribute and target.func.attr=="Chebyshev":
                replacements.insert(0,ast.parse("np.polynomial.Chebyshev([2,3])",mode="eval").body)
            replacements=[ast.Subscript(value=ast.List(elts=[copy.deepcopy(target),r],ctx=ast.Load()),
                                         slice=ast.Constant(1),ctx=ast.Load()) for r in replacements]
            if feature in ("call.list","call.tuple","node.ListComp"):
                # Keep symbolic Expr cells as Expr. Public-only replacements
                # incorrectly rejected a used ciphertext container at the ABI.
                # Evaluate the original call exactly once; only its result is
                # perturbed, so side effects in its arguments cannot earn credit.
                names={n.id for n in ast.walk(tree) if type(n) is ast.Name};suffix=0
                while "probeCell"+str(suffix) in names:suffix+=1
                cell="probeCell"+str(suffix)
                value="["+cell+"+1 for "+cell+" in ("+ast.unparse(target)+")]"
                if feature=="call.tuple":value="tuple("+value+")"
                replacements.insert(0,ast.parse(value,mode="eval").body)
            if feature in ("call.dict","node.DictComp") or type(target) is ast.Call and type(target.func) is ast.Name and target.func.id=="dict":
                names={n.id for n in ast.walk(tree) if type(n) is ast.Name};suffix=0
                while "probeKey"+str(suffix) in names or "probeValue"+str(suffix) in names:suffix+=1
                key="probeKey"+str(suffix);value="probeValue"+str(suffix)
                replacements.insert(0,ast.parse("{"+key+":"+value+"+1 for "+key+","+value+
                    " in ("+ast.unparse(target)+").items()}",mode="eval").body)
            replacements.insert(0,ast.BinOp(left=copy.deepcopy(target),op=ast.Add(),right=ast.Constant(1)))
            replacements.append(ast.BinOp(left=copy.deepcopy(target),op=ast.Mult(),right=ast.Constant(0)))
            # Binding coverage must not fall back to changing the whole call result
            # or helper return: unrelated operands could then supply false credit.
            binding_feature=feature.startswith(("signature.","lambda.")) or feature in ("expansion.star","expansion.kwstar")
            replacements=special+(replacements if isinstance(target,ast.expr) and not binding_feature else [])
            for replacement in replacements:
                attempts+=1;require(attempts<=MAX_INTERVENTIONS,"Public influence work bound")
                class Perturb(ast.NodeTransformer):
                    def visit(self,node):
                        if isinstance(node,type(target)) and span(node)==position:
                            return ast.copy_location(copy.deepcopy(replacement),node)
                        return super().visit(node)
                changed=ast.unparse(ast.fix_missing_locations(Perturb().visit(ast.parse(source))))
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore",SyntaxWarning)
                        actual=fingerprint(normalize(changed,request),request)
                except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):continue
                require(len(actual)==len(reference),"Changed influence ABI")
                delta=max((abs(a-b) for a,b in zip(actual,reference)),default=0.)
                if delta>1e-9:
                    witnesses[feature]=dict(span=list(position),replacement=ast.unparse(ast.fix_missing_locations(copy.deepcopy(replacement))),
                        changed_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),max_probe_delta=delta,
                        evidence="finite_executed_span_influence_not_per_invocation_or_semantic_proof")
                    break
            if feature in witnesses:break
        if feature not in witnesses:
            raise ConstructionEvidenceError("Missing contributing public expression: "+feature,
                feature=feature, reason="not_executed" if not positions else "no_witness_under_bounded_interventions",
                observed_positions=len(positions), attempts=attempts)
    structural=[f for f in witnesses if f in STRUCTURAL]
    numeric=[f for f in witnesses if f not in STRUCTURAL]
    return dict(schema=1,checker_contract="unified-public-execution-witness-v13",
        id=request["construction_exercise"]["id"],witnesses=witnesses,
        events_sha256=digest(events),normalized_sha256=expanded["construction"]["normalized_sha256"],
        numeric_features=numeric,structural_features=structural,probe_count=3,
        slot_period=request["layout"]["input_slot_period"],intervention_attempts=attempts,
        plaintext_reference_used=False,real_frontend_checked=False)

def verify_trace_coverage(checked,record):
    require(record["normalized_sha256"]==checked["normalized_sha256"],"Public normalized witness identity")
    require(record["candidate_python_executed"] is False,"Candidate Python must remain inert")
    require(digest(record["events"])==checked["events_sha256"],"Public execution event schedule changed")
    for feature,witness in checked["witnesses"].items():
        require(any(feature in e["features"] and e["span"]==witness["span"] for e in record["events"]),
                "Missing isolated public event: "+feature)
    scopes={f:w.get('scope','executed_operation') for f,w in checked['witnesses'].items()}
    return dict(schema=1,id=checked["id"],features=checked["numeric_features"],evidence_scopes=scopes,
        numeric_features=checked["numeric_features"],structural_features=checked["structural_features"],
        actual_frontend_checked=True,finite_influence_checked=bool(checked["numeric_features"]),structural_only=not bool(checked["numeric_features"]),
        all_input_semantic_proof=False,per_invocation_proof=False)
