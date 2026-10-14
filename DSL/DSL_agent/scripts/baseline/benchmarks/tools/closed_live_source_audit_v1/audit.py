"""Read-only diagnostics for closed Agent candidates, not an execution backend.

This interpreter handles a small straight-line expression subset on exact public
periodic bindings. It never runs uploaded Python, calls a provider, compiles,
decrypts or replaces an independent model reference.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

class Unsupported(ValueError):
    pass

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise Unsupported("finite numeric values required")
    return float(value)

def numeric(value, period):
    if type(value) is list:
        if len(value) != period:
            raise Unsupported("constant period mismatch")
        return tuple(finite(v) for v in value)
    return finite(value)

def arithmetic(left, right, op, period):
    if type(left) is not tuple:
        left = (finite(left),) * period
    if type(right) is not tuple:
        right = (finite(right),) * period
    if len(left) != period or len(right) != period:
        raise Unsupported("operand period mismatch")
    return tuple(finite(op(a, b)) for a, b in zip(left, right))

def evaluate(source, inputs, constants, period):
    """Interpret only assigned values, numeric arithmetic, rotate, and return."""
    tree = ast.parse(source)
    if len(list(ast.walk(tree))) > 4096:
        raise Unsupported("AST diagnostic budget")
    if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef:
        raise Unsupported("one function required")
    function = tree.body[0]
    if len(function.args.args) != len(inputs):
        raise Unsupported("input arity mismatch")
    symbols = {k: numeric(v, period) for k, v in constants.items()}
    for arg, value in zip(function.args.args, inputs):
        if type(value) is not list or len(value) != period:
            raise Unsupported("input period mismatch")
        symbols[arg.arg] = numeric(value, period)
    operations = {ast.Add: lambda a,b:a+b, ast.Sub: lambda a,b:a-b,
                  ast.Mult: lambda a,b:a*b}
    def expression(node):
        if type(node) is ast.Name:
            if node.id not in symbols:
                raise Unsupported("unbound symbol: " + node.id)
            return symbols[node.id]
        if type(node) is ast.Constant:
            return finite(node.value)
        if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
            return arithmetic(0.0, expression(node.operand), operations[ast.Sub], period)
        if type(node) is ast.BinOp and type(node.op) in operations:
            return arithmetic(expression(node.left), expression(node.right), operations[type(node.op)], period)
        if (type(node) is ast.Call and type(node.func) is ast.Attribute
            and node.func.attr == "rotate" and len(node.args) == 1 and not node.keywords):
            step_node = node.args[0]
            step = ast.literal_eval(step_node)
            if type(step) is not int or abs(step) > 4096:
                raise Unsupported("static integer rotation required")
            vector = expression(node.func.value)
            if type(vector) is not tuple or len(vector) != period:
                raise Unsupported("rotation vector required")
            return tuple(vector[(i + step) % period] for i in range(period))
        raise Unsupported("unsupported expression: " + type(node).__name__)
    returned = None
    for statement in function.body:
        if returned is not None:
            raise Unsupported("statement after return")
        if type(statement) is ast.Assign and len(statement.targets) == 1 and type(statement.targets[0]) is ast.Name:
            symbols[statement.targets[0].id] = expression(statement.value)
        elif type(statement) is ast.Return:
            nodes = statement.value.elts if type(statement.value) in (ast.List, ast.Tuple) else [statement.value]
            returned = [expression(n) for n in nodes]
            if any(type(v) is not tuple or len(v) != period for v in returned):
                raise Unsupported("vector outputs required")
        else:
            raise Unsupported("unsupported statement: " + type(statement).__name__)
    if returned is None:
        raise Unsupported("return required")
    return returned

def difference(actual, reference):
    if len(actual) != len(reference):
        raise ValueError("sample count mismatch")
    pairs = []
    for row, expected in zip(actual, reference):
        if len(row) != len(expected):
            raise ValueError("output count mismatch")
        pairs.extend((finite(a), finite(b)) for a, b in zip(row, expected))
    errors = [abs(a-b) for a,b in pairs]
    return dict(values=len(pairs), maximum_absolute_error=max(errors, default=0),
                mean_absolute_error=sum(errors)/len(errors) if errors else 0,
                matches_frozen_tolerance=all(abs(a-b) <= 1e-5 + 1e-4 * abs(b) for a,b in pairs))

def audit(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    frozen = []
    for group in ("source_hashes", "tool_hashes", "retained_hashes"):
        hashes = plan[group]
        base = Path("/home/lhohy/Code/Poseidon") if group == "source_hashes" else Path("/")
        mismatches = [p for p,h in hashes.items() if sha(base/p) != h]
        if mismatches:
            raise ValueError("freeze mismatch: " + group)
        frozen.append(dict(group=group, files=len(hashes), mismatches=0))
    report_file = root / "functional15/report.json"
    group = json.loads(report_file.read_text())
    if group["status"] != "complete" or len(group["rows"]) != 15:
        raise ValueError("only a closed complete group may be audited")
    snapshots = {str(root / "plan.json"): sha(root / "plan.json"), str(report_file): sha(report_file)}
    rows = []
    counts = dict(generations=0, http_attempts=0, unresolved_http_attempts=0)
    for item in group["rows"]:
        directory = root / "functional15" / item["id"] / "tasks" / item["task_id"]
        task = json.loads((directory / "task.json").read_text())
        state = json.loads((directory / "state.json").read_text())
        result = item["result"]
        if state["status"] != result["status"]:
            raise ValueError("terminal state mismatch")
        if not (state["generations"] <= task["budget"]["generations"] and state["http_attempts"] <= task["budget"]["http_attempts"]):
            raise ValueError("case budget mismatch")
        counts["generations"] += state["generations"]
        counts["http_attempts"] += state["http_attempts"]
        checkpoints = [v for v in state["ledger"] if v["event"] == "provider_checkpoint"]
        outcomes = checkpoints[-1]["outcomes"] if checkpoints else []
        counts["unresolved_http_attempts"] += sum(v["status"] == "in_flight" for v in outcomes)
        row = dict(id=item["id"], task_id=item["task_id"], status=item["status"],
                   generations=state["generations"], http_attempts=state["http_attempts"],
                   request_id=task["request"]["request_id"], rounds=[])
        for path in directory.glob("*.json"):
            snapshots[str(path)] = sha(path)
        for index in range(state["generations"]):
            path = directory / ("validation-%s.json" % index)
            v = json.loads(path.read_text())
            record = dict(round=index, failure=v.get("failure"), validation_status=v["status"])
            evidence = Path(v["evidence"]) if v.get("evidence") else None
            if evidence and (evidence / "attempt-00/report.json").exists():
                raw = json.loads((evidence / "attempt-00/report.json").read_text())
                record.update(evidence=str(evidence), compiled=raw.get("compiled"),
                              encrypted_execution=raw.get("executed"), numerically_validated=raw.get("numerically_correct"))
                for rel in ("request.json", "attempt-00/report.json", "attempt-00/candidate.py", "attempt-00/output/inputs.json"):
                    if (evidence/rel).is_file():
                        snapshots[str(evidence/rel)] = sha(evidence/rel)
                if item["status"] != "passed" and raw.get("executed"):
                    request = json.loads((evidence / "request.json").read_text())
                    samples = json.loads((evidence / "attempt-00/output/inputs.json").read_text())
                    source = (evidence / "attempt-00/candidate.py").read_text()
                    layout = request["layout"]
                    try:
                        simulated = []
                        for inputs in samples:
                            vectors = evaluate(source, inputs, request["public_constants"], layout["input_slot_period"])
                            simulated.append([vectors[ciphertext][slot] for ciphertext,slot in layout["output_selectors"]])
                        comparison = raw["comparison"]
                        record["plaintext_candidate_vs_reference"] = difference(simulated, comparison["reference"])
                        record["ciphertext_vs_plaintext_candidate"] = difference(comparison["actual"], simulated)
                        record["failure_explanation"] = (
                            "source_math_mismatch_on_retained_inputs"
                            if not record["plaintext_candidate_vs_reference"]["matches_frozen_tolerance"]
                            else "source_math_matches_on_retained_inputs"
                        )
                    except (Unsupported, ValueError, KeyError, SyntaxError) as error:
                        record["diagnostic_subset_status"] = str(error)
                record["original_maximum_absolute_error"] = raw.get("comparison",{}).get("max_absolute_error")
            row["rounds"].append(record)
        if item["status"] == "passed":
            if not result.get("generation",{}).get("agent_generated") or not result.get("acceptance",{}).get("numerically_validated"):
                raise ValueError("real Agent numerical acceptance required")
            if not item.get("reloaded") or item.get("new_input_acceptance",{}).get("status") != "passed":
                raise ValueError("package reload and new input acceptance required")
        rows.append(row)
    if counts["generations"] != group["generations"] or counts["http_attempts"] != group["http_attempts"]:
        raise ValueError("group call counts mismatch")
    if counts["unresolved_http_attempts"]:
        raise ValueError("uncertain closed call")
    for path,h in snapshots.items():
        if sha(path) != h:
            raise ValueError("evidence changed during audit")
    return dict(format="poseidon-closed-live-source-diagnostic-v1",binding=plan["binding"],
                group="functional15", passed=group["passed"],failed=group["failed"],not_run=group["not_run"],
                calls=counts, prior_generation_reservations=group["prior_generations"],
                prior_http_attempts=group["prior_http_attempts"], frozen=frozen, rows=rows,
                evidence_hashes=snapshots, audit_provider_calls=0,audit_compilations=0,
                audit_encrypted_executions=0,
                interpretation_scope="Straight-line periodic arithmetic on retained private inputs, diagnostic only. The model reference and actual ciphertext outputs come from the original independent numerical acceptance.",
                no_source_execution=True,threshold=dict(atol=1e-5,rtol=1e-4))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("fresh audit report required")
    result = audit(args.root)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("group","passed","failed","calls","frozen")}))
