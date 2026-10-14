"""r1558 source-specific terminal audit; explicit read-only outer command.

Only the final audit receipt is written, exclusively in this isolated directory.
Expected values come from independent NumPy mathematics and CPU float64 Torch.
"""
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
ROOT = (Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
SCOPE = BASE / "benchmarks/tools/provenance_transport_live_v2"
HELPER = BASE / "benchmarks/tools/static_broadcast_live_v1"
BINDING = "831a65834ae225eb1acfdea3f27742e13d7180065052c2d14f0c53044e83cbe6"
SDK = "1a825a4e0d45ea7f11fd2f2a5cf760c229b8f159e6930ac2761cb11f65538297"
CASE_IDS = ('rank_axis_8', 'wide_tail_linear', 'weights_linear_10k', 'rank_permute_32')
REUSE_IDS = ('wide_tail_linear', 'weights_linear_10k', 'rank_permute_32')
SEEN = {}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), "regular_file")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate_json_key")
            value[key] = item
        return value
    data = path.read_bytes()
    fingerprint = hashlib.sha256(data).hexdigest()
    require(str(path) not in SEEN or SEEN[str(path)] == fingerprint, "evidence_changed")
    SEEN[str(path)] = fingerprint
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def observe(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), "regular_evidence")
    fingerprint = sha(path)
    require(str(path) not in SEEN or SEEN[str(path)] == fingerprint, "evidence_changed")
    SEEN[str(path)] = fingerprint
    return fingerprint


def bound(value):
    require(value.get("binding") == digest({k: v for k, v in value.items() if k != "binding"}),
            "json_binding")


def verified_scope():
    own = read(HERE / "scope.json")
    bound(own)
    require(own["live_scope"] == BINDING and own["sdk"] == SDK, "audit_scope")
    for rel, expected in own["tool_files"].items():
        require(sha(HERE / rel) == expected, "audit_tool_changed")
    plan = read(SCOPE / "live-plan.json")
    bound(plan)
    require(plan["binding"] == BINDING and plan["sdk"] == SDK, "live_scope_changed")
    # Check executable helper sources before importing any of them.
    for rel in ("qualification_scope.py",):
        path = SCOPE / rel
        require(sha(path) == plan["frozen_files"][str(path.relative_to(ROOT))], "scope_helper_changed")
    for rel in ("run.py", "scope.py"):
        path = HELPER / rel
        require(sha(path) == plan["frozen_files"][str(path.relative_to(ROOT))], "run_helper_changed")
    sys.path[:0] = [str(SCOPE), str(HELPER), str(BASE)]
    import qualification_scope as q
    checked = q.verify("live")
    require(checked == plan, "scope_verification_changed")
    require(tuple(c["id"] for c in plan["cases"]) == CASE_IDS
            and tuple(plan["package_new_input_cases"]) == REUSE_IDS, "four_case_scope")
    require((plan["budget"]["max_generations"], plan["budget"]["max_http"],
             plan["budget"]["max_compiler_calls"], plan["budget"]["max_encrypted_calls"])
            == (16, 64, 16, 22), "call_limits")
    require(plan["budget"]["global_result_bytes"] == 48 * 1024**3, "storage_limit")
    return plan


def terminal(plan):
    import run as live
    out = Path(plan["campaign"])
    require((out / "terminal.json").is_file(), "terminal_required_before_audit")
    end, owner = read(out / "terminal.json"), read(out / "owner.json")
    require(end["plan_binding"] == owner["plan_binding"] == BINDING, "owner_terminal_binding")
    require(owner["pid"] == 929719 and str(owner["start"]) == "25413817", "owner_identity")
    info = live.process_info(owner["pid"])
    require(info is None or str(info["start"]) != str(owner["start"])
            or info["state"] in ("Z", "X"), "owner_still_active")
    live.no_active()
    return out, end


def classify_compile(attempt, folder):
    """A command/empty log alone does not prove hecate-opt actually started."""
    log = folder / "compile.log"
    artifacts = folder / "output"
    command = attempt.get("compile_command", [])
    command_ok = isinstance(command, list) and command and command[0] == "/hecate-opt"
    text = ""
    if log.is_file():
        observe(log)
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 16384))
            text = stream.read(16384).decode("utf-8", "replace")
    outputs = [p for p in (artifacts / "lowered.mlir", artifacts / "lowered.ckks.mlir")
               if p.is_file() and p.stat().st_size > 0]
    for path in outputs:
        observe(path)
    diagnostic = bool(re.search(r"LLVM ERROR|loc\([^\n]*\): (?:error|warning)|"
                                r"\.mlir:[0-9]+:[0-9]+: (?:error|warning)|"
                                r"PLEASE submit a bug report|Stack dump:", text))
    success = attempt.get("compiled") is True
    if command_ok and (success or outputs or diagnostic):
        return dict(attempt="confirmed", success=success,
                    basis="compiler_returned_zero" if success else
                          "compiler_output_artifact" if outputs else "compiler_specific_diagnostic",
                    log_sha256=SEEN.get(str(log)))
    if command or log.exists() or outputs or success:
        return dict(attempt="unknown", success=success,
                    basis="invocation_not_proven_by_available_records", log_sha256=SEEN.get(str(log)))
    return dict(attempt="not_started", success=False, basis="no_invocation_evidence")


def classify_execution(attempt, folder):
    output = folder / "output"
    metadata = output / "execution-metadata.json"
    decrypted = output / "decrypted.npy"
    if metadata.is_file():
        value = read(metadata)
        if value.get("encrypted_execution") is True and value.get("backend") == "poseidon_cpu":
            require(value.get("executor_secret_key") is False, "executor_private_key")
            if decrypted.is_file():
                observe(decrypted)
            return dict(attempt="confirmed", success=decrypted.is_file(), basis="runtime_metadata")
    if attempt.get("executed") is True:
        return dict(attempt="unknown", success=False, basis="execution_claim_missing_metadata")
    if (folder / "execute.log").exists() or attempt.get("failure_layer") == "poseidon_runtime":
        if (folder / "execute.log").is_file():
            observe(folder / "execute.log")
        return dict(attempt="unknown", success=False, basis="runtime_launch_not_proven")
    return dict(attempt="not_started", success=False, basis="no_execution_evidence")


def ledger(task, state):
    """Preserve unknown outcomes. Settlement is host accounting, not billing proof."""
    generations, http = state["generations"], state["http_attempts"]
    require(type(generations) is int and 0 <= generations <= 4
            and type(http) is int and 0 <= http <= 16, "task_call_limits")
    entries = state["ledger"]
    require([x["index"] for x in entries if x.get("event") == "generation_reserved"]
            == list(range(generations)), "generation_reservations")
    require([x["index"] for x in entries if x.get("event") == "http_reserved"]
            == list(range(http)), "http_reservations")
    checkpoints = [x for x in entries if x.get("event") == "provider_checkpoint"]
    outcomes = checkpoints[-1].get("outcomes", []) if checkpoints else []
    require(len(outcomes) <= http and all(x.get("index") == i for i, x in enumerate(outcomes)),
            "checkpoint_indices")
    known = [x for x in outcomes if x.get("status") in ("response_received", "failed")]
    result = dict(task=str(task), state=state["status"], generations=generations, http_attempts=http,
                  settled_http=len(known), unknown_http=http-len(known),
                  outcomes=[dict(index=i, status=outcomes[i].get("status", "unknown"))
                            if i < len(outcomes) else dict(index=i, status="unknown") for i in range(http)])
    receipt = task / "provider-transport.json"
    if receipt.is_file():
        from application_transport_receipt import snapshot
        transport = read(receipt)
        require(snapshot(transport) == transport, "transport_receipt_integrity")
        require(transport.get("request_attempts") == http, "transport_reserved_count")
        result["transport_sha256"] = SEEN[str(receipt)]
        observed = transport["calls"]
        require(len(observed) == http, "transport_call_count")
        for i, call in enumerate(observed):
            require(call.get("index") == i, "transport_order")
            if i < len(outcomes):
                require(call.get("status") == outcomes[i].get("status"), "transport_checkpoint_status")
    elif http:
        result["transport_receipt_missing"] = True
    return result


def main():
    from process_resource_limits import limits
    limits()
    import numpy as np
    from application_backend import PoseidonBackend
    from candidate_bundle import check_bundle, check_execution_dependencies
    from candidate_contract import strict_json
    from deepseek_provider import public_request
    from generation_receipt import verify as verify_generation
    from provenance_json_transport import unpack
    from program_execution import named_outputs
    from numerical_results import compare
    from benchmark_math import evaluate
    from benchmark_graph import validate as validate_model
    from benchmark_torch import evaluate as torch_evaluate

    plan = verified_scope()
    out, end = terminal(plan)
    require(not (HERE / "report.json").exists(), "one_shot_report_exists")
    require(PoseidonBackend().identity_manifest() == read(SCOPE / "identity-snapshot.json"),
            "current_sdk_changed")
    report = read(out / "report.json")
    bound(report)
    require(report["plan_binding"] == BINDING and report["sdk"] == SDK
            and report["status"] in ("complete", "failed", "stopped"), "terminal_report")
    declared = {x["case"]: x for x in report["cases"]}
    require(len(declared) == len(report["cases"]) and set(declared) <= set(CASE_IDS),
            "reported_case_scope")
    rows, calls, attempts, issues, numerical = [], [], [], [], []
    examined_evidence = set()

    model_shape_cache = {}

    def output_shapes(model):
        # Authority is the frozen logical graph's validated value-shape map.
        # Neither candidate declarations nor returned arrays determine this map.
        key = digest(model)
        if key not in model_shape_cache:
            analysis = validate_model(model)
            shapes = {decl["name"]: list(analysis["shapes"][decl["value"]])
                      for decl in model["outputs"]}
            require(shapes == analysis["output_shapes"], "logical_output_shape_authority")
            model_shape_cache[key] = shapes
        return model_shape_cache[key]

    def flatten(model, values):
        shapes = output_shapes(model)
        require(set(values) == set(shapes), "named_output_names")
        for decl in model["outputs"]:
            require(np.asarray(values[decl["name"]]).shape == tuple(shapes[decl["name"]]),
                    "named_output_shape")
        return np.concatenate([np.asarray(values[x["name"]], dtype=np.float64).reshape(-1)
                               for x in model["outputs"]])

    def expected(model, inputs):
        a, b = flatten(model, evaluate(model, inputs)), flatten(model, torch_evaluate(model, inputs))
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        return a

    def input_values(request, sample):
        model, layout = request["model"], request["layout"]
        vectors = sample[None, :] if sample.ndim == 1 else sample
        require(vectors.shape == (len(layout["inputs"]), layout["input_slot_period"]),
                "physical_input_shape")
        flat = {d["name"]: np.zeros(math.prod(d["shape"]), dtype=np.float64) for d in model["inputs"]}
        seen = {k: np.zeros(len(v), dtype=np.bool_) for k, v in flat.items()}
        for bind, vector in zip(layout["inputs"], vectors):
            name, start, count = bind["name"], bind.get("offset", 0), bind["elements"]
            require(name in flat and 0 <= start and 0 < count <= len(vector)
                    and start + count <= len(flat[name]) and not seen[name][start:start+count].any()
                    and np.all(vector[count:] == 0), "chunk_binding_or_padding")
            flat[name][start:start+count] = vector[:count]
            seen[name][start:start+count] = True
        require(all(v.all() for v in seen.values()), "complete_input_binding")
        return {d["name"]: flat[d["name"]].reshape(d["shape"]) for d in model["inputs"]}

    def numeric_case(request, evidence, attempt, name):
        folder = evidence / ("attempt-%02d" % attempt["index"])
        actual_path = folder / "output/decrypted.npy"
        arrays = evidence / "arrays.npz"
        observe(arrays)
        with np.load(arrays, allow_pickle=False) as data:
            inputs = np.array(data["inputs"], copy=True)
            frozen = np.array(data["reference"], copy=True)
        require(len(inputs) == 4, "four_validation_inputs")
        require(read(evidence / "logical-bindings.json")["inputs"] == request["layout"]["inputs"],
                "physical_logical_binding")
        reference = np.asarray([expected(request["model"], input_values(request, x)) for x in inputs])
        np.testing.assert_allclose(reference, frozen, atol=1e-12, rtol=1e-12)
        observe(actual_path)
        actual = np.asarray([flatten(request["model"], v) for v in
                             named_outputs(request, np.load(actual_path, allow_pickle=False), 4)])
        cmp = compare(actual, reference, 1e-5, 1e-4)
        require(cmp["passed"] is (attempt.get("numerically_correct") is True),
                "original_numerical_status_disagrees")
        item = dict(case=name, phase="initial", evidence=str(evidence), groups=4,
                    values=cmp["compared_values"], passed=cmp["passed"],
                    max_absolute_error=cmp["max_absolute_error"],
                    decrypted_sha256=SEEN[str(actual_path)], frozen_arrays_sha256=SEEN[str(arrays)])
        numerical.append(item)
        return item

    def inspect_evidence(evidence, request, name):
        if str(evidence) in examined_evidence:
            return
        examined_evidence.add(str(evidence))
        require(evidence.is_relative_to(Path(plan["campaign"]).parent)
                and evidence.resolve() == evidence, "evidence_root")
        ep = evidence / "report.json"
        value = read(ep) if ep.is_file() else {}
        if value:
            require(value.get("request_id") == request["request_id"]
                    and value.get("backend") == "poseidon_cpu", "validation_request_backend")
            require(value.get("tolerance") == dict(atol=1e-5, rtol=1e-4), "tolerance_changed")
        indexed = {a["index"]: a for a in value.get("attempts", [])}
        paths = {int(p.name.split("-")[1]): p for p in evidence.glob("attempt-[0-9][0-9]") if p.is_dir()}
        for index in sorted(set(indexed) | set(paths)):
            a = indexed.get(index, {})
            folder = paths.get(index, evidence / ("attempt-%02d" % index))
            record = dict(case=name, evidence=str(evidence), index=index,
                          terminal_attempt_reported=index in indexed,
                          failure_layer=a.get("failure_layer"), status=a.get("status"),
                          compiler=classify_compile(a, folder), execution=classify_execution(a, folder))
            candidate_path = folder / "candidate.py"
            if candidate_path.exists():
                record["candidate_sha256"] = observe(candidate_path)
            for rel, fingerprint in a.get("artifact_hashes", {}).items():
                path = folder / "output" / rel
                require(path.resolve().is_relative_to(folder / "output") and observe(path) == fingerprint,
                        "validation_artifact_hash")
            gate = a.get("artifact_gate")
            if gate:
                require(gate["raw"]["security"] == gate["bound"]["security"] == "tc128"
                        and gate["binding"]["source_plan_sha256"] == gate["raw"]["plan_sha256"]
                        and gate["binding"]["plan_sha256"] == gate["bound"]["plan_sha256"], "artifact_binding")
            if a.get("executed") is True:
                record["numerical"] = numeric_case(request, evidence, a, name)
            attempts.append(record)

    for case in plan["cases"]:
        name, folder = case["id"], out / case["id"]
        row = declared.get(name)
        paths = sorted(folder.glob("tasks/*/state.json"))
        request = read(case["request_file"])
        require(public_request(request) == read(case["public_request_file"]), "public_request_projection")
        record = dict(case=name, declared_status=None if row is None else row["status"],
                      status=("started_unreported" if folder.exists() else "not_run") if row is None else row["status"],
                      origin="current_real_agent", source_kind=case["execution_source_kind"],
                      request_id=request["request_id"], numerical_pass=False, package_reloaded=False,
                      cache_verified=False, reuse_phases=[], raw_answers=[], failures=[])
        try:
            require(case["execution_source_kind"] == ("graph"if name=="rank_axis_8"else"trusted_pytorch"),"frozen_source_kind")
            if row:
                require(row["origin"] == "current_real_agent" and row["plan_binding"] == BINDING
                        and row["request_id"] == request["request_id"], "case_origin_binding")
                record["failure_layer"] = row.get("failure_layer")
                record["terminal_result"] = row.get("result", {}).get("status")
                record["failure"] = row.get("result", {}).get("failure")
            task_id = read(folder / "task-id.json")["task_id"] if (folder / "task-id.json").is_file() else None
            for state_file in paths:
                task, state = state_file.parent, read(state_file)
                calls.append(ledger(task, state))
                meta = unpack(read(task / "task.json"), purpose="application-task")
                bound(meta)
                require(meta["request"] == request, "task_request_binding")
                require(meta["validation_level"] == "numerical", "acceptance_level_downgraded")
                require(meta["model_source"]["kind"]==case["execution_source_kind"],"application_source_kind")
                if task.name == task_id:
                    record["task_status"] = state["status"]
                    record["generation_counts"] = dict(generations=state["generations"], http=state["http_attempts"])
                for g in sorted(task.glob("generation-[0-9]*.json")):
                    index = int(g.stem.split("-")[1])
                    value = read(g)
                    raw = value.get("raw")
                    r = task / ("response-%d.json" % index)
                    if r.exists():
                        response = read(r)
                        require(response.get("raw") == raw, "retained_response_changed")
                    item = dict(task=task.name, index=index, generation_file_sha256=SEEN[str(g)],
                                response_file_sha256=SEEN.get(str(r)), response_available=raw is not None)
                    if raw is not None:
                        item["raw_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
                        try:
                            answer = strict_json(raw)
                            item["candidate_sha256"] = hashlib.sha256(answer["hecate_source"].encode()).hexdigest()
                            item["answer_request_id_matches"] = answer["request_id"] == request["request_id"]
                        except (ValueError, TypeError, KeyError):
                            item["valid_candidate_json"] = False
                    record["raw_answers"].append(item)
                for vpath in sorted(task.glob("validation-*.json")):
                    v = read(vpath)
                    record["failures"].append(dict(file=str(vpath), status=v["status"], failure=v.get("failure"),
                                                   observed_stages=v.get("observed_stages")))
                    if v.get("evidence"):
                        inspect_evidence(Path(v["evidence"]), request, name)
                journal = task / "artifact-usage.json"
                if journal.exists():
                    j = read(journal)
                    bound(j)
                    for root in j["roots"][1:]:
                        ev = Path(root["path"])
                        if (ev / "arrays.npz").is_file():
                            inspect_evidence(ev, request, name)
            if row is not None and row["status"] == "passed":
                package = Path(row["package"])
                require(task_id is not None and package == folder / "tasks" / task_id / "program",
                        "package_owner")
                manifest, loaded = check_execution_dependencies(package)
                require(check_bundle(package) == manifest and loaded == request
                        and manifest["agent_generated"] is True
                        and manifest["validation"]["level"] == "numerical", "live_numerical_package")
                if case["execution_source_kind"]=="trusted_pytorch":
                    from candidate_bundle import PROVENANCE, check_provenance
                    # A program carries the complete sealed provenance payload.
                    # The model-import directory loader reads different files.
                    pv=unpack(read(package/PROVENANCE),purpose="model-provenance",limit=4*1024**2)
                    check_provenance(pv,request["model"])
                    require(pv["trusted_pytorch"]["manifest"]==read(case["source_manifest"]),"original_source_manifest")
                    cp=pv["reference_report"]["capture_metadata"]["public_value_expressions"]
                    require(cp["source_intake_version"]=="poseidon-trusted-pytorch-v36","complete_source_version")
                    proof=cp["logical_view_normalization"];bound(proof)
                    require(proof["normalized_sha256"]==digest(request["model"])and proof["production_active"]is True,"source_graph_authority")
                    require(cp["logical_view_reference"]["groups"]==16,"source_reference_scope")
                    observe(package/"model-provenance.json")
                    record["complete_source_provenance"]=dict(status="passed",format=cp["source_intake_version"],
                      proof_binding=proof["binding"],source_manifest_sha256=observe(case["source_manifest"]),
                      original_files=dict(pv["trusted_pytorch"]["manifest"]["files"]),independent_reference_groups=16)
                observe(package / "manifest.json")
                for rel in manifest["files"]:
                    observe(package / rel)
                require(all(any(p.endswith(suffix) for p in manifest["files"]) for suffix in
                            ("earth.mlir", "ckks.mlir", "runtime-plan.json", "operator-spec.json"))
                        and any("/bound/bundle/manifest.json" in p for p in manifest["files"]),
                        "complete_runtime_artifacts")
                task = package.parent
                source = (package / "candidate.py").read_text()
                receipt = read(task / "generation-provenance.json")
                meta = unpack(read(task / "task.json"), purpose="application-task")
                generation = verify_generation(receipt, request, source,
                                               generation_identity_sha256=meta["identity"]["generation"]["sha256"])
                state = read(task / "state.json")
                require(generation["generations"] == state["generations"]
                        and generation["http_attempts"] == state["http_attempts"]
                        and generation["http_attempts"] > 0, "fresh_generation_accounting")
                for answer in receipt["answers"]:
                    index = answer["index"]
                    require(read(task / ("generation-%d.json" % index)) == dict(raw=answer["raw"])
                            and read(task / ("response-%d.json" % index)) ==
                            dict(raw=answer["raw"], feedback=answer["feedback"]), "original_answer_binding")
                candidate_sha = observe(package / "candidate.py")
                matches = [x for x in attempts if x["case"] == name
                           and x.get("candidate_sha256") == candidate_sha and x.get("numerical", {}).get("passed")]
                require(len(matches) == 1, "qualified_exact_agent_candidate")
                record.update(numerical_pass=True, package_reloaded=True,
                              package_binding=manifest["binding"], candidate_sha256=candidate_sha,
                              generation=generation)
                if name in REUSE_IDS:
                    for phase, count in (("new_inputs", 3), ("repeat", 1)):
                        extra = read(folder / (phase + ".json"))
                        execution = extra["execution"]
                        require(extra["status"] == "passed" and execution["encrypted_execution"] is True
                                and execution["backend"] == "poseidon_cpu"
                                and execution["agent_calls"] == execution["compiler_calls"] == 0
                                and execution["executor_secret_key"] is False
                                and execution["bundle_binding"] == manifest["binding"], "reuse_execution")
                        evidence = Path(execution["evidence"])
                        stored = read(evidence / "report.json")
                        # Application adapters may append run-context/operational metadata.
                        require(all(execution.get(k) == v for k, v in stored.items()), "reuse_report_binding")
                        require(read(evidence / "output/execution-metadata.json") == execution["execution"]
                                and read(evidence / "output/resource-precheck.json") == execution["resource_precheck"]
                                and execution["resource_precheck"]["failure"] is None, "reuse_runtime_receipt")
                        reference = []
                        for i in range(count):
                            rng = np.random.default_rng(20261008 + i)
                            values = {d["name"]: rng.uniform(-.2, .2, d["shape"]).astype(np.float64)
                                      for d in request["model"]["inputs"]}
                            reference.append(expected(request["model"], values))
                        actual_path = evidence / "output/decrypted.npy"
                        observe(actual_path)
                        actual = np.asarray([flatten(request["model"], v) for v in
                                             named_outputs(request, np.load(actual_path, allow_pickle=False), count)])
                        returned = np.asarray([flatten(request["model"], v) for v in execution["outputs"]])
                        np.testing.assert_array_equal(actual, returned)
                        cmp = compare(actual, np.asarray(reference), 1e-5, 1e-4)
                        require(cmp["passed"], "reuse_numeric_failure")
                        record["reuse_phases"].append(phase)
                        numerical.append(dict(case=name, phase=phase, groups=count, values=cmp["compared_values"],
                                              passed=True, max_absolute_error=cmp["max_absolute_error"],
                                              decrypted_sha256=SEEN[str(actual_path)]))
                        attempts.append(dict(case=name, phase=phase, evidence=str(evidence),
                                             compiler=dict(attempt="not_started", success=False, basis="reload_no_compile"),
                                             execution=dict(attempt="confirmed", success=True, basis="runtime_metadata")))
                cache = read(folder / "cache.json")
                require(cache["package"]["reused"] is True
                        and cache["package"]["binding"] == manifest["binding"]
                        and cache["generations"] == cache["http_attempts"] == 0, "cache_no_regeneration")
                record["cache_verified"] = True
        except Exception as error:
            record["audit_error"] = dict(type=type(error).__name__, code=str(error)[:300])
            record["numerical_pass"] = False
            issues.append(dict(case=name, **record["audit_error"]))
        rows.append(record)

    # Read reuse receipts even when later cache/delivery checks failed. An absent
    # receipt after accepted generation is an unknown launch, never a zero count.
    for case in plan["cases"]:
        name = case["id"]
        if name not in REUSE_IDS:
            continue
        row = declared.get(name, {})
        after_acceptance = row.get("encrypted_execution") is True
        for phase in ("new_inputs", "repeat"):
            if any(x.get("case") == name and x.get("phase") == phase for x in attempts):
                continue
            receipt_path = out / name / (phase + ".json")
            if receipt_path.is_file():
                receipt = read(receipt_path)
                execution = receipt.get("execution", {})
                ev = Path(execution["evidence"])
                probe = classify_execution({}, ev)
                attempts.append(dict(case=name, phase=phase, evidence=str(ev),
                                     compiler=dict(attempt="not_started", success=False, basis="reload_no_compile"),
                                     execution=probe))
            elif after_acceptance:
                attempts.append(dict(case=name, phase=phase,
                                     compiler=dict(attempt="not_started", success=False, basis="reload_no_compile"),
                                     execution=dict(attempt="unknown", success=False,
                                                    basis="delivery_interrupted_before_receipt")))
    # Include every state even if an earlier per-case audit check failed.
    counted = {x["task"] for x in calls}
    for state_file in sorted(out.glob("*/tasks/*/state.json")):
        if str(state_file.parent) not in counted:
            try:
                calls.append(ledger(state_file.parent, read(state_file)))
            except Exception as error:
                issues.append(dict(scope="call_ledger", file=str(state_file), code=str(error)[:300]))
    generations = sum(x["generations"] for x in calls)
    http = sum(x["http_attempts"] for x in calls)
    unknown_http = sum(x["unknown_http"] for x in calls)
    def counts(key):
        return dict(confirmed_attempts=sum(x[key]["attempt"] == "confirmed" for x in attempts),
                    unknown_attempts=sum(x[key]["attempt"] == "unknown" for x in attempts),
                    successful=sum(x[key]["success"] for x in attempts))
    compiled, encrypted = counts("compiler"), counts("execution")
    require(generations <= 36 and http <= 144, "call_budget_exceeded")
    require(compiled["confirmed_attempts"] + compiled["unknown_attempts"] <= 36
            and encrypted["confirmed_attempts"] + encrypted["unknown_attempts"] <= 42, "backend_budget_exceeded")
    require(report["generations"] == generations and report["http_attempts"] == http,
            "terminal_call_totals_disagree")
    unresolved_tasks = [x for x in calls if x["state"] in ("running", "queued", "preparing")]
    incomplete = [x["case"] for x in rows if x["status"] == "started_unreported"]
    if unknown_http or unresolved_tasks:
        issues.append(dict(scope="settlement", unknown_http=unknown_http,
                           nonterminal_tasks=[x["task"] for x in unresolved_tasks]))
    if incomplete:
        issues.append(dict(scope="started_unreported", cases=incomplete))
    if compiled["unknown_attempts"] or encrypted["unknown_attempts"]:
        issues.append(dict(scope="backend_call_accounting", compiler=compiled, encrypted=encrypted))
    require(PoseidonBackend().identity_manifest() == read(SCOPE / "identity-snapshot.json"), "end_sdk_changed")
    verified_scope()
    for path, fingerprint in SEEN.items():
        require(sha(path) == fingerprint, "evidence_changed_during_audit")
    passed_checks = [x for x in numerical if x["passed"]]
    result = dict(format="poseidon-capacity-shallow-live-terminal-audit-v3",
                  audit_status="incomplete" if issues else "passed",
                  evaluation_status=report["status"], plan_binding=BINDING, sdk=SDK,
                  terminal_exit_code=end["exit_code"], report_binding=report["binding"],
                  planned=4, passed=sum(x["numerical_pass"] for x in rows),
                  failed=sum(x["status"] == "failed" for x in rows),
                  not_run=sum(x["status"] == "not_run" for x in rows),
                  started_unreported=len(incomplete), skipped=0, cases=rows,
                  generations=generations, http_attempts=http, unknown_http=unknown_http,
                  calls_resolved=not unknown_http, call_ledgers=calls,
                  compiler=compiled, encrypted=encrypted, attempts=attempts,
                  call_count_scope="confirmed and unknown launches; success is counted separately",
                  origin="current_real_agent",
                  source_scope="one original graph failure and three full trusted PyTorch v36 submissions",
                  complete_source_cases_planned=3,
                  complete_source_cases_passed=sum(x.get("complete_source_provenance",{}).get("status")=="passed"for x in rows),
                  provenance_scope="retained host transport and exact returned bytes; not provider-signed or billing proof",
                  independent_references="NumPy mathematical interpreter and CPU float64 Torch; no rule candidate",
                  output_shape_authority="frozen model validated value shapes; not candidate output shapes",
                  reference_tolerance=dict(atol=1e-12, rtol=1e-12),
                  numerical_tolerance=dict(atol=1e-5, rtol=1e-4),
                  numerical_checks=numerical, passed_input_groups=sum(x["groups"] for x in passed_checks),
                  passed_output_values=sum(x["values"] for x in passed_checks),
                  maximum_absolute_error=max((x["max_absolute_error"] for x in passed_checks), default=None),
                  package_reuse_completed=sum(len(x["reuse_phases"]) == 2 for x in rows),
                  package_reuse_planned=3, audit_issues=issues, evidence_sha256=SEEN,
                  tool_sha256=sha(Path(__file__)), new_provider_calls=0, new_compiler_calls=0,
                  new_encrypted_calls=0, historical_records_rewritten=False)
    result["binding"] = digest(result)
    with (HERE / "report.json").open("xb") as stream:
        stream.write(canonical(result))
    print(json.dumps({k: result[k] for k in ("audit_status", "evaluation_status", "planned", "passed",
                                          "failed", "not_run", "started_unreported", "generations",
                                          "http_attempts", "unknown_http", "compiler", "encrypted",
                                          "package_reuse_completed")}, ensure_ascii=False))
    return 0 if not issues else 2


def outer():
    plan = verified_scope()
    terminal(plan)
    require(not (HERE / "report.json").exists(), "one_shot_report_exists")
    from hecate_python_env import VENV, enter_nix
    command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ' + shlex.join(
        [str(VENV / "bin/python"), "-B", str(Path(__file__).resolve()), "--inside"])
    return enter_nix(command, seconds=300, environment="poseidon-cpu")


if __name__ == "__main__":
    require(sys.argv[1:] in (["--audit"], ["--inside"]), "explicit_audit_required")
    if sys.argv[1:] == ["--inside"]:
        # Imports in main need the checked paths; no backend activity happens here.
        verified_scope()
        raise SystemExit(main())
    raise SystemExit(outer())
