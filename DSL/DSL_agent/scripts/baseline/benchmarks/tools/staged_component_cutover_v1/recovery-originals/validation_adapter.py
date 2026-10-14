"""Trusted validation adapter shared by CLI and component qualification.

The context is private host state; never put it in a provider request.
No provider, credentials or paid transport is constructed by this module.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from candidate_contract import strict_json, trace_payload_json, validate_candidate, request_rotations, request_input_names
from numerical_results import require, compare, dump
from compiler_configuration import verify_artifact_configuration
from hecate_python_env import ROOT, WORK, VENV, digest
from poseidon_runtime_paths import PROFILE

class CompilationAccepted(Exception):
    """Internal control flow; cannot originate in a candidate process."""

@dataclass(frozen=True)
class ValidationContext:
    result: Path
    request: dict
    report: dict
    frozen: dict
    fixed: dict
    waterline: int
    keys: Path
    unified: bool
    native_run: Callable
    verify_frozen: Callable
    validation_level: str = "numerical"

class CandidateValidationAdapter:
    def __init__(self, context):
        if not isinstance(context, ValidationContext):
            raise TypeError("ValidationContext required")
        require(context.validation_level in ("numerical", "compiled"), "Validation level")
        self.context = context

    def evaluate(self, raw, index):
        import numpy as np
        context = self.context
        result, request, report = context.result, context.request, context.report
        frozen, fixed, waterline = context.frozen, context.fixed, context.waterline
        keys, unified = context.keys, context.unified
        native_run, verify_frozen = context.native_run, context.verify_frozen
        def read_json(path):
            from candidate_contract import MAX_BYTES
            require(path.is_file() and path.stat().st_size <= MAX_BYTES, "JSON file size limit")
            return strict_json(path.read_text())
        item = dict(index=index, status="failed", parsed=False, source_parsed=False, checked=False, compiled=False,
                    executed=False, numerically_correct=False)
        report["attempts"].append(item)
        attempt = result / f"attempt-{index:02d}"
        attempt.mkdir()
        output = attempt / "output"
        output.mkdir()
        stage = "response_parse"
        artifact_hashes = {}
        try:
            verify_frozen()
            candidate = strict_json(raw)
            item["parsed"] = True
            stage = "source_parse"
            require(type(candidate) is dict and type(candidate.get("hecate_source")) is str,
                    "Missing Hecate source string")
            require(len(candidate["hecate_source"].encode()) <= 65536, "Source size limit")
            ast.parse(candidate["hecate_source"])
            item["source_parsed"] = True
            stage = "static_check"
            item["static_check"] = validate_candidate(candidate, request)
            item["checked"] = True
            trace_payload = attempt / "trace-payload.json"
            trace_payload.write_text(trace_payload_json(request, candidate))
            frozen[str(trace_payload.relative_to(result))] = digest(trace_payload)
            (attempt / "candidate.py").write_text(candidate["hecate_source"])
            stage = "dsl_trace"
            code = native_run(trace_payload, output, [str(VENV / "bin/python"), "/app/candidate_trace.py"],
                               attempt / "trace.log")
            require(code == 0, f"Isolated Hecate tracing failed (exit {code}); see trace.log")
            item["trace"] = read_json(output / "trace-evidence.json")
            if "upstream_helpers" in request:
                from upstream_candidate_helpers import verify_events
                item['upstream_helper_trace']=verify_events(read_json(output/'upstream-calls.json'),request,candidate['hecate_source'])
                if 'upstream_exercise' in request:
                    from upstream_helper_coverage import verify_trace
                    item['upstream_helper_coverage']=verify_trace(item['static_check']['upstream_exercise'],read_json(output/'upstream-calls.json'))
            if unified and "construction_profile" in request:
                observed=read_json(output/"construction.json")
                require(observed==item["static_check"]["public_construction"],
                        "Actual public normalization differs from checked metadata")
                require(digest(output/"normalized-source.py")==observed["normalized_sha256"],
                        "Normalized source hash mismatch")
                derived=read_json(output/"derived-constants.json")
                from benchmark_graph import digest as data_digest
                require(data_digest(derived)==observed["derived_constants_sha256"],
                        "Derived constants hash mismatch")
                events=read_json(output/"public-construction-events.json")
                require(events["profile"]==request["construction_profile"] and
                        events["normalized_sha256"]==observed["normalized_sha256"] and
                        events["candidate_python_executed"] is False,
                        "Actual public construction trace identity")
                item["public_construction_trace"]=dict(profile=events["profile"],events=len(events["events"]),
                    real_frontend_checked=item["trace"]["frontend"]=="real_Hecate",
                    source_and_normalization_bound=True,directed_features_validated=False)
                if "construction_exercise" in request:
                    from unified_public_coverage import verify_trace_coverage
                    item["public_expression_coverage"]=verify_trace_coverage(
                        item["static_check"]["construction_exercise"],events)
                    item["public_construction_trace"]["directed_features_validated"]=True

            if request['task']=='hecate-periodic-packed-native-synthesis-v2' or (unified and 'construction_exercise' in request and "construction_profile" not in request):
                if unified:
                    from unified_native_coverage import verify_trace_coverage
                else:
                    from packed_native_exercises import verify_trace_coverage
                records={key:read_json(output/file) for key,file in (
                    ('storage','native-array-events.json'),('star','native-star-events.json'),
                    ('augmented','native-augmented-events.json'),('mutation','native-array-mutation-events.json'),
                    ('calls','native-call-events.json'))}
                item['packed_native_coverage']=verify_trace_coverage(
                    item['static_check']['construction_exercise'],records)
            if request['task'] == 'hecate-native-function-synthesis-v2':
                from native_function_exercises import verify_trace_coverage
                item['native_call_coverage'] = verify_trace_coverage(
                    item['static_check']['construction_exercise'], read_json(output/'native-call-events.json'))
            if request['task'] == 'hecate-native-function-synthesis-v4':
                from native_array_exercises import verify_trace_coverage
                item['native_array_coverage'] = verify_trace_coverage(
                    item['static_check']['construction_exercise'], read_json(output/'native-array-events.json'))
            if request['task'] == 'hecate-native-function-synthesis-v8':
                from native_star_exercises import verify_trace_coverage
                item['native_star_coverage'] = verify_trace_coverage(
                    item['static_check']['construction_exercise'], read_json(output/'native-star-events.json'))
            stage = "compiler"
            from poseidon_runtime_compile import command as compile_command, prepare_entrypoint
            from poseidon_validation import gate, prepare_inputs, execute
            require(request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu",
                    "Poseidon CPU configuration required; no legacy backend execution")
            item["runtime_plan_entry"] = prepare_entrypoint(output, request["compiler_configuration"])
            command = compile_command(waterline, entry_policy=request["compiler_configuration"].get("runtime_plan_entry_policy"))
            item["compile_command"] = command
            code = native_run(trace_payload, output, command, attempt / "compile.log")
            require(code == 0, f"Isolated compiler failed (exit {code}); see compile.log")
            item["compiled"] = True
            stage = "artifact_gate"
            item["artifact_gate"] = gate(output, request)
            if context.validation_level == "compiled":
                artifact_hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
                item.update(status="passed", validation_level="compiled", compiled_validated=True)
                # finally still verifies frozen inputs/artifacts and persists the attempt.
                # The common feedback construction below handles integrity overrides.
                raise CompilationAccepted()
            artifact_hashes = prepare_inputs(result, output, request)
            stage = "poseidon_runtime"
            execution = execute(trace_payload, output, request, native_run, attempt / "execute.log")
            item.update(executed=True, execution=execution)
            stage = "numerical_comparison"
            with np.load(result / "arrays.npz", allow_pickle=False) as data:
                item["comparison"] = compare(np.load(output / "decrypted.npy", allow_pickle=False),
                                             data["reference"], fixed["atol"], fixed["rtol"])
                if 'execution_abi' in request['layout'] and len(request['layout']['output_shape'])>1:
                    from packed_input_abi import decode_output
                    logical=np.load(output/'decrypted-logical.npy',allow_pickle=False)
                    expected=decode_output(np.load(output/'decrypted.npy',allow_pickle=False),request['layout'])
                    require(np.array_equal(logical,expected) and logical.shape==data['reference_logical'].shape,
                            'Logical output shape/order differs from declared binding')
                    item['logical_output_shape']=list(logical.shape[1:])
                    item['logical_output_sha256']=digest(output/'decrypted-logical.npy')
            require(item["comparison"]["passed"], "Frozen numerical tolerance failed; inspect reduction/layout/operator semantics")
            item.update(status="passed", numerically_correct=True)
        except CompilationAccepted:
            pass
        except Exception as error:
            category = ("candidate" if stage in ("response_parse", "source_parse", "static_check", "numerical_comparison")
                        else "pipeline_unclassified")
            if isinstance(error, (OSError, TimeoutError)):
                category = "infrastructure"
            item.update(failure_layer=stage, diagnostic=str(error)[:2000], category=category)
            if stage == "compiler":
                # Advisory only: preserve the original failure and frozen integrity gate.
                from earth_failure_diagnostic import diagnose
                try:
                    log_path = attempt / "compile.log"
                    with log_path.open("rb") as log:
                        log.seek(max(0, log_path.stat().st_size - 4000))
                        text = log.read(4000).decode("utf-8", errors="replace")
                    diagnosis = diagnose(text, PROFILE.read_bytes())
                    if diagnosis is not None:
                        item["compiler_diagnosis"] = diagnosis
                except OSError:
                    pass  # A missing diagnostic must not hide the original failure.
        finally:
            try:
                verify_frozen()
                require(all(digest(output / n) == h for n, h in artifact_hashes.items()), "Compiled artifact/input mutated")
            except Exception as error:
                item.update(status="failed", failure_layer="integrity", category="integrity",
                            diagnostic=str(error), numerically_correct=False)
            item["artifact_hashes"] = artifact_hashes
            dump(attempt / "report.json", item)
        # Feedback deliberately excludes reference/inputs/actual vectors and host paths.
        feedback = dict(status=item["status"], layer=item.get("failure_layer", "complete"),
                        category=item.get("category"), diagnostic=item.get("diagnostic", "All numerical checks passed"))
        if item.get("failure_layer")=="static_check" and "generation_guidance" in request:
            from unified_graph_contract import static_repair_hint
            hint=static_repair_hint(candidate["hecate_source"],request,item.get("diagnostic",""))
            if hint:feedback["diagnostic"]+="\nContract clarification:\n"+hint
        if (item.get("failure_layer") in ("compiler", "artifact_gate") and
                request.get("generation_guidance", {}).get("version") in ("explicit-v9","explicit-v10")):
            if request["generation_guidance"]["version"]=="explicit-v10":
                from unified_poseidon_guidance import repair_hint
            else:
                from unified_expression_guidance import repair_hint
            hint = repair_hint(request, candidate["hecate_source"], stage=item["failure_layer"])
            if hint:
                feedback["diagnostic"] += "\nExpression guidance:\n" + hint
        for private_path, label in ((str(result), "<run>"), (str(ROOT), "<source>"), (str(WORK), "<work>")):
            feedback["diagnostic"] = feedback["diagnostic"].replace(private_path, label)
        if "compiler_diagnosis" in item:
            d = item["compiler_diagnosis"]
            feedback["diagnostic"] += (
                "\nCompiler type diagnosis (advisory): "
                + ", ".join(d["failed_conditions"])
                + "; Earth lhs accumulated scale " + str(d["compiler_accumulated_scale"])
                + ", fixed compiler budget " + str(d["compiler_budget"])
                + ". Earth level counts consumed levels, not RuntimePlan/Poseidon data-modulus indices."
                + " This is not a backend modulus-capacity test or proof against every equivalent expression.")
        if "comparison" in item:
            feedback["numerical_summary"] = {k: item["comparison"][k] for k in
                                            ("mae", "max_absolute_error", "compared_values", "passed")}
        # Compiler/frontend text is untrusted diagnostic data, never an instruction
        # or executable repair command. Runtime logs may mention key handling and
        # are deliberately not included in the public feedback envelope.
        log_name = {"dsl_trace": "trace.log", "compiler": "compile.log"}.get(item.get("failure_layer"))
        if log_name and (attempt / log_name).is_file():
            with (attempt / log_name).open("rb") as log:
                log.seek(max(0, (attempt / log_name).stat().st_size - 4000))
                diagnostic = log.read(4000).decode("utf-8", errors="replace")
            for private_path, label in ((str(result), "<run>"), (str(ROOT), "<source>"), (str(WORK), "<work>")):
                diagnostic = diagnostic.replace(private_path, label)
            feedback["tool_diagnostic"] = dict(trust="untrusted_tool_output", text=diagnostic)
        return feedback
