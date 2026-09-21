"""Model -> DeepSeek Hecate generation -> isolated Dacapo/SEAL -> compare/repair.

--deepseek makes real service calls; --prepare/--self-test/--replay remain offline.
Accepts preset schema-1 cases or user-defined schema-2/3 graphs, not arbitrary Python.
"""
from platform_config import identity, require_python_packages
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

from candidate_contract import (MAX_BYTES, ReplayProvider, make_request, strict_json,
                                validate_candidate, run_feedback_loop, request_rotations, request_input_names)
from hecate_python_env import ROOT, WORK, VENV, digest, enter_nix
from python_compiler_smoke import BUILD, SOURCE, logged
from seal_artifact_gate import inspect_artifacts, require
from seal_cpu_golden import KEY_BUILD, PROFILE, WATERLINE, compare, dump
from native_execution_slots import native_slot, NATIVE_CONCURRENCY
from compiler_configuration import (CONFIGURATIONS, configuration, options as compiler_options,
                                    verify_execution_configuration, verify_artifact_configuration)

SCRIPT = ROOT / "scripts/baseline/run_candidate.py"


def read_json(path):
    require(path.is_file() and path.stat().st_size <= MAX_BYTES, "JSON file size limit")
    return strict_json(path.read_text())


def select_provider(args, request, responses=None, api_key=""):
    if not args.deepseek:
        return ReplayProvider(responses)
    from deepseek_provider import Config, DeepSeekProvider, HTTPSTransport
    require(bool(api_key), "Set DEEPSEEK_API_KEY locally before --deepseek; no offline fallback")
    config = Config(model=args.model, service_provider=args.provider, reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens,
                    max_calls=args.max_repairs + 1, timeout_seconds=args.api_timeout, stream=getattr(args, 'stream', False),
                    provider_retries=getattr(args, 'provider_retries', 0),
                    loopback_proxy_port=getattr(args, 'loopback_proxy_port', 0))
    # --deepseek explicitly requests generation from this public test model.
    transport = HTTPSTransport(api_key=api_key, approved_request_id=request["request_id"],
                              approved_config=config, enabled=True)
    return DeepSeekProvider(config, transport=transport)


def update_provider_report(report, provider):
    if provider is None:
        return
    report.update(provider=provider.kind, agent_calls=provider.agent_calls,
                  artifact_replay_only=provider.kind != "deepseek_api")
    if callable(getattr(provider, "metrics", None)):
        report["provider_metrics"] = provider.metrics()
    report["llm_generation_validated"] = bool(
        provider.kind == "deepseek_api" and provider.agent_calls > 0 and report["status"] == "passed"
        and any(item.get("numerically_correct") and item.get("executed")
                and item.get("execution", {}).get("encrypted_execution") for item in report["attempts"]))


def print_outcome(report):
    from deepseek_provider import SAFE_FAILURE_CODES
    status = report.get("status", "unknown")
    print(f"Candidate status: {status}", flush=True)
    calls = report.get("provider_metrics", {}).get("calls", [])
    if status == "provider_failed" and calls:
        code = calls[-1].get("error", "transport_worker_failed")
        http_codes = {f"http_status_{i}" for i in range(100, 600)}
        safe = code if code in SAFE_FAILURE_CODES | http_codes else "transport_worker_failed"
        print(f"Provider failure: {safe}", flush=True)


def inside(args):
    # Remove the credential from the process environment before native children.
    # Only the HTTPS transport retains it and passes it to its worker over stdin.
    api_key = os.environ.pop("DEEPSEEK_API_KEY", "") if args.deepseek else ""
    import numpy as np
    import torch
    import candidate_sandbox as sandbox
    from fx_to_hecate import translate
    from model_catalog import validate_descriptor
    from run_model_batch import prepare_case

    os.umask(0o077)
    torch.set_num_threads(2)
    require_python_packages(torch, np)
    result = Path(tempfile.mkdtemp(prefix="agent-deepseek-" if args.deepseek else "candidate-replay-",
                                   dir=WORK / "results"))
    print(f"Candidate evidence: {result}", flush=True)
    report = dict(platform_identity=identity(), status="running", provider="deepseek_api" if args.deepseek else "scripted_replay", agent_calls=0,
                  backend="upstream_SEAL_HEVM_CPU", poseidon_gpu_validated=False, attempts=[],
                  metadata_observer_sha256=digest(KEY_BUILD / 'libseal_golden_metadata.so'),
                  metadata_observer_source_sha256=digest(ROOT / 'scripts/baseline/seal_keys/metadata.cpp'),
                  artifact_replay_only=not args.deepseek, llm_generation_validated=False,
                  source_hashes={p.name: digest(p) for p in [SCRIPT, *[ROOT / "scripts/baseline" / x for x in
                      (*sandbox.MODULES, "candidate_sandbox.py", "fx_to_hecate.py", "run_model_batch.py",
                       "deepseek_provider.py", "deepseek_http_worker.py", "model_graph.py", "model_catalog.py", "chunked_model.py", "packed_model.py", "logical_reshape.py", "packed_spatial.py",
                       "batch_norm_ops.py", "concat_ops.py", "tensor_permutation.py", "multi_input_fixtures.py", "native_execution_slots.py")]]},
                  native_resources=dict(concurrency=NATIVE_CONCURRENCY, acquisitions=0, wait_seconds=0.0))
    def native_run(*positional, **keywords):
        with native_slot(WORK / 'cache/agent-native-slots', metrics=report['native_resources']):
            return sandbox.run(*positional, **keywords)
    provider = None
    frozen = {}
    phase = "model_description"
    try:
        descriptor = read_json(args.case)
        validate_descriptor(descriptor)
        if descriptor['schema']==5:
            report['packed_observer_sha256']=digest(KEY_BUILD/'libseal_packed_metadata.so')
            report['packed_observer_source_sha256']=digest(ROOT/'scripts/baseline/seal_keys/packed_metadata.cpp')
            report['packed_key_helper_sha256']=digest(KEY_BUILD/'seal_packed_keys')
            report['packed_key_source_sha256']=digest(ROOT/'scripts/baseline/seal_keys/packed_keys.cpp')
        dump(result / "model.json", descriptor)
        phase = "model_reference"
        model, shape = prepare_case(descriptor, result)
        phase = "fx_translation"
        payload = translate(model, shape)
        chosen_name = getattr(args, 'compiler_configuration', None)
        compiler_config = configuration(chosen_name, digest(PROFILE)) if chosen_name is not None else None
        waterline = compiler_config['waterline'] if compiler_config is not None else WATERLINE
        request = make_request(payload, descriptor, digest(PROFILE), str(model),
                               compiler_configuration=compiler_config,
                               extended_arithmetic=getattr(args, 'extended_arithmetic', False),
                               public_construction=getattr(args, 'public_construction', False),
                               function_composition=getattr(args, 'function_composition', False),
                               closures=getattr(args, 'closures', False),
                               call_binding=getattr(args, 'call_binding', False),
                               public_iteration=getattr(args, 'public_iteration', False),
                               function_literals=getattr(args, 'function_literals', False),
                               public_sequences=getattr(args, 'public_sequences', False),
                               public_numbers=getattr(args, 'public_numbers', False),
                               public_control=getattr(args, 'public_control', False),
                               public_strings=getattr(args, 'public_strings', False),
                               public_polynomial=getattr(args, 'public_polynomial', False),
                               object_arrays=getattr(args, 'object_arrays', False),
                               public_mappings=getattr(args, 'public_mappings', False),
                               object_arithmetic=getattr(args, 'object_arithmetic', False),
                               scalar_conversion=getattr(args, 'scalar_conversion', False),
                               object_unary=getattr(args, 'object_unary', False),
                               native_functions=getattr(args, 'native_functions', False),
                               native_arrays=getattr(args, 'native_arrays', False),
                               native_starred=getattr(args, 'native_starred', False),
                               native_array_arithmetic=getattr(args, 'native_array_arithmetic', False),
                               native_public_loops=getattr(args, 'native_public_loops', False),
                               native_scalar_augmented=getattr(args, 'native_scalar_augmented', False),
                               native_array_mutation=getattr(args, 'native_array_mutation', False),
                               construction_exercise=getattr(args, 'construction_exercise', None))
        # Shapes/modules are public model semantics, not the deterministic DSL answer.
        (result / "model-structure.txt").write_text(str(model))
        dump(result / "request.json", request)
        (result / "rule-answer.py").write_text(payload["hecate_source"])
        frozen = {name: digest(result / name) for name in
                  ("model.json", "weights.npz", "arrays.npz", "request.json", "rule-answer.py")}
        fixed = read_json(SOURCE / "fixtures.json")
        report.update(request_id=request["request_id"], tolerance={k: fixed[k] for k in ("atol", "rtol")},
                      compiler_profile_sha256=digest(PROFILE), waterline=waterline,
                      runtime_sha256=digest(BUILD / "lib/libSEAL_HEVM.so"),
                      frontend_sha256=digest(BUILD / "lib/libHecateFrontend.so"),
                      frontend_source_sha256=digest(ROOT / "third_party/dacapo/tools/frontend.cpp"),
                      frontend_python_sha256=digest(ROOT / 'third_party/dacapo/python/hecate/hecate/expr.py'))
        if compiler_config is not None:
            report['compiler_configuration'] = compiler_config
        verify_execution_configuration(request, digest(PROFILE), waterline)
        if args.prepare:
            report["status"] = "request_prepared_not_generated"
            return 0
        if args.deepseek:
            phase = "provider_setup"
            provider = select_provider(args, request, api_key=api_key)
            def checkpoint_provider():
                update_provider_report(report, provider)
                dump(result / 'report.json', report)
            provider.on_attempt = checkpoint_provider
            api_key = ""

        # The private sentinel is not a key. Probe before any candidate tracing.
        phase = "sandbox_preflight"
        (result / "private-sentinel.txt").write_text("sandbox must not see this")
        probe_payload = result / "probe-payload.json"
        dump(probe_payload, dict(workspace=str(ROOT), sentinel=str(result / "private-sentinel.txt"),
                                net_ns=os.readlink("/proc/self/ns/net"), pid_ns=os.readlink("/proc/self/ns/pid")))
        probe_dir = result / "probe"
        probe_dir.mkdir()
        code = native_run(probe_payload, probe_dir, [str(VENV / "bin/python"), "/app/candidate_worker.py", "probe"],
                           result / "probe.log")
        require(code == 0, "Sandbox preflight failed; no unsandboxed fallback; see probe.log")
        report["sandbox_probe"] = read_json(probe_dir / "probe.json")

        # Reuse the approved built key helper, without implicit configure/install.
        phase = "key_setup"
        keys = result / "private-keys"
        keys.mkdir(mode=0o700)
        key_command=([str(KEY_BUILD/'seal_packed_keys'),str(keys),str(request['layout']['input_slot_period'])]
                     if descriptor['schema']==5 else [str(KEY_BUILD/'seal_golden_keys'),str(keys),
                                                      *[str(s) for s in request_rotations(request)]])
        with native_slot(WORK / 'cache/agent-native-slots', metrics=report['native_resources']):
            require(logged(key_command, result / "parameters.json", 120) == 0,
                    "Key setup failed")
        report["parameters"] = read_json(result / "parameters.json")
        phase = "response_input"
        if args.self_test:
            require(descriptor.get("family") == "linear", "Scripted fault test requires catalog Linear")
            candidate = dict(schema=1, request_id=request["request_id"], hecate_source=payload["hecate_source"])
            wrong = dict(candidate, hecate_source=candidate["hecate_source"].replace("rotate(1)", "rotate(2)"))
            responses = ["{invalid json", json.dumps(wrong), json.dumps(candidate)]
            report["scenario"] = "scripted invalid JSON -> wrong reduction -> saved rule answer"
            dump(result / "replay-responses.json", responses)
        elif args.golden_file:
            require(args.golden_file.is_file() and args.golden_file.stat().st_size <= 65536,
                    "Golden source must be a bounded local file")
            responses = [json.dumps(dict(schema=1, request_id=request["request_id"],
                                        hecate_source=args.golden_file.read_text()))]
            report["scenario"] = "manual golden through isolated candidate pipeline; not Agent inference"
        elif not args.deepseek:
            responses = read_json(args.replay)
        if not args.deepseek:
            provider = select_provider(args, request, responses=responses)
        phase = "feedback_loop"

        def verify_frozen():
            if descriptor['schema']==5:
                require(digest(KEY_BUILD/'libseal_packed_metadata.so')==report['packed_observer_sha256'] and
                        digest(KEY_BUILD/'seal_packed_keys')==report['packed_key_helper_sha256'],
                        'Packed verification/key helper changed during execution')
            require(all((result / k).is_file() and digest(result / k) == v for k, v in frozen.items()),
                    "Immutable reference/request/weights changed")
            require(digest(PROFILE) == report["compiler_profile_sha256"], "Compiler profile changed")
            verify_execution_configuration(request, digest(PROFILE), waterline)
            require(report.get('compiler_configuration') == compiler_config and report['waterline'] == waterline,
                    'Compiler configuration report changed')
            require(digest(KEY_BUILD / 'libseal_golden_metadata.so') == report['metadata_observer_sha256'],
                    'Metadata observer changed during execution')
            require(digest(BUILD / "lib/libHecateFrontend.so") == report["frontend_sha256"], "Frontend library changed")
            require(digest(ROOT / 'third_party/dacapo/python/hecate/hecate/expr.py') == report['frontend_python_sha256'],
                    'Frontend Python changed')

        def evaluate(raw, index):
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
                dump(trace_payload, dict(request=request, candidate=candidate))
                frozen[str(trace_payload.relative_to(result))] = digest(trace_payload)
                (attempt / "candidate.py").write_text(candidate["hecate_source"])
                stage = "dsl_trace"
                code = native_run(trace_payload, output, [str(VENV / "bin/python"), "/app/candidate_trace.py"],
                                   attempt / "trace.log")
                require(code == 0, f"Isolated Hecate tracing failed (exit {code}); see trace.log")
                item["trace"] = read_json(output / "trace-evidence.json")
                if request['task']=='hecate-periodic-packed-native-synthesis-v2':
                    from packed_native_exercises import verify_trace_coverage
                    records={key:read_json(output/file) for key,file in (
                        ('storage','native-array-events.json'),('star','native-star-events.json'),
                        ('augmented','native-augmented-events.json'),('mutation','native-array-mutation-events.json'))}
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
                command = ["/hecate-opt", "/out/candidate_trace.mlir", "--eva", "--ckks-config=/profile.json",
                           f"--waterline={waterline}", "--enable-debug-printer", "--mlir-disable-threading",
                           "--verify-each", "-o", "/out/lowered.mlir"]
                item["compile_command"] = command
                code = native_run(trace_payload, output, command, attempt / "compile.log")
                require(code == 0, f"Isolated compiler failed (exit {code}); see compile.log")
                item["compiled"] = True
                stage = "artifact_gate"
                hevm, cst = output / "lowered._hecate_golden.hevm", output / "_hecate_golden.cst"
                require(hevm.stat().st_size <= 1024**2 and cst.stat().st_size <= 1024**2, "Artifact size limit")
                from cipher_abi import artifact_options
                item["artifact_gate"] = inspect_artifacts(hevm.read_bytes(), cst.read_bytes(),
                                                          rotation_steps=request_rotations(request),
                                                          expected_inputs=len(request_input_names(request)),
                                                          **artifact_options(request['layout']))
                require(len(item["artifact_gate"]["res_dst"]) == request["layout"]["output_ciphertexts"],
                        "Artifact return count mismatch")
                verify_artifact_configuration(request, item['artifact_gate'], digest(PROFILE))
                with np.load(result / "arrays.npz", allow_pickle=False) as data:
                    # Runtime sees only inputs, never the plaintext reference.
                    np.savez(output / "arrays.npz", inputs=data["inputs"])
                artifact_hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
                stage = "seal_runtime"
                code = native_run(trace_payload, output, [str(VENV / "bin/python"), "/app/candidate_worker.py", "execute"],
                                   attempt / "execute.log", 150, keys)
                require(code == 0, f"Isolated SEAL execution failed (exit {code}); see execute.log")
                item.update(executed=True, execution=read_json(output / "execution.json"))
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
            except Exception as error:
                category = ("candidate" if stage in ("response_parse", "source_parse", "static_check", "numerical_comparison")
                            else "pipeline_unclassified")
                if isinstance(error, (OSError, TimeoutError)):
                    category = "infrastructure"
                item.update(failure_layer=stage, diagnostic=str(error)[:2000], category=category)
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
            for private_path, label in ((str(result), "<run>"), (str(ROOT), "<source>"), (str(WORK), "<work>")):
                feedback["diagnostic"] = feedback["diagnostic"].replace(private_path, label)
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

        def record(index, raw, feedback):
            attempt = result / f"attempt-{index:02d}"
            (attempt / "response.txt").write_text(raw)
            dump(attempt / "feedback.json", feedback)
            update_provider_report(report, provider)
            dump(result / "report.json", report)
            print(f"attempt {index}: {feedback['status']} / {feedback['layer']}", flush=True)

        report["loop"] = run_feedback_loop(request, provider, evaluate, record, max_repairs=args.max_repairs)
        report["status"] = report["loop"]["status"]
        if args.self_test:
            observed = [a.get("failure_layer", "complete") for a in report["attempts"]]
            require(observed == ["response_parse", "numerical_comparison", "complete"], "Fault scenario did not reach expected gates")
        return 0 if report["status"] == "passed" else 1
    except Exception as error:
        category = "input" if phase in ("model_description", "model_reference", "fx_translation", "response_input") else "infrastructure"
        report.update(status=category + "_failed", failure_layer=phase, diagnostic=str(error))
        print(str(error), flush=True)
        return 1
    finally:
        update_provider_report(report, provider)
        report["frozen_hashes"] = frozen
        dump(result / "report.json", report)
        print_outcome(report)
        # Native execution is finished and durable numerical evidence is saved.
        # No intermediate/random key material needs to survive the experiment.
        if (result / 'private-keys').exists():
            from result_retention import cleanup_run
            try:
                cleaned = cleanup_run(result, WORK / 'results')
                print(f"Key cleanup: {cleaned['bytes']} bytes", flush=True)
            except (OSError, ValueError) as error:
                print(f"Key cleanup deferred: {type(error).__name__}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument('--compiler-configuration', choices=tuple(CONFIGURATIONS),
                        help='Explicit immutable compiler profile; omission preserves legacy waterline40 requests')
    parser.add_argument('--extended-arithmetic', action='store_true',
                        help='Versioned reverse arithmetic, rebinding and augmented assignment grammar')
    parser.add_argument('--public-construction', action='store_true',
                        help='Versioned bounded public loops, branches and containers; includes extended arithmetic')
    parser.add_argument('--function-composition', action='store_true',
                        help='Versioned lexical helper calls and returns; includes public construction')
    parser.add_argument('--closures', action='store_true',
                        help='Versioned nested helpers, late-bound closures and nonlocal cells')
    parser.add_argument('--call-binding', action='store_true',
                        help='Versioned defaults, keywords, positional-only/keyword-only and varargs; includes closures')
    parser.add_argument('--public-iteration', action='store_true',
                        help='Versioned scoped comprehensions and lazy public iterators; includes call binding')
    parser.add_argument('--function-literals', action='store_true',
                        help='Versioned lambda/function-valued calls and stable public-key sorted')
    parser.add_argument('--public-sequences', action='store_true',
                        help='Versioned public sequence slicing, concatenation, repetition and slice writes')
    parser.add_argument('--object-arithmetic', action='store_true', help='Use request-v20 object arithmetic contract')
    parser.add_argument('--public-mappings', action='store_true', help='Use request-v19 public mapping contract')
    parser.add_argument('--construction-exercise', help='Frozen targeted construction exercise ID')
    parser.add_argument('--object-arrays', action='store_true', help='Use request-v18 object storage contract')
    parser.add_argument('--public-polynomial', action='store_true',
                        help='Enable bounded public Chebyshev data and GenPoly construction')
    parser.add_argument('--public-strings', action='store_true',
                        help='Enable bounded public coefficient/tree string parsing')
    parser.add_argument('--public-control', action='store_true',
                        help='Enable bounded public control, short circuit and immutable mapping keys')
    parser.add_argument('--public-numbers', action='store_true',
                        help='Versioned public real/array arithmetic and audited derived constants')
    parser.add_argument('--object-unary', action='store_true')
    parser.add_argument('--native-functions', action='store_true', help='Opt-in typed native decorated-function core')
    parser.add_argument('--native-arrays', action='store_true', help='Opt-in native Expr object-array return contract')
    parser.add_argument('--native-starred', action='store_true', help='Opt-in typed native positional argument unpacking')
    parser.add_argument('--native-array-arithmetic', action='store_true', help='Opt-in native Expr object-array arithmetic')
    parser.add_argument('--native-public-loops', action='store_true', help='Opt-in construction-time native public range loops')
    parser.add_argument('--native-scalar-augmented', action='store_true', help='Opt-in scalar native +=, -= and *= rebinding')
    parser.add_argument('--native-array-mutation', action='store_true', help='Opt-in alias-aware native ndarray inplace operations')
    parser.add_argument('--scalar-conversion', action='store_true')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Export public request without inference/execution")
    mode.add_argument("--self-test", action="store_true", help="Scripted faults/answer, NOT Agent-generated repairs")
    mode.add_argument("--replay", type=Path, help="JSON array of 1..4 raw response strings; never run commands")
    mode.add_argument("--golden-file", type=Path, help="Check a manual Hecate fragment through real isolated execution; no API")
    mode.add_argument("--deepseek", "--live", dest="deepseek", action="store_true", help="Real model API generation; --provider selects credential and route")
    parser.add_argument("--provider", choices=("deepseek",), default="deepseek")
    parser.add_argument("--model", default="deepseek-flash", choices=("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"))
    parser.add_argument("--max-repairs", type=int, default=3, choices=range(4))
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high")
    parser.add_argument("--api-timeout", type=float, default=1200)
    parser.add_argument('--provider-retries', type=int, choices=range(4), default=0)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--loopback-proxy-port', type=int, default=0,
                        help='Explicit DeepSeek CONNECT via 127.0.0.1; zero keeps the default route')
    args = parser.parse_args(argv)
    from deepseek_provider import MODEL_OUTPUT_LIMITS
    if args.max_tokens is None:
        args.max_tokens = MODEL_OUTPUT_LIMITS[args.model]
    if args.deepseek:
        from deepseek_provider import Config
        Config(model=args.model, service_provider=args.provider, reasoning_effort=args.reasoning_effort, max_calls=args.max_repairs + 1, max_tokens=args.max_tokens,
               timeout_seconds=args.api_timeout, stream=args.stream, provider_retries=args.provider_retries,
               loopback_proxy_port=args.loopback_proxy_port)
    return args


def forward_options(args):
    options = ["--case", str(args.case.resolve()), "--max-repairs", str(args.max_repairs)]
    options += compiler_options(args)
    if getattr(args, 'native_starred', False):
        options += ['--native-starred']
    if getattr(args, 'native_array_arithmetic', False):
        options += ['--native-array-arithmetic']
    if getattr(args, 'native_public_loops', False):
        options += ['--native-public-loops']
    if getattr(args, 'native_scalar_augmented', False):
        options += ['--native-scalar-augmented']
    if getattr(args, 'native_array_mutation', False):
        options += ['--native-array-mutation']
    if getattr(args, 'native_arrays', False):
        options += ['--native-arrays']
    if getattr(args, 'native_functions', False):
        options += ['--native-functions']
    if getattr(args, 'extended_arithmetic', False):
        options += ['--extended-arithmetic']
    if getattr(args, 'public_construction', False):
        options += ['--public-construction']
    if getattr(args, 'function_composition', False):
        options += ['--function-composition']
    if getattr(args, 'closures', False):
        options += ['--closures']
    if getattr(args, 'call_binding', False):
        options += ['--call-binding']
    if getattr(args, 'public_iteration', False):
        options += ['--public-iteration']
    if getattr(args, 'function_literals', False):
        options += ['--function-literals']
    if getattr(args, 'public_sequences', False):
        options += ['--public-sequences']
    if getattr(args, 'object_unary', False):
        options += ['--object-unary']
    if getattr(args, 'scalar_conversion', False):
        options += ['--scalar-conversion']
    if getattr(args, 'object_arithmetic', False):
        options += ['--object-arithmetic']
    if getattr(args, 'public_mappings', False):
        options += ['--public-mappings']
    if getattr(args, 'construction_exercise', None):
        options += ['--construction-exercise', args.construction_exercise]
    if getattr(args, 'object_arrays', False):
        options += ['--object-arrays']
    if getattr(args, 'public_polynomial', False):
        options += ['--public-polynomial']
    if getattr(args, 'public_strings', False):
        options += ['--public-strings']
    if getattr(args, 'public_control', False):
        options += ['--public-control']
    if getattr(args, 'public_numbers', False):
        options += ['--public-numbers']
    if args.deepseek:
        if args.stream:
            options += ['--stream']
        options += ["--deepseek", "--provider", args.provider, "--model", args.model, "--max-tokens", str(args.max_tokens),
                    "--api-timeout", str(args.api_timeout), "--reasoning-effort", args.reasoning_effort,
                    '--provider-retries', str(getattr(args, 'provider_retries', 0)),
                    '--loopback-proxy-port', str(getattr(args, 'loopback_proxy_port', 0))]
    elif args.prepare:
        options += ["--prepare"]
    elif args.self_test:
        options += ["--self-test"]
    elif args.golden_file:
        options += ["--golden-file", str(args.golden_file.resolve())]
    else:
        options += ["--replay", str(args.replay.resolve())]
    return shlex.join(options)


def main():
    args = parse_args()
    require(Path.cwd().resolve() == ROOT, f"Requires cwd {ROOT}")
    if args.inside and args.deepseek and not os.environ.get("DEEPSEEK_API_KEY"):
        print("Missing DEEPSEEK_API_KEY inside the isolated environment.", file=sys.stderr)
        return 2
    if not args.inside:
        command = (f'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '
                   f'{shlex.quote(str(VENV / "bin/python"))} {shlex.quote(str(SCRIPT))} --inside {forward_options(args)}')
        if not args.deepseek:
            # Offline modes never discover or read credentials.
            return enter_nix(command, seconds=900)
        from agent_credentials import CredentialError, load_api_key
        previous = os.environ.get("DEEPSEEK_API_KEY")
        try:
            os.environ["DEEPSEEK_API_KEY"] = load_api_key(ROOT, provider=args.provider)
        except CredentialError as error:
            print(str(error), file=sys.stderr)
            return 2
        try:
            from deepseek_provider import generation_deadline
            return enter_nix(command, seconds=1800 + generation_deadline(args.api_timeout, args.max_repairs + 1, args.provider_retries),
                             keep_env=("DEEPSEEK_API_KEY",))
        finally:
            if previous is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = previous
    require(bool(os.environ.get("IN_NIX_SHELL")) and Path(sys.prefix) == VENV, "Requires pinned Nix/venv")
    return inside(args)


if __name__ == "__main__":
    raise SystemExit(main())
