#!/usr/bin/env python3
"""Controlled eager/Waterline CKKS benchmarks; generate once, run on either host.

Every variant keeps the same ciphertext work, placement, weights and Release
instructions. Eager encodes/uploads all plaintexts before execution. Waterline
replaces only those Encode/Transfer pairs with complete PreparePlaintext.
"""
import argparse
import copy
import hashlib
import json
import os
import resource
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MIB, GIB = 2**20, 2**30


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            sha.update(block)
    return "sha256:" + sha.hexdigest()


def renumber(plan):
    ordinal = 0
    for phase in ("initialization", "execution", "finalization"):
        for op in plan[phase]:
            op["ordinal"] = ordinal
            ordinal += 1


def binary_copy(args):
    """Use the production CKKSMF01 manifest wire format with unchanged blobs."""
    import shutil
    import struct
    original = args.suite.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(original/"bundle", args.output/"bundle", copy_function=os.link)
    for name in ("operator-spec.json", "fixture.json", "mock.json"):
        if (original/name).exists():
            shutil.copyfile(original/name, args.output/name)
    manifest = json.loads((original/"bundle/manifest.json").read_text())
    if manifest["bundle_format_version"] != 2:
        raise ValueError("binary manifest comparison requires packed V2 bundle")
    blobs = manifest["blobs"]
    manifest["blobs"] = []
    metadata = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    with (args.output/"bundle/manifest.bin").open("wb") as stream:
        stream.write(b"CKKSMF01"+struct.pack("<II", 1, len(metadata))+metadata+struct.pack("<Q", len(blobs)))
        for blob in sorted(blobs, key=lambda b:b["content"]):
            stream.write(bytes.fromhex(blob["content"].removeprefix("sha256:")))
            stream.write(struct.pack("<QQ", blob["offset"], blob["byte_length"]))
    suite = json.loads((original/"suite.json").read_text())
    for case in suite["cases"]:
        plan = json.loads((original/case["plan"]).read_text())
        plan["plaintext_bundle"].pop("manifest_sha256")
        plan["plaintext_bundle"]["manifest_format"] = "binary"
        save(args.output/case["plan"], plan)
    suite["manifest_comparison"] = dict(source_suite=str(original), only_change="plaintext_bundle manifest_format and manifest file; raw weight bytes and ciphertext work unchanged",
        source_manifest_sha256=digest(original/"bundle/manifest.json"), binary_manifest_sha256=digest(args.output/"bundle/manifest.bin"),
        data_sha256=digest(original/"bundle/data.bin"))
    save(args.output/"suite.json", suite)
    print(args.output, flush=True)


def waterline(eager, capacity, reserve, plain_mib, window, backend):
    plan = copy.deepcopy(eager)
    desc = {v["id"]: v for v in plan["values"]}
    encodes = {o["output"]: o for o in plan["initialization"] if o["kind"] == "encode"}
    removed, prepare, init = set(), [], []
    for op in plan["initialization"]:
        if op["kind"] == "encode":
            removed.add(op["output"])
        elif op["kind"] in ("transfer", "replicate") and op["inputs"][0] in encodes:
            for output in op["outputs"]:
                assert desc[output]["kind"] == "plaintext"
                prepare.append(dict(kind="prepare_plaintext", output=output,
                                    payload=encodes[op["inputs"][0]]["payload"]))
        elif op["kind"] == "release" and op["value"] in encodes:
            pass
        else:
            init.append(op)
    plan["values"] = [v for v in plan["values"] if v["id"] not in removed]
    plan["initialization"] = []
    plan["execution"] = prepare + init + plan["execution"]
    plan["format_version"] = 4
    devices = plan["target"]["device_counts"][0]
    plan["waterline"] = dict(device_capacity_bytes=[capacity]*devices,
        device_reserve_bytes=[reserve]*devices,
        device_fractions=[plain_mib*MIB/capacity]*devices,
        host_budget_bytes=4*GIB, pinned_budget_bytes=512*MIB,
        raw_budget_bytes=128*MIB, lookahead_groups=window, backend=backend)
    renumber(plan)
    return plan


def generate(args):
    import numpy as np
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    if args.kind == "qproj":
        import shutil
        source_dir = args.source_directory.resolve()
        source = next(source_dir.glob("*.runtime-plan.json"))
        eager = json.loads(source.read_text())
        eager.pop("waterline")
        eager["format_version"] = 3
        descriptions = {v["id"]: v for v in eager["values"]}
        next_id = max(int(v) for v in descriptions)+1
        for phase in ("initialization", "execution", "finalization"):
            kept = []
            for op in list(eager[phase]):
                if op["kind"] != "prepare_plaintext":
                    kept.append(op)
                    continue
                host_id = str(next_id)
                next_id += 1
                desc = copy.deepcopy(descriptions[op["output"]])
                desc.update(id=host_id, place=dict(kind="host", rank=0))
                eager["values"].append(desc)
                eager["initialization"] += [dict(kind="encode", output=host_id, payload=op["payload"]),
                    dict(kind="transfer", transfer_id=host_id, hint="point_to_point", inputs=[host_id], outputs=[op["output"]],
                         sources=[desc["place"]], destinations=[descriptions[op["output"]]["place"]], output_kinds=["plaintext"]),
                    dict(kind="release", value=host_id)]
            if phase != "initialization":
                eager[phase] = kept
        renumber(eager)
        shutil.copytree(Path(str(source).removesuffix(".runtime-plan.json")+".bundle"), out/"bundle")
        shutil.copyfile(source_dir.parent/"profiles/operator-spec.json", out/"operator-spec.json")
        shutil.copyfile(source_dir/"fixture.json", out/"fixture.json")
        save(out/"eager.json", eager)
        cases.append(dict(name="eager", plan="eager.json", runner="qwen"))
        for budget in args.budgets:
            for backend in args.backends:
                name=f"wl-{budget}MiB-{backend}-w{args.window}"
                save(out/f"{name}.json", waterline(eager, args.capacity_gib*GIB,
                     args.reserve_gib*GIB, budget, args.window, backend))
                cases.append(dict(name=name, plan=f"{name}.json", runner="qwen"))
    elif args.kind == "mlp":
        original_spec = ROOT / "test-results/e2e-new-profile-e63ec18/profiles/operator-spec.json"
        # The historical Q17/drop4 MLP predates integer rescale normalization.
        # Q21/drop5 provides precision for the current scale160 -> scale40 path.
        s = json.loads(original_spec.read_text())
        s["spec_id"] = "poseidon-mlp-q21-waterline-benchmark"
        s["context"]["rns_moduli_log2"] = [30]*21
        s["levels"] = dict(lower_bound=5, upper_bound=20)
        s["operators"]["rescale"]["max_levels_per_op"] = 5
        for op in s["operators"].values():
            for key, numbers in op.items():
                if isinstance(numbers, list):
                    op[key] = numbers + [numbers[-1]]*(21-len(numbers))
        for boot in s["boot_profiles"]:
            boot.update(input_level_min=5, input_level_max=20, output_level=20)
            boot["latency_us_by_input_level"] = [10000]*21
        spec = out/"operator-spec.json"
        save(spec, s)
        fixture = ROOT / "test-results/e2e-new-profile-e63ec18/fixture/input.json"
        mock = fixture.parent / "mock-result.json"
        folder = ROOT / "test-results/memory-v1-review/1gpu"
        source = folder / "mlp.reuse._hecate_MLP.runtime-plan.json"
        if args.devices == 2:
            compiler = ROOT / "third_party/ckks-runtime/third_party/dacapo/build/nix/bin/hecate-opt"
            # Start before physical-level materialization and placement.
            original = ROOT / "test-results/e2e-new-profile-e63ec18/artifacts/1gpu/mlp.optimized.ckks.mlir"
            # This historical dump used pretty debug locations, which are not
            # parseable MLIR. Remove only those location suffixes.
            import re
            original_text = re.sub(r" \[/.*\]$", "", original.read_text(), flags=re.MULTILINE)
            original_text = re.sub(r" /[^\s]+:\d+:\d+(?=\))", "", original_text)
            original_text = re.sub(r' "earth"\(/[^\n]*\)$', "", original_text, flags=re.MULTILINE)
            original = out/"logical.ckks.mlir"
            original.write_text(original_text)
            prefix = out / "mlp"
            passes = [f"materialize-ckks-physical-levels{{operator-spec={spec} levels-per-logical-level=5}}",
                f"assign-ckks-placement{{device-counts=2 operator-spec={spec} boot-profile={s['boot_profiles'][0]['profile_id']}}}",
                "materialize-ckks-communication", "plan-runtime-memory",
                f"emit-runtime-plan{{prefix={prefix} plan-format=json bundle-format=pack plan-id=991 ntt=true target-id={s['target_id']} operator-spec-id={s['spec_id']} operator-spec-version={s['version']} operator-spec-sha256={digest(spec)} context-id={s['context']['context_id']} boot-profile={s['boot_profiles'][0]['profile_id']} boot-implementation=decrypt_reencrypt}}"]
            command = [str(compiler), f"--ckks-config={ROOT/'test-results/v100-profile-e63ec18/applied-profile/compiler-profile.json'}", str(original), "-p=builtin.module(func.func("+",".join(passes)+"))", "-o", str(out/"mlp.mlir")]
            save(out/"compile-command.json", command)
            with (out/"compile.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            source = out / "mlp._hecate_MLP.runtime-plan.json"
        eager = json.loads(source.read_text())
        if args.devices == 1:
            for v in eager["values"]:
                v["level"] = v["level"]*5//4
            for phase in ("initialization", "execution", "finalization"):
                for op in eager[phase]:
                    if "target_level" in op.get("attrs", {}):
                        op["attrs"]["target_level"] = op["attrs"]["target_level"]*5//4
            eager["target"]["operator_spec"].update(id=s["spec_id"], source_sha256=digest(spec))
        import shutil
        shutil.copytree(Path(str(source).removesuffix(".runtime-plan.json")+".bundle"), out/"bundle")
        shutil.copyfile(fixture, out/"fixture.json")
        shutil.copyfile(mock, out/"mock.json")
        save(out/"eager.json", eager)
        cases.append(dict(name="eager", plan="eager.json", runner="mlp"))
        for budget in args.budgets:
            for backend in args.backends:
                name = f"wl-{budget}MiB-{backend}-w{args.window}"
                save(out/f"{name}.json", waterline(eager, args.capacity_gib*GIB,
                     args.reserve_gib*GIB, budget, args.window, backend))
                cases.append(dict(name=name, plan=f"{name}.json", runner="mlp"))
    else:
        spec = json.loads(args.operator_spec.read_text())
        save(out/"operator-spec.json", spec)
        context = spec["context"]
        n, q = context["poly_degree"], len(context["rns_moduli_log2"])
        slots = n//2
        host = dict(kind="host", rank=0)
        dev = dict(kind="device", rank=0, index=0)
        eager = dict(format_version=2, plan_id="181026", target=dict(target_id=spec["target_id"],
            world_size=1, device_counts=[1], capability_version=1,
            operator_spec=dict(id=spec["spec_id"], version=spec["version"], source_sha256=digest(out/"operator-spec.json"))),
            values=[], external_inputs=["0"], final_outputs=[], initialization=[], execution=[], finalization=[])
        def value(kind, place, scale=40):
            idx = str(len(eager["values"]))
            eager["values"].append(dict(id=idx, kind=kind, context=context["context_id"],
                level=q-1, scale_log2=scale, components=2 if kind=="ciphertext" else 1, ntt=True, place=place))
            return idx
        value("ciphertext", host)
        x = value("ciphertext", dev)
        def transfer(source, target, source_place, target_place, kind):
            return dict(kind="transfer", transfer_id=target, hint="point_to_point", inputs=[source], outputs=[target],
                sources=[source_place], destinations=[target_place], output_kinds=[kind])
        eager["initialization"].append(transfer("0", x, host, dev, "ciphertext"))
        eager["initialization"].append(dict(kind="release", value="0"))
        bundle = out/"bundle"
        bundle.mkdir()
        rng = np.random.default_rng(20261018)
        inputs = rng.uniform(-0.2, 0.2, slots)
        expected = np.zeros(slots)
        blobs = []
        accumulator = None
        with (bundle/"data.bin").open("wb") as data:
            for i in range(args.count):
                weights = rng.uniform(-0.1, 0.1, slots)/np.sqrt(args.count)
                raw = weights.astype("<f8").tobytes()
                content = "sha256:"+hashlib.sha256(raw).hexdigest()
                blobs.append(dict(content=content, byte_length=len(raw), offset=data.tell()))
                data.write(raw)
                expected += inputs*weights
                hp, dp = value("plaintext", host), value("plaintext", dev)
                eager["initialization"] += [dict(kind="encode", output=hp, payload=dict(kind="bundle", content=content)),
                    transfer(hp, dp, host, dev, "plaintext"), dict(kind="release", value=hp)]
                product = value("ciphertext", dev, 80)
                eager["execution"] += [dict(kind="compute", op="mul_cp", inputs=[x, dp], output=product, place=dev),
                    dict(kind="release", value=dp)]
                if accumulator is None:
                    accumulator = product
                else:
                    output = value("ciphertext", dev, 80)
                    eager["execution"] += [dict(kind="compute", op="add_cc", inputs=[accumulator, product], output=output, place=dev),
                        dict(kind="release", value=accumulator), dict(kind="release", value=product)]
                    accumulator = output
        eager["execution"].append(dict(kind="release", value=x))
        eager["final_outputs"] = [accumulator]
        save(bundle/"manifest.json", dict(bundle_format_version=2, bundle_id="waterline-benchmark", version=1,
             pack_byte_length=(bundle/"data.bin").stat().st_size, blobs=blobs))
        eager["plaintext_bundle"] = dict(id="waterline-benchmark", version=1, manifest_sha256=digest(bundle/"manifest.json"))
        renumber(eager)
        save(out/"eager.json", eager)
        save(out/"fixture.json", dict(format_version=1, model=dict(name="weighted_sum_memory_pressure", synthetic_weights=True, complete_qwen=False),
            context=dict(poly_degree=n, q_moduli_bits=context["rns_moduli_log2"], p_moduli_bits=[30]*8, default_scale_log2=40),
            tolerances=dict(atol=0.001, rtol=0.0001),
            inputs=[dict(name="input", value_id=0, values=inputs.tolist())],
            expected_outputs=[dict(name="weighted_sum", value_id=int(accumulator), values=expected.tolist())]))
        cases.append(dict(name="eager", plan="eager.json", runner="qwen"))
        for budget in args.budgets:
            for backend in args.backends:
                name=f"wl-{budget}MiB-{backend}-w{args.window}"
                save(out/f"{name}.json", waterline(eager, args.capacity_gib*GIB, args.reserve_gib*GIB, budget, args.window, backend))
                cases.append(dict(name=name, plan=f"{name}.json", runner="qwen"))
    save(out/"suite.json", dict(kind=args.kind, devices=args.devices, capacity_gib=args.capacity_gib,
        reserve_gib=args.reserve_gib, window=args.window, count=args.count,
        plaintext_resident_bytes=sum((v["level"]+1)*json.loads((out/"operator-spec.json").read_text())["context"]["poly_degree"]*4
            for v in eager["values"] if v["kind"]=="plaintext" and v["place"]["kind"]=="device"),
        source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), cases=cases))
    print(out, flush=True)


def run(args):
    suite_dir = args.suite.resolve()
    suite = json.loads((suite_dir/"suite.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.cuda, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        POSEIDON_GPU_MLP_ALLOW_NO_BOOT="1", POSEIDON_RUNTIME_ROTATION_KEY_POLICY="binary",
        POSEIDON_RUNTIME_DEVICE_WORKERS=str(suite["devices"]),
        POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB=str(args.pool_mib), POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=str(args.pool_mib))
    summary = dict(suite=suite, cuda=args.cuda, pool_mib=args.pool_mib,
        timing_scope="Fresh process per sample; forward includes setup/init/online; key generation and input encryption separate; OS file cache not flushed.",
        runner_hashes={name:digest(args.build/"bin"/f"poseidon_runtime_gpu_{name}_e2e") for name in {c["runner"] for c in suite["cases"]}}, samples=[], statistics={})
    save(args.output/"environment.json", dict(environment={k:v for k,v in env.items() if k.startswith(("POSEIDON_","OMP_","CUDA_VISIBLE","OPENBLAS"))},
        nvidia_smi=subprocess.check_output(["nvidia-smi"],text=True)))
    selected = [c for c in suite["cases"] if not args.cases or c["name"] in args.cases]
    summary["selected_case_names"] = [c["name"] for c in selected]
    summary["requested_repeats"] = args.repeats
    for repeat in range(args.repeats):
        shift = repeat%len(selected)
        for case in selected[shift:]+selected[:shift]:
            if repeat >= case.get("repeat_count", args.repeats):
                continue
            folder=args.output/f"r{repeat}-{case['name']}"
            folder.mkdir()
            command=[str(args.build/"bin"/f"poseidon_runtime_gpu_{case['runner']}_e2e"), str(suite_dir/case["plan"]), str(suite_dir/"operator-spec.json")]
            if case["runner"]=="mlp":
                command += [str(suite_dir/"bundle"), str(suite_dir/"fixture.json"), str(suite_dir/"mock.json"), str(folder/"report.json")]
            else:
                command += [str(suite_dir/"fixture.json"), str(folder/"report.json"), str(folder/"decoded.json"), "--bundle", str(suite_dir/"bundle")]
            save(folder/"command.json",command)
            started=time.monotonic()
            cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN)
            with (folder/"run.log").open("w") as log:
                try:
                    result=subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    status=result.returncode
                except subprocess.TimeoutExpired:
                    status=124
            report=json.loads((folder/"report.json").read_text()) if (folder/"report.json").exists() else {}
            cpu_after=resource.getrusage(resource.RUSAGE_CHILDREN)
            sample=dict(case=case["name"], repeat=repeat, exit_code=status, passed=not status and report.get("passed",False),
                process_wall_ms=1000*(time.monotonic()-started),
                process_cpu_seconds=cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,
                report=report)
            if sample["passed"]:
                t=report["runtime_timing"]
                if case["runner"]=="mlp":
                    sample.update(forward_ms=1000*sum(t[k] for k in ("setup_seconds","initialization_seconds","online_execution_seconds")),
                        online_ms=1000*t["online_execution_seconds"], init_ms=1000*t["initialization_seconds"], setup_ms=1000*t["setup_seconds"])
                else:
                    sample.update(forward_ms=report["runtime_ms"], online_ms=t["online_execution_ms"], init_ms=t["initialization_ms"], setup_ms=t["setup_ms"])
            summary["samples"].append(sample)
            for name in {s["case"] for s in summary["samples"]}:
                rows=[s for s in summary["samples"] if s["case"]==name and s["passed"]]
                if rows:
                    summary["statistics"][name]={k:dict(median=statistics.median(s[k] for s in rows),min=min(s[k] for s in rows),max=max(s[k] for s in rows)) for k in ("forward_ms","online_ms","init_ms","setup_ms","process_wall_ms")}
            save(args.output/"summary.json",summary)
            print(f"{case['name']} repeat={repeat} exit={status} forward_ms={sample.get('forward_ms')} online_ms={sample.get('online_ms')}",flush=True)
            if not sample["passed"]:
                print((folder/"run.log").read_text()[-2500:],flush=True)
                raise RuntimeError(f"benchmark failed: {folder}")


def poll_existing(args):
    """Pair existing private wait binaries on the same binary-manifest input."""
    import resource
    variants = json.loads((args.variants/"summary.json").read_text())["variants"]
    args.output.mkdir(parents=True, exist_ok=False)
    plans = sorted(args.suite.glob("wl-64MiB-gpu-w*.json"))
    if len(plans) != 1:
        raise ValueError("one 64 MiB GPU plan required")
    plan = json.loads(plans[0].read_text())
    plan["waterline"]["lookahead_groups"] = 8
    save(args.output/"plan.json", plan)
    summary = dict(scope="Same binary-manifest q_proj, GPU preparation, lookahead=8; only dispatcher wait statement differs",
        variants=variants, samples=[], statistics={})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB="6144", POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB="6144")
    for repeat in range(args.repeats):
        names = ("100us", "10us", "yield")
        names = names[repeat%3:]+names[:repeat%3]
        for name in names:
            folder = args.output/f"r{repeat}-{name}"
            folder.mkdir()
            binary = Path(variants[name]["binary"])
            if digest(binary) != variants[name]["binary_sha256"]:
                raise RuntimeError("private binary changed")
            command = [str(binary), str(args.output/"plan.json"), str(args.suite/"operator-spec.json"),
                str(args.suite/"fixture.json"), str(folder/"report.json"), str(folder/"decoded.json"),
                "--bundle", str(args.suite/"bundle"), "--backend", "gpu"]
            save(folder/"command.json", command)
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            started = time.monotonic()
            with (folder/"run.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            report = json.loads((folder/"report.json").read_text())
            if result.returncode or not report["passed"] or report["comparison"]["slots_checked"] != 8192:
                raise RuntimeError(f"private wait numerical failure: {folder}")
            summary["samples"].append(dict(variant=name, repeat=repeat, passed=True, report=report,
                process_wall_ms=1000*(time.monotonic()-started),
                process_cpu_seconds=after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime))
            for variant in variants:
                rows = [r for r in summary["samples"] if r["variant"]==variant]
                if rows:
                    summary["statistics"][variant] = {key:dict(median=statistics.median(numbers), min=min(numbers), max=max(numbers))
                        for key,numbers in dict(runtime_ms=[r["report"]["runtime_ms"] for r in rows],
                            online_ms=[r["report"]["runtime_timing"]["online_execution_ms"] for r in rows],
                            process_cpu_seconds=[r["process_cpu_seconds"] for r in rows]).items()}
            save(args.output/"summary.json", summary)
            print(f"{name} repeat={repeat} forward_ms={report['runtime_ms']}", flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command", required=True)
    g=sub.add_parser("generate")
    g.add_argument("--kind",choices=["mlp","pressure","qproj"],required=True)
    g.add_argument("--output",type=Path,required=True)
    g.add_argument("--devices",type=int,choices=[1,2],default=1)
    g.add_argument("--capacity-gib",type=int,default=7)
    g.add_argument("--reserve-gib",type=int,default=1)
    g.add_argument("--count",type=int,default=1024)
    g.add_argument("--budgets",type=int,nargs="+",default=[4096,256,32,8])
    g.add_argument("--backends",nargs="+",choices=["cpu","gpu","auto"],default=["cpu","gpu"])
    g.add_argument("--window",type=int,default=512)
    g.add_argument("--operator-spec",type=Path,default=ROOT/"scripts/qwen3_dsl/profiles/operator-spec.json")
    g.add_argument("--source-directory",type=Path,default=ROOT/"build-waterline-sync/qwen3-q50-rescale-candidate/q_proj")
    r=sub.add_parser("run")
    r.add_argument("--suite",type=Path,required=True)
    r.add_argument("--output",type=Path,required=True)
    r.add_argument("--build",type=Path,required=True)
    r.add_argument("--cuda",default="0")
    r.add_argument("--pool-mib",type=int,default=6144)
    r.add_argument("--repeats",type=int,default=3)
    r.add_argument("--timeout",type=int,default=600)
    r.add_argument("--cases",nargs="+")
    b=sub.add_parser("binary-copy")
    b.add_argument("--suite",type=Path,required=True)
    b.add_argument("--output",type=Path,required=True)
    w=sub.add_parser("poll-existing")
    w.add_argument("--suite",type=Path,required=True)
    w.add_argument("--variants",type=Path,required=True)
    w.add_argument("--output",type=Path,required=True)
    w.add_argument("--repeats",type=int,default=3)
    args=p.parse_args()
    args.output=args.output.resolve()
    if args.command=="run": args.build=args.build.resolve()
    if args.command=="poll-existing":
        args.suite=args.suite.resolve()
        args.variants=args.variants.resolve()
    if args.command=="generate": generate(args)
    elif args.command=="binary-copy": binary_copy(args)
    elif args.command=="poll-existing": poll_existing(args)
    else: run(args)


if __name__=="__main__":
    main()
