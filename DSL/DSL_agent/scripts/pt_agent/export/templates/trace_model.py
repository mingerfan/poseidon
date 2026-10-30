"""Export the package graph, or inspect its interface without a native import."""

import argparse
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bytecode", action="store_true")
    parser.add_argument("--mode", choices=("compiler", "managed"), default="compiler")
    parser.add_argument('--hecate-opt', type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT/"config/model.json").read_text())
    signature = json.loads((ROOT/"interfaces/default.json").read_text())
    approximations = json.loads((ROOT/"config/approximations.json").read_text())
    if args.plan:
        signature["approximations"] = {name: {key: value for key, value in item.items()
                                            if key != "coefficients"} for name, item in approximations.items()}
        print(json.dumps(signature, indent=2))
        return
    if args.weights is None or args.output_dir is None:
        parser.error("--weights and --output-dir are required unless --plan")
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing artifacts are preserved")
    if args.mode == 'managed':
        if args.hecate_opt is None:
            parser.error('--hecate-opt is required for managed frontend preprocessing')
        if args.bytecode:
            parser.error('managed mode emits Earth text')
        if config['refresh'] != 'compiler' or config['placement'] != 'depth-dp':
            parser.error('暂不支持：managed mode requires compiler/depth-dp')
        import inspect
        import hashlib
        import subprocess
        from types import SimpleNamespace
        runtime = importlib.import_module(config['name']+'.runtime')
        manager = importlib.import_module(config['name']+'.core.manage')
        weights = importlib.import_module(config['name']+'.weights').load(args.weights)
        management = json.loads((ROOT/'config/management.json').read_text())

        def owner():
            frame = inspect.currentframe().f_back
            try:
                while frame is not None:
                    if frame.f_code is runtime.model.forward.__code__:
                        data = management['source_lines'].get(str(frame.f_lineno))
                        return SimpleNamespace(**data) if data else None
                    frame = frame.f_back
            finally:
                del frame

        backend = manager.RecordingBackend(config['slots'], owner_resolver=owner)
        runtime.invoke(backend, weights)
        constants = {value.name:value.data for value in backend._constants.values()}
        args.output_dir.mkdir(parents=True)
        raw_metrics = {}
        raw = manager.emit_raw(backend,args.output_dir/'raw.earth.mlir',management['profile'],
                               name=config['name'],metrics=raw_metrics)
        from importlib import import_module
        streams = import_module(config['name']+'.core.manage.streams')
        substitutions = dict(hecate_opt=str(args.hecate_opt.resolve()), input=str(raw.resolve()),
                             package=str(ROOT), output_dir=str(args.output_dir.resolve()))
        command = [arg.format(**substitutions) for arg in
                   json.loads((ROOT/'config/frontend_preprocess_argv.json').read_text())]
        status = streams.run(command,cwd=args.output_dir)
        if status.returncode:
            raise subprocess.CalledProcessError(status.returncode,status.args)
        raw.unlink()
        command = [arg.format(**substitutions) for arg in
                   json.loads((ROOT/'config/frontend_structure_argv.json').read_text())]
        status = streams.run(command,cwd=args.output_dir)
        if status.returncode:
            raise subprocess.CalledProcessError(status.returncode,status.args)
        structure = streams.compress_ir(args.output_dir/'preprocessed.structure.mlir')
        shared = manager.load_graph(structure,config['slots'],constants=constants,profile=management['profile'])
        del backend
        function = manager.manage(shared, management['profile'], name=config['name'])
        manager.check(function)
        sizes = manager.constant_sizes(function)
        (args.output_dir/'constant_sizes.json').write_text(json.dumps(sizes,indent=2)+'\n')
        if sizes['estimated_cst_bytes']>8*1024**3:
            raise ValueError('Estimated .cst exceeds 8 GiB: '+str(sizes))
        emit_metrics = {}
        managed = manager.emit(function, args.output_dir/'managed.mlir', metrics=emit_metrics)
        summary = dict(function.summary(),input_dag='frontend-preprocessed')
        (args.output_dir/'management.json').write_text(json.dumps(summary, indent=2)+'\n')
        management.update(summary,mode=management['mode'])
        management_path = ROOT/'config/management.json'
        data = (json.dumps(management,indent=2)+'\n').encode()
        management_path.write_bytes(data)
        # This new package's generated descriptor is finalized at its first
        # managed trace. Keep the distribution manifest entry in sync.
        manifest_path = ROOT/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['config/management.json'] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        signature.update(function='_hecate_'+config['name'], mlir=managed.name,
                         native_export_format='earth-managed-text', native_tracing_completed=False,
                         management_completed=True,
                         management_emit=emit_metrics,
                         frontend_emit=raw_metrics, frontend_preprocessing_completed=True,
                         frontend_preprocess_format='bytecode-and-structure',constant_sizes=sizes,
                         weights_binding=dict(file=args.weights.name, bytes=args.weights.stat().st_size))
        (args.output_dir/'signature.json').write_text(json.dumps(signature, indent=2)+'\n')
        print(str(managed))
        return
    import hecate as hc

    runtime = importlib.import_module(config["name"]+".runtime")
    weights = importlib.import_module(config["name"]+".weights").load(args.weights)
    runtime.register(hc, weights)
    args.output_dir.mkdir(parents=True)
    if args.bytecode:
        print(hc.save(str(args.output_dir), str(args.output_dir), bytecode=True))
    else:
        print(hc.save(str(args.output_dir), str(args.output_dir)))
        streams = importlib.import_module(config['name']+'.core.manage.streams')
        text_path = streams.compress_ir(args.output_dir/'trace_model.mlir')
    signature.update(function="_hecate_"+config["name"], mlir="trace_model.mlirbc" if args.bytecode else "trace_model.mlir",
                     native_export_format="mlir-bytecode" if args.bytecode else "mlir-text",
                     native_tracing_completed=True,
                     weights_binding=dict(file=args.weights.name, bytes=args.weights.stat().st_size))
    if not args.bytecode and text_path.suffix=='.gz':
        signature['mlir'] = text_path.name
    with (args.output_dir/"signature.json").open("x") as stream:
        json.dump(signature, stream, indent=2)


if __name__ == "__main__":
    main()
