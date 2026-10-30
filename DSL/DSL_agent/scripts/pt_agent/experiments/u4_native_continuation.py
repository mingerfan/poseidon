"""U4 §7.5: retain the BERT package and run managed before compiler bytecode.

The package entry points and management algorithms are unchanged. Function
wrappers below record stage timing, sampled RSS and IR sizes only.
"""

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import functools
import importlib
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import subprocess
import sys
import threading
import time

from ..paths import REPO, RESULTS_DIR
from .u4_point_diagnosis import write


class Stages:
    def __init__(self, path, directory, streams):
        self.path, self.directory, self.streams = path, directory, streams
        self.rows, self.active = [], []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def memory(self, pid):
        try:
            fields = Path(f'/proc/{pid}/statm').read_text().split()
            own = int(fields[1])*os.sysconf('SC_PAGE_SIZE')/1024**2
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except (OSError, IndexError):
            return 0.0, 0.0
        values = [self.memory(int(child)) for child in children]
        return max([own, *(v[0] for v in values)]), own+sum(v[1] for v in values)

    def sample(self):
        while not self.done.is_set():
            process, tree = self.memory(os.getpid())
            for row in tuple(self.active):
                row['sampled_peak_process_rss_mib'] = max(row['sampled_peak_process_rss_mib'], process)
                row['sampled_peak_tree_rss_mib'] = max(row['sampled_peak_tree_rss_mib'], tree)
            self.done.wait(0.2)

    def ir_files(self):
        result = {}
        if not self.directory.exists():
            return result
        for path in self.directory.iterdir():
            if not path.is_file() or '.mlir' not in path.name:
                continue
            item = dict(bytes=path.stat().st_size)
            if path.suffix == '.gz':
                item['uncompressed_bytes'] = self.streams.uncompressed_size(path)
            elif path.suffix == '.mlir':
                item['uncompressed_bytes'] = item['bytes']
            result[path.name] = item
        return result

    @contextmanager
    def stage(self, label):
        started = time.monotonic()
        process, tree = self.memory(os.getpid())
        row = dict(stage=label, complete=False, sampled_peak_process_rss_mib=process,
                   sampled_peak_tree_rss_mib=tree, sampling_seconds=0.2)
        self.rows.append(row)
        self.active.append(row)
        write(self.path, dict(stages=self.rows))
        print('STAGE '+label, flush=True)
        try:
            yield row
            row['complete'] = True
        except BaseException as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            row['seconds'] = time.monotonic()-started
            self.active.remove(row)
            row['ir_files'] = self.ir_files()
            write(self.path, dict(stages=self.rows))

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.stage(label):
                return original(*args, **kwargs)

        setattr(obj, name, measured)


def entry(args):
    """Run an unchanged package script, recording its existing phase boundaries."""
    sys.path.insert(0, str(args.package))
    config = json.loads((args.package/'config/model.json').read_text())
    prefix = config['name']+'.core.manage'
    streams = importlib.import_module(prefix+'.streams')
    stages = Stages(args.out/(args.entry+'-stages.json'), args.directory, streams)
    stages.thread.start()
    timing = ExitStack()
    try:
        if args.entry == 'trace' and args.mode == 'managed':
            runtime = importlib.import_module(config['name']+'.runtime')
            manager = importlib.import_module(prefix)
            stages.wrap(runtime, 'invoke', 'Python recording')
            for name, label in (('emit_raw', 'raw IR emission'), ('load_graph', 'preprocessed IR read'),
                                ('manage', 'Python management'), ('check', 'managed checks'),
                                ('emit', 'managed IR emission')):
                stages.wrap(manager, name, label)
            stages.wrap(streams, 'run', 'frontend preprocess')
            greedy = importlib.import_module(prefix+'.greedy')
            timing.enter_context(greedy.timing_log(observer=stages.stage))
        elif args.entry == 'compile' and args.mode == 'managed':
            stages.wrap(streams, 'run', 'compiler tail')
        argv = [str(args.package/(args.entry+'_model.py'))]
        if args.entry == 'trace':
            argv += ['--weights', str(args.weights), '--output-dir', str(args.directory)]
            if args.mode == 'compiler':
                argv.append('--bytecode')
            else:
                argv += ['--mode', 'managed', '--hecate-opt', str(args.hecate_opt)]
        else:
            argv += ['--hecate-opt', str(args.hecate_opt), '--input', str(args.input),
                     '--output-dir', str(args.directory), '--mode', args.mode]
        sys.argv = argv
        with stages.stage(args.entry+'/'+args.mode):
            try:
                runpy.run_path(argv[0], run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
    finally:
        timing.close()
        stages.done.set()
        stages.thread.join()


def server_jobs(args, result):
    from ..run import _server_job
    jobs = {}
    for mode in ('managed', 'compiler'):
        package = args.out/mode/args.source_package.name
        shutil.copytree(args.source_package, package)
        weights = args.out/mode/args.source_weights.name
        shutil.copy2(args.source_weights, weights)
        job = _server_job(args.out/mode, package, weights, mode=mode)
        if mode == 'compiler':
            text = job.read_text()
            old = ' --output-dir trace\n'
            if text.count(old) != 1 or text.count('--input trace/trace_model.mlir ') != 1:
                raise ValueError('Unexpected server command shape')
            job.write_text(text.replace(old, ' --output-dir trace --bytecode\n').replace(
                '--input trace/trace_model.mlir ', '--input trace/trace_model.mlirbc '))
        subprocess.run(['bash', '-n', str(job)], check=True)
        jobs[mode] = dict(path=str(job), syntax_exit=0, executed=False, bytecode=mode=='compiler')
    result['server_jobs'] = jobs
    write(args.out/'SERVER_JOBS.json', dict(order=['managed', 'compiler'], limits=result['limits'], jobs=jobs))


def inside(args):
    from .. import toolchain
    from .s0_pipeline_spike import compare_plans
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    previous = json.loads((args.source_run/'REPORT.json').read_text())
    original = previous['modes']['compiler']['export']
    args.source_package, args.source_weights = Path(original['package']), Path(original['weights'])
    result = dict(round='U4',section='7.5',complete=False,program=previous['program'],
                  evaluation=previous['evaluation'],source_run=str(args.source_run),cost=previous['cost'],
                  limits=dict(toolchain.TRUSTED_LIMITS),modes={},llm_calls=0,report_images=0,ckks_calls=0)
    try:
        server_jobs(args, result)
        write(args.out/'REPORT.json', result)
        toolchain.setup_env()
        from hecate_python_env import VENV
        from ..manage import streams
        name = json.loads((args.source_package/'config/model.json').read_text())['name']
        plan_name = f'lowered._hecate_{name}.runtime-plan.json'
        for mode in ('managed', 'compiler'):
            directory = args.out/mode
            package = directory/args.source_package.name
            trace_dir, compiled_dir = directory/'trace', directory/'compiled'
            record = result['modes'][mode] = dict(bytecode=mode=='compiler')
            write(args.out/'REPORT.json', result)
            base = [str(VENV/'bin/python'), '-B', '-m', 'pt_agent.experiments.u4_native_continuation',
                    '--package', str(package), '--weights', str(directory/args.source_weights.name),
                    '--mode', mode, '--out', str(directory), '--hecate-opt', str(toolchain.host_paths()['hecate_opt'])]
            trace_env = toolchain.package_trace_env()
            trace_env['PYTHONPATH'] = str(REPO/'DSL/DSL_agent/scripts')+os.pathsep+trace_env['PYTHONPATH']
            record['trace'] = toolchain.trusted_run(directory, base+['--entry', 'trace', '--directory', str(trace_dir)],
                directory/'trace.log', env_extra=trace_env)
            for kind in ('trace',):
                path = directory/(kind+'-stages.json')
                if path.exists():record[kind+'_stages'] = json.loads(path.read_text())['stages']
            write(args.out/'REPORT.json', result)
            if record['trace']['exit']:
                raise RuntimeError(mode+': package trace failed; see trace stages')
            mlir = streams.resolve(trace_dir/('managed.mlir' if mode=='managed' else 'trace_model.mlirbc'))
            record['ir'] = dict(path=str(mlir),bytes=mlir.stat().st_size)
            record['compile'] = toolchain.trusted_run(directory, base+['--entry', 'compile', '--directory', str(compiled_dir),
                '--input', str(mlir)], directory/'compile.log',
                env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH'],
                               PYTHONPATH=trace_env['PYTHONPATH']))
            phase_path = directory/'compile-stages.json'
            if phase_path.exists():record['compile_stages'] = json.loads(phase_path.read_text())['stages']
            write(args.out/'REPORT.json', result)
            if record['compile']['exit']:
                raise RuntimeError(mode+': compilation failed; see compile stages')
            record['plan'] = toolchain.plan_stats(compiled_dir, name)
            record['boots_equal'] = record['plan']['boots'] == result['cost']['boots_pred']
            write(args.out/'REPORT.json', result)
            if not record['boots_equal']:
                raise RuntimeError(mode+': predicted and compiled bootstrap counts differ')
            record['pruned_trace'] = toolchain.prune(trace_dir)
            record['pruned_compiled'] = toolchain.prune(compiled_dir,keep=(f'lowered._hecate_{name}.bundle',))
        result['runtime_plan'] = compare_plans(args.out/'compiler/compiled'/plan_name,
                                               args.out/'managed/compiled'/plan_name)
        if not result['runtime_plan']['equal']:
            raise RuntimeError('Compiler/managed RuntimePlan differs')
        result['complete'] = True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result['error'] = f'{type(exc).__name__}: {exc}'
        write(args.out/'STOP.json', result)
    finally:
        result['free_before_prune_bytes'] = shutil.disk_usage(args.out).free
        result['pruned_after_run'] = {str(folder.relative_to(args.out)): toolchain.prune(folder)
            for mode in ('managed', 'compiler') for part in ('trace', 'compiled')
            if (folder := args.out/mode/part).is_dir()}
        result['free_after_prune_bytes'] = shutil.disk_usage(args.out).free
        write(args.out/'REPORT.json', result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--entry', choices=('trace','compile'))
    parser.add_argument('--mode', choices=('managed','compiler'))
    for field in ('source-run','out','package','directory','weights','input','hecate-opt'):
        parser.add_argument('--'+field, type=Path)
    args = parser.parse_args()
    if args.entry:
        entry(args)
        return 0
    if args.inside:
        return inside(args)
    stamp = datetime.now(timezone.utc)
    run_id = 'u4-bert-native75-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out = args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    command = [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--',
               '-m','pt_agent.experiments.u4_native_continuation','--inside','--source-run',str(args.source_run),
               '--out',str(args.out)]
    write(args.out/'argv.json', command)
    print('RUN='+str(args.out),flush=True)
    started = time.monotonic()
    with (args.out/'run.log').open('w') as log:
        status = subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT)
    path = args.out/'REPORT.json'
    result = json.loads(path.read_text()) if path.exists() else dict(complete=False,error='see run.log')
    rev = subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=status.returncode,seconds=time.monotonic()-started)
    write(path,result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model='bert',mode='native75',git_rev=rev,
                metrics=dict(complete=result['complete'],llm_calls=0),artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(path),complete=result['complete'],exit=status.returncode),indent=2),flush=True)
    return status.returncode


if __name__ == '__main__':
    raise SystemExit(main())
