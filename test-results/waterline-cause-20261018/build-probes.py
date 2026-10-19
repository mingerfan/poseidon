#!/usr/bin/env python3
"""Build frozen scheduler controls, keeping production sources unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--libraries', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--headers', type=Path, required=True)
args = parser.parse_args()
runtime, libraries, out = args.runtime.resolve(), args.libraries.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
current = (args.headers / 'current.hpp').read_text()
legacy = (args.headers / 'legacy.hpp').read_text()
shortcut = '''
        if (!waterline_) {
            if (!coordinator_groups_.empty()) throw std::runtime_error("single-rank probe only");
            std::vector<std::vector<std::size_t>> direct(queues.size());
            for(std::size_t i=0;i<tasks.size();++i)direct[tasks[i].worker].push_back(i);
            std::vector<std::thread> direct_workers;
            for(std::size_t w=0;w<direct.size();++w)if(!direct[w].empty())
                direct_workers.emplace_back([&,w] {
                    try {for(auto i:direct[w]) {
                        if(parallel_failed_.load())return;
                        if(tasks[i].task.preparation)throw std::runtime_error("eager probe only");
                        tasks[i].task.execute();
                    }} catch(...) {record_parallel_failure(std::current_exception());}
                });
            for(auto &worker:direct_workers)worker.join();
            if(parallel_failed_.load())std::rethrow_exception(parallel_failure_);
            for(const auto &group:parallel_groups_)if(!group->completed)finish_parallel_group(group,std::nullopt);
            parallel_groups_.clear();coordinator_groups_.clear();
            return;
        }
'''
marker = '        if(baseline_only)return;'
assert current.count(marker) == 1
headers = dict(legacy=legacy, current=current,
    prepared_direct=current.replace(marker, marker + shortcut))
records = []
for variant, header in headers.items():
    folder = out / variant
    overlay = folder / 'include/runtime/runtime.hpp'
    overlay.parent.mkdir(parents=True)
    overlay.write_text(header)
    command = ['c++', '-O3', '-DNDEBUG', '-std=c++20', '-pthread', '-fopenmp',
        '-I'+str(folder/'include'), '-I'+str(runtime), '-I'+str(runtime/'third_party'),
        str(Path(__file__).resolve().parent/'cpu-probe.cpp'),
        str(libraries/'libruntime_plan_json.a'), str(libraries/'libruntime_core.a'),
        '-o', str(folder/'probe')]
    records.append(dict(variant=variant, command=command,
        header_sha256=hashlib.sha256(header.encode()).hexdigest()))
    (out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
    with (folder/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Built '+variant, flush=True)
