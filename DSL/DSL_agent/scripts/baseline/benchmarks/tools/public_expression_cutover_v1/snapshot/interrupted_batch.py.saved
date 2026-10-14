"""Seal an interrupted checkpoint without overwriting historical evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time


def select_remaining(catalog, prior):
    if prior.get('status') != 'interrupted' or not prior.get('interruption', {}).get('workers_stopped'):
        raise ValueError('Requires a sealed interrupted batch')
    rows = prior.get('cases', [])
    if [r.get('descriptor') for r in rows] != catalog:
        raise ValueError('Source batch must contain the exact full catalog in order')
    chosen = []
    for row in rows:
        status, metrics = row.get('status'), row.get('metrics')
        if status == 'pending':
            if metrics is not None:
                raise ValueError('Pending case has terminal metrics')
        elif status in ('passed', 'failed'):
            if type(metrics) is not dict or type(metrics.get('passed')) is not bool:
                raise ValueError('Invalid terminal metrics')
            if (status == 'passed') != metrics['passed']:
                raise ValueError('Inconsistent terminal status')
        else:
            raise ValueError('Unknown case status')
        if status != 'passed':
            chosen.append(row['descriptor'])
    if not chosen:
        raise ValueError('No remaining cases')
    return chosen


def load_remaining_source(catalog, source, results, seen=None):
    source, results = Path(source).resolve(), Path(results).resolve()
    seen = set() if seen is None else set(seen)
    if source in seen or len(seen) >= 16 or not source.is_relative_to(results) or source.stat().st_size > 4 * 1024**2:
        raise ValueError('Invalid interrupted report path or size')
    seen.add(source)
    prior = json.loads(source.read_text())
    selection = prior.get('selection')
    if selection:
        if selection.get('kind') != 'previous_failures_and_unfinished':
            raise ValueError('Unsupported interrupted selection lineage')
        parent = Path(selection['source_report']).resolve()
        _, catalog = load_remaining_source(catalog, parent, results, seen)
        if hashlib.sha256(parent.read_bytes()).hexdigest() != selection['source_sha256']:
            raise ValueError('Interrupted lineage hash changed')
    selected = select_remaining(catalog, prior)
    origin = Path(prior['interruption']['source_report']).resolve()
    if not origin.is_relative_to(results) or origin == source or origin.stat().st_size > 4 * 1024**2:
        raise ValueError('Invalid original report')
    if hashlib.sha256(origin.read_bytes()).hexdigest() != prior['interruption']['source_sha256']:
        raise ValueError('Original checkpoint hash changed')
    original = json.loads(origin.read_text())
    if original['cases'] != prior['cases']:
        raise ValueError('Interrupted snapshot changed case evidence')
    return prior, selected


def seal(source):
    from hecate_python_env import WORK
    from run_agent_batch import batch_plan, case_metrics
    source = source.resolve()
    if source.parent.parent != WORK / 'results' or source.name != 'report.json':
        raise ValueError('Expected a batch report in the results root')
    # Inspect command names/arguments only, never process environments.
    processes = subprocess.run(['ps', '-eo', 'args='], capture_output=True, text=True,
                               check=True, timeout=10).stdout
    if any(name in processes for name in ('run_agent_batch.py', 'run_candidate.py', 'deepseek_http_worker.py')):
        raise ValueError('Stop all live batch/candidate workers before sealing')
    raw = source.read_bytes()
    prior = json.loads(raw)
    if prior.get('status') not in ('running', 'paused_provider_failure'):
        raise ValueError('Expected an interrupted checkpoint')
    for row in prior['cases']:
        if row.get('status') in ('passed', 'failed'):
            evidence = Path(row['evidence']).resolve()
            if evidence.parent != WORK / 'results':
                raise ValueError('Invalid case evidence directory')
            if json.loads((evidence / 'model.json').read_text()) != row['descriptor']:
                raise ValueError('Case identity mismatch')
            if case_metrics(json.loads((evidence / 'report.json').read_text())) != row['metrics']:
                raise ValueError('Saved case metrics mismatch')
    frozen = copy.deepcopy(prior)
    frozen['status'] = 'interrupted'
    frozen['interruption'] = dict(workers_stopped=True, sealed_unix=time.time(),
        reason='user_authorized_remaining_case_restart', source_report=str(source),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        in_flight_cost_unknown=True)
    rows, _ = batch_plan(bool(prior.get('benchmark')))
    catalog = [r['descriptor'] for r in rows]
    if prior.get('selection'):
        selection = prior['selection']
        if selection.get('kind') != 'previous_failures_and_unfinished':
            raise ValueError('Unsupported interrupted selection lineage')
        parent = Path(selection['source_report'])
        _, catalog = load_remaining_source(catalog, parent, WORK / 'results')
        if hashlib.sha256(parent.read_bytes()).hexdigest() != selection['source_sha256']:
            raise ValueError('Source lineage changed')
    selected = select_remaining(catalog, frozen)
    destination = source.with_name('interrupted-report.json')
    with destination.open('x') as stream:
        json.dump(frozen, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(snapshot=str(destination), preserved_passed=len(catalog)-len(selected),
                          selected=len(selected), selected_ids=[r['id'] for r in selected]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    seal(parser.parse_args().seal)
