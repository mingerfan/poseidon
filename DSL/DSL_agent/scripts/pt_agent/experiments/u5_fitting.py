"""Finite U5 fitting-quality checks. The outer process owns results/registry."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time


def write(path, value):
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {k:finite(v) for k,v in item.items()}
        if isinstance(item, (tuple, list)):
            return [finite(v) for v in item]
        return item
    Path(path).write_text(json.dumps(finite(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def remez_quality(directory, resume=None):
    import numpy as np
    from ..kernels.approx import ApproximationError, function, remez, remez_report, tolerance
    from ..kernels.poly import cheb_eval, cheb_fit
    report = dict(round='U5',part='remez-quality',complete=False,rows=[],llm_calls=0,report_images=0)
    previous = (json.loads(Path(resume).read_text())['rows'] if resume else [])
    reusable = {(r['fn'],tuple(r['domain']),r['degree']):r for r in previous if r.get('passed')}
    for fn in ('silu', 'gelu', 'sigmoid', 'tanh', 'exp'):
        for domain in ((-8.0, 8.0), (-16.0, 16.0)):
            for degree in (7, 15, 31, 63):
                row = dict(fn=fn,domain=domain,degree=degree)
                report['rows'].append(row)
                started = time.monotonic()
                try:
                    reused = reusable.get((fn,domain,degree))
                    if reused is None:
                        coefficients = remez(fn, degree, domain)
                        exchange = remez_report(fn, degree, domain)
                    else:
                        coefficients = np.asarray(reused['coefficients'])
                        exchange = reused['exchange']
                        row['reused_from'] = str(resume)
                    normalized = np.linspace(-1.0, 1.0, 2**16)
                    x = ((domain[1]-domain[0])*normalized+sum(domain))/2
                    exact = function(fn, x)
                    allowed, tau, criterion = tolerance(exchange['E'],float(np.abs(exact).max()))
                    error = float(np.max(np.abs(cheb_eval(None, normalized, coefficients)-exact)))
                    cheb_error = float(np.max(np.abs(cheb_eval(None, normalized,
                        cheb_fit(fn, degree, *domain))-exact)))
                    extrema = np.asarray(exchange['errors'])
                    spread = (float((np.abs(extrema).max()-np.abs(extrema).min())/
                                    np.abs(extrema).max()) if np.abs(extrema).max() else 0.0)
                    alternating = bool(np.all(extrema[:-1]*extrema[1:] < 0))
                    row.update(coefficients=coefficients.tolist(),max_error=error,cheb_max_error=cheb_error,
                        exchange=exchange,alternating=alternating,extrema=len(extrema),relative_spread=spread,
                        E=exchange['E'],tau=tau,gap=exchange['max_error']-abs(exchange['E']),criterion=criterion,
                        seconds=time.monotonic()-started)
                    if (error > cheb_error+tau or len(extrema) < degree+2 or not alternating
                            or np.any(np.abs(np.abs(extrema)-abs(exchange['E'])) > allowed)):
                        row['passed'] = False
                        report['stop'] = 'Fitting quality differs from U5 section 3.2'
                        write(directory/'REPORT.json',report)
                        return report
                    row['passed'] = True
                except ApproximationError as exc:
                    row.update(passed=False,error=str(exc),iterations=exc.iterations,
                        max_error=exc.max_error,E=exc.error_amplitude,history=exc.history,
                        seconds=time.monotonic()-started)
                    report['stop'] = 'Remez did not converge; no interpolation fallback'
                    write(directory/'REPORT.json',report)
                    return report
                write(directory/'REPORT.json',report)
                print(f'U5 {fn} {domain} degree={degree} remez={error} cheb={cheb_error} spread={spread}',flush=True)
    report['complete'] = True
    write(directory/'REPORT.json',report)
    return report


def composite_quality(directory):
    import numpy as np
    from ..kernels.approx import (ApproximationError,CompositeSearchError,composite_coeffs,composite_report,
                                  search_composite,tolerance)
    from ..kernels.layout import KernelError
    from ..kernels.poly import cheb_eval,minimax_coeffs
    eps = 2.0**-10
    degrees = (15,15,27)
    report = dict(round='U5',part='composite-quality',complete=False,rows=[],llm_calls=0,report_images=0)
    row = dict(kind='anchor',degrees=degrees,eps=eps,grid_points=2**16)
    report['rows'].append(row)
    try:
        grid = np.linspace(eps,1.0,2**16)
        baseline = grid
        for c in minimax_coeffs(degrees):
            baseline = cheb_eval(None,baseline,c)
        row['dacapo_error'] = float(np.max(np.abs(baseline-0.5)))
        coefficients = composite_coeffs(degrees,eps)
        sign = grid
        for c in coefficients:
            sign = cheb_eval(None,sign,c)
        error = float(np.max(np.abs(sign-0.5)))
        _,tau,_ = tolerance(0.0,1.0)
        prior_error = 0.0030955092661315575
        row.update(error=error,tau=tau,stages=composite_report(degrees,eps),
                   coefficients=[c.tolist() for c in coefficients],prior_error=prior_error,
                   unchanged_or_better=bool(error <= prior_error+tau),
                   passed=bool(error <= row['dacapo_error']+tau and error <= prior_error+tau))
        write(directory/'REPORT.json',report)
        if not row['passed']:
            report['stop'] = 'Composite anchor differs from U5 section 1.2 / 7'
            write(directory/'REPORT.json',report)
            return report
        for target in (0.05,0.01,0.001):
            row = dict(kind='search',target_error=target,eps=0.2,max_depth=12)
            report['rows'].append(row)
            found = search_composite(12,0.2,target)
            row.update(found,passed=True)
            write(directory/'REPORT.json',report)
    except KernelError as exc:
        report['stop'] = str(exc)
        row.update(passed=False,error=str(exc))
        if isinstance(exc,ApproximationError):
            row.update(max_error=exc.max_error,E=exc.error_amplitude,iterations=exc.iterations,history=exc.history)
        if isinstance(exc,CompositeSearchError):
            row.update(excluded=len(exc.exclusions),exclusions=exc.exclusions,attempted=exc.attempted)
        write(directory/'REPORT.json',report)
        return report
    report['complete'] = True
    write(directory/'REPORT.json',report)
    return report


def newton_quality(directory):
    import numpy as np
    from ..backends import CountBackend
    from ..kernels.approx import act_remez, function, newton, remez_report
    report = dict(round='U5',part='newton-quality',complete=False,rows=[],llm_calls=0,report_images=0)
    for fn,domains in (('inv',((0.01,1.0),(0.1,10.0))),
                       ('rsqrt',((0.01,1.0),(0.5,2.0)))):
        for domain in domains:
            x = np.linspace(*domain,2**16)
            exact = function(fn,x)
            for degree in (3,7,15):
                initial_backend = CountBackend(8)
                initial = act_remez(initial_backend,initial_backend.input(0),fn,degree,domain)
                for iterations in (0,1,2,3):
                    backend = CountBackend(8)
                    result = newton(backend,backend.input(0),fn,degree,iterations,domain)
                    values = newton(None,x,fn,degree,iterations,domain)
                    predicted = initial.depth+(2 if fn=='inv' else 3)*iterations
                    row = dict(fn=fn,domain=domain,deg0=degree,iters=iterations,
                        initial_depth=initial.depth,depth=result.depth,predicted_depth=predicted,
                        max_relative_error=float(np.max(np.abs(values-exact)/np.abs(exact))),
                        initial_fit=remez_report(fn,degree,domain),
                        passed=bool(result.depth==predicted and np.isfinite(values).all()))
                    report['rows'].append(row)
                    write(directory/'REPORT.json',report)
                    if not row['passed']:
                        report['stop']='Newton depth or finite arithmetic differs'
                        write(directory/'REPORT.json',report)
                        return report
    report['complete']=True
    write(directory/'REPORT.json',report)
    return report


def relu_quality(directory):
    import numpy as np
    from ..kernels.approx import ApproximationError,remez,remez_report
    from ..kernels.poly import cheb_eval
    report=dict(round='U5',part='relu-quality',complete=False,rows=[],llm_calls=0,report_images=0)
    for degree in (7,15,31,63):
        row=dict(fn='relu',domain=(-8.0,8.0),degree=degree)
        try:
            c=remez('relu',degree,(-8.0,8.0))
            x=np.linspace(-8.0,8.0,2**16)
            row.update(converged=True,max_error=float(np.max(np.abs(cheb_eval(None,x/8,c)-np.maximum(x,0)))),
                       coefficients=c.tolist(),exchange=remez_report('relu',degree,(-8.0,8.0)))
        except ApproximationError as exc:
            row.update(converged=False,max_error=exc.max_error,E=exc.error_amplitude,
                       iterations=exc.iterations,history=exc.history,error=str(exc))
        report['rows'].append(row)
        write(directory/'REPORT.json',report)
    report['complete']=True
    write(directory/'REPORT.json',report)
    return report


def coefficient_inventory(directory):
    import numpy as np
    from ..kernels.approx import ApproximationError,remez,composite_coeffs
    rows=[]
    for fn in ('silu','gelu','sigmoid','tanh','exp'):
        for domain in ((-8.0,8.0),(-16.0,16.0)):
            for degree in (7,15,31,63):
                c=remez(fn,degree,domain)
                rows.append(dict(fn=fn,domain=domain,degree=degree,bits=c.view(np.uint64).tolist()))
    for fn,domains in (('inv',((0.01,1.0),(0.1,10.0))),('rsqrt',((0.01,1.0),(0.5,2.0)))):
        for domain in domains:
            for degree in (3,7,15):
                c=remez(fn,degree,domain)
                rows.append(dict(fn=fn,domain=domain,degree=degree,bits=c.view(np.uint64).tolist()))
    for degree in (7,15,31,63):
        try:
            c=remez('relu',degree,(-8.0,8.0))
            rows.append(dict(fn='relu',degree=degree,bits=c.view(np.uint64).tolist()))
        except ApproximationError as exc:
            rows.append(dict(fn='relu',degree=degree,converged=False,error=str(exc)))
    for degrees,eps in (((15,15,27),2.0**-10),((9,),0.2),((15,),0.2),((27,),0.2)):
        rows.append(dict(fn='composite',degrees=degrees,eps=eps,
                         bits=[c.view(np.uint64).tolist() for c in composite_coeffs(degrees,eps)]))
    result=dict(complete=True,part='coefficients',pid=__import__('os').getpid(),rows=rows)
    write(directory/'REPORT.json',result)
    return result


def determinism(directory):
    results=[]
    for index in range(2):
        child=directory/('process-'+str(index+1))
        child.mkdir()
        command=[sys.executable,'-B','-m','pt_agent.experiments.u5_fitting','--inside',
                 '--part','coefficients','--out',str(child)]
        with (child/'run.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=240)
        results.append(json.loads((child/'REPORT.json').read_text()))
    if results[0]['pid']==results[1]['pid'] or results[0]['rows']!=results[1]['rows']:
        raise AssertionError('Independent-process coefficient bit patterns differ')
    result=dict(round='U5',part='determinism',complete=True,processes=[r['pid'] for r in results],
                parameter_groups=len(results[0]['rows']),bitwise_equal=True,llm_calls=0,report_images=0)
    write(directory/'REPORT.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--part',choices=('unit','remez','composite','newton','relu','determinism','coefficients'),default='remez')
    parser.add_argument('--resume',type=Path)
    args = parser.parse_args()
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        if args.out is None:
            parser.error('--inside requires --out')
        if args.part == 'unit':
            import unittest
            files=sorted((Path(__file__).resolve().parents[1]/'tests').glob('test_u5_*.py'))
            suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem)
                                     for p in files)
            result = unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
            report = dict(round='U5',part='unit',complete=result.wasSuccessful(),tests=result.testsRun,
                          failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))
            write(args.out/'REPORT.json',report)
            return 0 if report['complete'] else 1
        if args.part=='remez':
            result=remez_quality(args.out,args.resume)
        else:
            try:
                result=dict(composite=composite_quality,newton=newton_quality,relu=relu_quality,
                            determinism=determinism,coefficients=coefficient_inventory)[args.part](args.out)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                path=args.out/'REPORT.json'
                result=json.loads(path.read_text()) if path.exists() else dict(rows=[])
                result.update(complete=False,error=f'{type(exc).__name__}: {exc}')
                write(path,result)
        return 0 if result['complete'] else 1
    from ..paths import RESULTS_DIR
    now = datetime.now(timezone.utc)
    run_id = 'u5-fitting-'+now.strftime('%Y%m%dT%H%M%S%f')
    directory = args.out or RESULTS_DIR/run_id
    directory.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    command = [sys.executable,'-m','pt_agent.tools.nixrun','--seconds','600','--',
        '-m','pt_agent.experiments.u5_fitting','--inside','--part',args.part,'--out',str(directory)]
    if args.resume is not None:
        command += ['--resume',str(args.resume)]
    process = subprocess.run(command,
        capture_output=True,text=True)
    (directory/'run.log').write_text(process.stdout+process.stderr)
    path = directory/'REPORT.json'
    report = json.loads(path.read_text()) if path.exists() else dict(
        complete=False,rows=[],error='Inner process failed before writing REPORT.json; see run.log')
    rev = subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
    report.update(run_id=run_id,git_rev=rev,exit=process.returncode,seconds=time.monotonic()-started,
        llm_calls=0,report_images=0,compiler_calls=0)
    write(path,report)
    passed = (report['tests']-report['failures']-report['errors']-report['skipped']
              if 'tests' in report else sum(r.get('passed',False) for r in report.get('rows',[])))
    with (RESULTS_DIR/'registry.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(run_id=run_id,round='U5',model='analytic-functions',mode='fitting-'+args.part,
            git_rev=rev,metrics=dict(complete=report['complete'],exit=process.returncode,
                rows=len(report.get('rows',[])),tests=report.get('tests'),
                passed=passed,llm_calls=0),
            artifacts_dir=str(directory),ts=now.isoformat()))+'\n')
    print(json.dumps(dict(report=str(path),exit=process.returncode,complete=report['complete'],
                         rows=len(report.get('rows',[])),tests=report.get('tests')),indent=2))
    return process.returncode


if __name__ == '__main__':
    sys.exit(main())
