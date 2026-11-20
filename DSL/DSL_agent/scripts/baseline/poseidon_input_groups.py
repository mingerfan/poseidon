"""Bounded native input groups and their complete external execution records."""
import copy
import json
import math
from pathlib import Path

LIMIT = 16
FORMAT = 'poseidon-native-input-groups-v1'
PLAN_FILE = 'native-input-groups.json'
GROUP_FIELD = 'native_input_groups'
MAX_FILE_BYTES = 16 * 1024**2


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=MAX_FILE_BYTES,
            'Bounded regular native group record')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate native group record field');result[key]=value
        return result
    def invalid(value):raise ValueError('Nonfinite native group record')
    def number(value):
        result=float(value);require(math.isfinite(result),'Finite native group number');return result
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=invalid,parse_float=number)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False)


def samples(payload):
    rows=payload.get('samples') if type(payload)is dict else payload
    require(type(rows)is list and 1<=len(rows)<=1024,'Native group input count')
    return rows


def plan(payload):
    count=len(samples(payload))
    rows=[dict(index=i,start=start,stop=min(start+LIMIT,count),directory='native-groups/%03d'%i)
          for i,start in enumerate(range(0,count,LIMIT))]
    return dict(format=FORMAT,inputs=count,maximum_per_process=LIMIT,groups=rows)


def portion(payload, row):
    rows=copy.deepcopy(samples(payload)[row['start']:row['stop']])
    if type(payload)is list:return rows
    result=copy.deepcopy(payload);result['samples']=rows;return result


def argv(row):
    directory='/out/'+row['directory']
    return ['/poseidon-agent-cpu','/out/bound/plan.json','/operator-spec.json',
            '/out/bound/bundle',directory+'/inputs.json',directory+'/execution.json']


def merge(payload, reports):
    selected=plan(payload);require(len(selected['groups'])==len(reports),'Complete native group result count')
    outputs=[];metadata=None
    for row,report in zip(selected['groups'],reports):
        require(type(report)is dict and GROUP_FIELD not in report and type(report.get('outputs'))is list
                and len(report['outputs'])==row['stop']-row['start'],'Native group result scope')
        fields={k:v for k,v in report.items()if k!='outputs'}
        if metadata is None:metadata=fields
        else:require(fields==metadata,'Native groups used different execution metadata')
        outputs.extend(copy.deepcopy(report['outputs']))
    result=copy.deepcopy(metadata);result['outputs']=outputs
    result[GROUP_FIELD]=dict(format=FORMAT,manifest=PLAN_FILE,inputs=selected['inputs'],processes=len(reports))
    return result


def execute(trace_payload, output, payload, native_run, log, validate_report, check_precheck):
    """Invoke the same native client/program once per planned group, with no retry."""
    output=Path(output);selected=plan(payload);save(output/PLAN_FILE,selected);reports=[]
    with Path(log).open('x')as summary:
        summary.write(json.dumps(dict(format=FORMAT,inputs=selected['inputs'],processes=len(selected['groups'])))+'\n');summary.flush()
        for row in selected['groups']:
            directory=output/row['directory'];directory.mkdir(parents=True)
            save(directory/'inputs.json',portion(payload,row));command=argv(row)
            save(directory/'attempt.json',dict(index=row['index'],start=row['start'],stop=row['stop'],argv=command,native_invocations_consumed=1,maximum=1,retries=0))
            returned=None
            try:
                returned=native_run(trace_payload,output,command,directory/'execute.log',150)
                check_precheck()
                require(type(returned)is int,'Native group return code')
                save(directory/'terminal.json',dict(index=row['index'],status='returned',returncode=returned))
                require(returned==0,'Isolated native input group %d failed (exit %d)'%(row['index'],returned))
                report=read(directory/'execution.json');validate_report(report,row['stop']-row['start']);reports.append(report)
                summary.write(json.dumps(dict(group=row['index'],status='validated',inputs=row['stop']-row['start']))+'\n');summary.flush()
            except BaseException as error:
                if not (directory/'terminal.json').exists():
                    save(directory/'terminal.json',dict(index=row['index'],status='exception',returncode=returned,type=type(error).__name__))
                failure=dict(format=FORMAT,group=row['index'],planned_processes=len(selected['groups']),attempted_groups=row['index']+1,completed_groups=len(reports),type=type(error).__name__,message=str(error)[:1200],aggregate_created=False,retries=0)
                save(output/'native-input-group-failure.json',failure)
                summary.write(json.dumps(failure)+'\n');summary.flush();raise
    report=merge(payload,reports);save(output/'execution.json',report);return report


def audit(output, *, allow_test_only=False):
    """Match every group to the original inputs and retained native result files."""
    output=Path(output);payload=read(output/'inputs.json');aggregate=read(output/'execution.json')
    require(allow_test_only or aggregate.get('test_only')is not True,'Protocol controls are not native execution evidence')
    expected=plan(payload);manifest=output/PLAN_FILE
    if not manifest.exists():
        require(GROUP_FIELD not in aggregate and len(samples(payload))<=LIMIT,'Native group execution evidence missing')
        return None
    require(len(expected['groups'])>1 and read(manifest)==expected,'Complete original native input group plan')
    reports=[]
    for row in expected['groups']:
        directory=output/row['directory']
        require(directory.resolve()==directory and directory.is_dir(),'Native group evidence directory')
        require(read(directory/'inputs.json')==portion(payload,row),'Native group input order or values changed')
        attempt=read(directory/'attempt.json');terminal=read(directory/'terminal.json')
        require(attempt==dict(index=row['index'],start=row['start'],stop=row['stop'],argv=argv(row),native_invocations_consumed=1,maximum=1,retries=0),'Original compiled program and one native group call required')
        require(terminal==dict(index=row['index'],status='returned',returncode=0),'Successful native group terminal required')
        require((directory/'execute.log').is_file() and not (directory/'execute.log').is_symlink(),'Retained native group log required')
        report=read(directory/'execution.json')
        require(allow_test_only or report.get('test_only')is not True,'Controlled group is not native evidence')
        require(report.get('backend')=='poseidon_cpu' and report.get('encrypted_execution')is True
                and report.get('executor_secret_key')is False and report.get('security')=='tc128','Native group execution identity')
        reports.append(report)
    require(aggregate==merge(payload,reports),'Combined native outputs differ from the complete group records')
    return expected
