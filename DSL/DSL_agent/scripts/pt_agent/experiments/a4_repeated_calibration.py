"""Independent repeated public native cells and in-memory timing client.

The original39 metadata identities are retained. Five copies of the complete
public plan plus five extra bootstrap copies give five observations per
ordinary cell and ten bootstraps, in one context with one final comparison.
"""
from array import array
from collections import Counter
import copy
import json
import math
from pathlib import Path
import statistics
import sys

from . import a4_prepared_disk_client as disk
from .a4_prepared_disk_run import binding
from .r10_paper_metrics import write_json
from .r31_lattigo_profile import make_profile_plan, profile_cells
from .r31_lattigo_profile39 import build_profile, make_catalog_source
from .r31_lattigo_profile_reference import compare_references, direct_profile_sum, replay_profile
from .r31_lattigo_profile_subset import deterministic_fixture
from .r31_lattigo_runtime_plan import content_filename, plan_inventory


def cell_key(cell):
    return json.dumps({k:cell[k] for k in ('op','inputs','output','attrs')},
                      sort_keys=True,separators=(',',':'))


def repeat_profile(base, manifest, metadata, *, ordinary=5, bootstrap=10):
    if (type(ordinary) is not int or type(bootstrap) is not int
            or ordinary<2 or bootstrap<ordinary):
        raise ValueError('At least two ordinary observations and no fewer bootstrap observations required')
    cells=copy.deepcopy(metadata['cells'])
    boots=[row for row in cells if row['op']=='boot']
    if len(boots)!=1:
        raise ValueError('Require the one explicit native bootstrap metadata cell')
    micro,micro_manifest,micro_metadata=make_profile_plan(
        make_catalog_source(boots,c2s_bits=52),binding=binding())
    catalog={cell_key(row):row['index'] for row in cells}
    plan=copy.deepcopy(base)
    common=base['external_inputs'][0]
    plan.update(values=[copy.deepcopy(next(v for v in base['values'] if v['id']==common))],
                initialization=[],execution=[],finalization=[],final_outputs=[])
    targets=[];accumulator=None;blobs={};constants={}
    for iteration in range(bootstrap):
        source,source_manifest,source_metadata=(base,manifest,metadata) if iteration<ordinary else (micro,micro_manifest,micro_metadata)
        prefix='repeat-'+str(iteration)+'-'
        mapping={v['id']:(common if v['id']==source['external_inputs'][0] else prefix+v['id'])
                 for v in source['values']}
        for value in source['values']:
            if value['id']==source['external_inputs'][0]:continue
            value=copy.deepcopy(value);value['id']=mapping[value['id']];plan['values'].append(value)
        for item in source['initialization']:
            item=copy.deepcopy(item);item['output']=mapping[item['output']]
            item['ordinal']=len(plan['initialization']);plan['initialization'].append(item)
        by_ordinal={row['target_ordinal']:row for row in source_metadata['target_operations']}
        source_cells={row['index']:row for row in source_metadata['cells']}
        for item in source['execution']:
            previous=item['ordinal'];item=copy.deepcopy(item)
            item['ordinal']=len(plan['execution'])+100
            item['inputs']=[mapping[name] for name in item['inputs']]
            item['output']=mapping[item['output']];plan['execution'].append(item)
            if previous in by_ordinal:
                target=by_ordinal[previous];index=catalog[cell_key(source_cells[target['cell_index']])]
                targets.append(dict(cell_index=index,iteration=iteration,target_ordinal=item['ordinal'],
                    target_output=item['output'],source_count=target['source_count']))
        for row in source_manifest['blobs']:
            if row['content'] in blobs and blobs[row['content']]!=row:raise ValueError('Public constant locator collision')
            blobs[row['content']]=copy.deepcopy(row)
        for row in source_metadata['constant_vectors']:
            if row['content'] in constants and constants[row['content']]!=row:raise ValueError('Public constant value collision')
            constants[row['content']]=copy.deepcopy(row)
        output=mapping[source['final_outputs'][0]]
        if accumulator is None:accumulator=output
        else:
            name='repeat-aggregate-'+str(iteration)
            desc=copy.deepcopy(next(v for v in source['values'] if v['id']==source['final_outputs'][0]))
            desc['id']=name;plan['values'].append(desc)
            plan['execution'].append(dict(kind='compute',ordinal=len(plan['execution'])+100,
                op='add_cc',inputs=[accumulator,output],output=name,place=copy.deepcopy(desc['place'])))
            accumulator=name
    plan['final_outputs']=[accumulator]
    bundle_id='a4-repeated39-c2s52-p61-public-fixture'
    plan['plaintext_bundle']['id']=bundle_id
    result_manifest=dict(manifest,bundle_id=bundle_id,blobs=list(blobs.values()))
    result_metadata=dict(metadata,cells=cells,target_operations=targets,
        constant_vectors=list(constants.values()),ordinary_repetitions=ordinary,
        bootstrap_repetitions=bootstrap,target_execution_count=len(targets),
        setup_and_aggregation_count=len(plan['execution'])-len(targets),
        old_target_durations_reused=False,model_forwards=0,model_quality_measurement=False,
        observation_order='All39 cells in canonical order five times, then five extra bootstrap microplans',
        measurement_scope='Operation span includes prepared plaintext lookup/read and native compute with bounded-cache prepared key loads; excludes key generation, constant encoding, setup operations and aggregation operations')
    counts=Counter(row['cell_index'] for row in targets)
    if any(counts[row['index']]!=(bootstrap if row['op']=='boot' else ordinary) for row in cells):
        raise ValueError('Repeated target counts differ')
    return plan,result_manifest,result_metadata


def prepare_public(out, base_metadata, missing_metadata):
    """Create only public arithmetic fixtures; never a model or crypto object."""
    base=json.loads(Path(base_metadata).read_text());missing=json.loads(Path(missing_metadata).read_text())
    if base.get('cell_count')==39:
        excluded={cell_key(row)for row in missing['cells']}
        base=dict(base,cell_count=36,cells=[row for row in base['cells']if cell_key(row)not in excluded])
    plan,manifest,metadata,_=build_profile(base,missing,c2s_bits=52)
    plan,manifest,metadata=repeat_profile(plan,manifest,metadata)
    inventory=plan_inventory(plan,manifest,binding=binding())
    packed=deterministic_fixture()
    constants={row['content']:array('d',[row['value']])*len(packed) for row in metadata['constant_vectors']}
    replay,summary=replay_profile(plan,constants,packed)
    direct=direct_profile_sum(metadata,packed,precision_bits=50)
    comparison=compare_references(direct,replay)
    if comparison['passed'] is not True:raise ValueError('Repeated public plan differs from independent cell formulas')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    bundle=out/'profile.bundle';(bundle/'data').mkdir(parents=True)
    def vector(path,values):
        raw=array('d',values)
        if sys.byteorder!='little':raw.byteswap()
        path.write_bytes(raw.tobytes())
    for content,values in constants.items():vector(bundle/'data'/content_filename(content),values)
    vector(out/'input.bin',packed)
    write_json(out/'plan.json',plan);write_json(bundle/'manifest.json',manifest)
    write_json(out/'metadata.json',metadata);write_json(out/'reference-summary.json',summary)
    write_json(out/'expected.json',dict(indices=list(range(len(packed))),values=list(direct),
        scope='Public repeated39 cells; one final comparison across16384 slots; no model quality data',
        mathematical_comparison=comparison))
    source=dict(plan=str(out/'plan.json'),bundle=str(bundle),metadata=str(out/'metadata.json'),
        input=str(out/'input.bin'),expected=str(out/'expected.json'),inventory=inventory,
        base_metadata=str(base_metadata),missing_metadata=str(missing_metadata),
        math_comparison=comparison,source_model_execution=False,old_durations_reused=False)
    write_json(out/'preparation.json',source)
    return source


def build_source():
    """Add memory-only samples to a private disk-prepared client copy."""
    source=disk.build_source()
    source=disk._once(source,
        'func stage(name string, metadata map[string]any, fn func() error) (err error) {',
        'func stage(name string, metadata map[string]any, fn func() error) (err error) {\n'
        '\tif name=="runtime.compute" && preparedTimed {return calibrationStage(metadata,fn)}')
    source=disk._once(source,'\tfor index, op := range plan.Execution {',
        '\tfor index, op := range plan.Execution {\n\t\tcalibrationOperationBegin=time.Now()')
    source=disk._once(source,'\tpreparedDirFlag:=flag.String("prepared-dir","","fresh disk store for prepared public keys/plaintexts")',
        '\tcalibrationMetadataFlag:=flag.String("calibration-metadata","","explicit repeated public target metadata")\n'
        '\tpreparedDirFlag:=flag.String("prepared-dir","","fresh disk store for prepared public keys/plaintexts")')
    source=disk._once(source,'\tif *mode == "execute" {\n\t\tif checkInputSSA || traceKeyRequests',
        '\tif *mode == "execute" {\n'
        '\t\tif err:=calibrationLoadTargets(*calibrationMetadataFlag);err!=nil {panic(err)}\n'
        '\t\tif checkInputSSA || traceKeyRequests')
    source=disk._once(source,'elapsed:=time.Since(preparedBegin).Nanoseconds();preparedTimed=false',
        'elapsed:=time.Since(preparedBegin).Nanoseconds();preparedTimed=false\n\tcalibrationPublish()')
    return source+'\n\n'+Path(__file__).with_suffix('.go').read_text()


def summarize(client, metadata):
    """Validate complete fresh observations; no historical durations are inputs."""
    rows=client.get('calibration_observations')
    if not isinstance(rows,list) or len(rows)!=len(metadata['target_operations']):
        raise ValueError('Complete new target observations required')
    wanted={row['target_ordinal']:row for row in metadata['target_operations']}
    groups={row['index']:[] for row in metadata['cells']};seen=set()
    for row in rows:
        ordinal=row.get('ordinal');target=wanted.get(ordinal)
        if (target is None or ordinal in seen or row.get('cell_index')!=target['cell_index']
                or row.get('iteration')!=target['iteration'] or row.get('completed')is not True):
            raise ValueError('Unknown, duplicate or failed calibration target')
        seen.add(ordinal)
        for key in ('operation_ns','compute_ns'):
            value=row.get(key)
            if type(value)is not int or value<=0:raise ValueError('Positive integer new target duration required')
        if row['operation_ns']<row['compute_ns']:raise ValueError('Nested operation/compute clocks inconsistent')
        groups[row['cell_index']].append(row)
    cells=[]
    for cell in metadata['cells']:
        samples=sorted(groups[cell['index']],key=lambda row:row['iteration'])
        repeat=metadata['bootstrap_repetitions'] if cell['op']=='boot' else metadata['ordinary_repetitions']
        if len(samples)!=repeat:raise ValueError('Per-cell repetition allowance differs')
        stats={}
        for name in ('operation_ns','compute_ns'):
            values=[row[name] for row in samples]
            stats[name]=dict(samples=values,count=len(values),mean=statistics.mean(values),
                sample_std=statistics.stdev(values),min=min(values),max=max(values))
        cells.append(dict(cell=cell,repetitions=repeat,statistics=stats,
            iterations=[row['iteration']for row in samples]))
    return dict(status='ok',cells=cells,cell_count=len(cells),target_count=len(rows),
        parameter_group='r31-N16-C2S52-P61-scale50',old_durations_reused=False,
        cost_scope=metadata['measurement_scope'],host_page_cache_state='unforced',
        repeated_operation_order=metadata['observation_order'],
        source='Fresh single-context public microbenchmark; final strict CKKS validation required separately')


def project(plan, summary):
    if summary.get('status')!='ok' or summary.get('old_durations_reused') is not False:
        raise ValueError('Require complete new repeated calibration')
    measured={cell_key(row['cell']):row for row in summary['cells']}
    rows=[];missing=[]
    for cell in profile_cells(plan):
        key=cell_key(cell)
        if key not in measured:
            missing.append(cell);continue
        stat=measured[key]['statistics']['operation_ns'];count=cell['source_count']
        rows.append(dict(cell=cell,source_count=count,mean_unit_ns=stat['mean'],
            sample_std_unit_ns=stat['sample_std'],component_seconds=count*stat['mean']/1e9,
            repetitions=stat['count']))
    complete=not missing
    seconds=sum(row['component_seconds']for row in rows)
    ranking=sorted(rows,key=lambda row:row['component_seconds'],reverse=True)
    total=seconds;covered=0;major=[]
    for row in ranking:
        if covered>=.9*total:break
        covered+=row['component_seconds'];major.append(row)
    return dict(status='complete_operator_projection' if complete else 'partial_operator_projection',
        execution_count=len(plan['execution']),bootstrap_count=sum(row['op']=='boot'for row in plan['execution']),
        measured_cell_count=len(rows),missing_cells=missing,weighted_cells=rows,
        operator_component_prediction_s=seconds if complete else None,
        covered_operator_component_s=seconds,major90_percent_cells=major,
        major90_covered_fraction=covered/total if total else None,
        full_inference_prediction_s=None,
        unmeasured_components=['Input encryption and final decrypt/decode','Runtime loop outside operation spans',
            'Transfer between public-fixture representative rotation/key loads and full-model disk/cache schedule'],
        scope=summary['cost_scope'],fit_to_whole_model_clock=False,old_durations_reused=False)
