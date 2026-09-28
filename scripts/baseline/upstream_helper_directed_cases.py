"""Directed helper fixtures and task bindings; never provider reference answers."""
from benchmark_graph import digest
from upstream_candidate_cases import cases as silu_cases
from upstream_bn_candidate_cases import cases as bn_cases

def cases():
    selected=[]
    for row in silu_cases():
        if row['name'] in ('silu_frozen_0','silu_nested_native','silu_shared_rotation'):
            selected.append(dict(row,required_helpers=['HE_SiLU'],configuration='seal-cpu-eva-w40-v1'))
    for row in bn_cases():
        if row['name'] in ('bn_asymmetric_0','bn_then_linear','linear_then_bn','bn_nested_native','bn_multi_residual'):
            selected.append(dict(row,required_helpers=['HE_BN0','HE_BN1'] if row['name']=='bn_multi_residual' else ['HE_BN0'],configuration='seal-cpu-eva-w45-v1'))
    from upstream_concat_cases import cases as concat_cases
    for row in concat_cases():
        if row['name'] in ('concat_frozen_1','concat_square_negate','concat_then_linear'):
            selected.append(dict(row,profile='upstream-poly-concat-bn-silu-v3'))
    from upstream_spatial_cases import directed_cases as spatial_cases
    selected.extend(spatial_cases())
    from upstream_spatial_mapped_cases import directed_cases as mapped_cases
    selected.extend(mapped_cases())
    from upstream_fused_cases import directed_cases as fused_cases
    selected.extend(fused_cases())
    from upstream_downsample_cases import directed_cases as ds_cases
    selected.extend(ds_cases())
    from upstream_virtual_cases import directed_cases as virtual_cases
    selected.extend(virtual_cases())
    from upstream_chunk_cases import cases as chunk_cases
    selected.extend(chunk_cases())
    for row in selected:
        row['task_sha256']=digest({k:v for k,v in row.items() if k!='source'})
    return selected


def tasks():
    from benchmark_graph import signature
    rows=[]
    for row in cases():
        task=dict(id='upstream_'+row['name'],track='directed_upstream',
                  model=row['model'],model_sha256=row['model_sha256'],topology=signature(row['model'],True),
                  profile=row.get('profile','upstream-poly-bn-silu-v2'),required_helpers=row['required_helpers'],
                  compiler_configuration=row['configuration'],
                  acceptance='actual_trace_and_finite_return_intervention_and_encrypted_numerics',
                  execution_status='not_run',counts_as_new_corpus_model=False)
        if 'chunk_period' in row:task['chunk_period']=row['chunk_period']
        task['task_sha256']=digest(task);rows.append(task)
    return rows
