"""One original public-registry check before full request construction.

This is planning evidence only. It never constructs DSL, authorizes a Provider,
claims backend admission or evaluates a model/reference.
"""
from benchmark_graph import canonical,digest,validate,require
from delivery_component import bound

VERSION='poseidon-single-original-registry-preflight-v1'


def prepare(model,*,period,packing_version):
    from unified_capacity_constants import CHANNEL_VERSION,INDEX_VERSION,SHALLOW_VERSION
    from unified_capacity_layout import WIDE_PERIODS
    require(type(period)is int and period in WIDE_PERIODS,'Existing wide period required')
    require(type(packing_version)is str and packing_version in (CHANNEL_VERSION,INDEX_VERSION,SHALLOW_VERSION),'Existing packing required')
    validate(model);before=canonical(model)
    from unified_graph_contract import layout,registry,COMPACT
    physical=layout(model,period)
    public=registry(model,physical,COMPACT,packing_version)
    require(canonical(model)==before,'Original registry mutated graph')
    return public,bound(dict(format=VERSION,model_sha256=digest(model),layout_sha256=digest(physical),
        period=period,packing_version=packing_version,registry_sha256=digest(public),registry_entries=len(public),
        scalar_entries=sum(type(v)in(int,float)for v in public.values()),
        vector_entries=sum(type(v)is list for v in public.values()),
        registry_canonical_bytes=len(canonical(public)),original_registry_calls=1,
        full_factory_calls=0,model_reference_forwards=0,Provider_calls=0,DSL_generated=False,
        backend_admission=False,real_compiler=False,real_CKKS=False,model_name_dispatch=False))
