"""Optional generic statistical branch preparation, preserving dense convolutions."""
from benchmark_graph import digest
from delivery_component import bound
from pool_channel_normalization import normalize as pool_channels
from spatial_window_normalization import normalize as pool_windows
from reduction_branch_normalization import fuse_means,shard_retained_axes
from broadcast_projection_fusion import normalize as projections
from compact_graph_names import normalize as names
from singleton_reduction_normalization import normalize as singleton_reductions
from topological_metadata_scheduling import normalize as scheduling
from dyadic_pool_normalization import normalize as dyadic_pools


def prepare(model,*,pool_tile_elements=32,pool_channels_per_part=2):
    original=digest(model);steps=[]
    model,receipt=fuse_means(model);steps.append(receipt)
    model,receipt=pool_channels(model,channels_per_part=pool_channels_per_part);steps.append(receipt)
    model,receipt=dyadic_pools(model);steps.append(receipt)
    model,receipt=pool_windows(model,maximum_tile_output_elements=pool_tile_elements);steps.append(receipt)
    model,receipt=shard_retained_axes(model);steps.append(receipt)
    model,receipt=projections(model);steps.append(receipt)
    model,receipt=singleton_reductions(model);steps.append(receipt)
    model,receipt=scheduling(model);steps.append(receipt)
    model,receipt=names(model);steps.append(receipt)
    return model,bound(dict(format='poseidon-statistical-branch-preparation-v5',input_model_sha256=original,output_model_sha256=digest(model),steps=steps,
        pool_tile_elements=pool_tile_elements,pool_channels_per_part=pool_channels_per_part,model_name_dispatch=False,compiler_or_numeric_feedback_used=False,DSL_generated=False,native_authority=False))
