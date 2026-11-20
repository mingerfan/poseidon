"""Read the existing model format using independent read-only member mappings."""

import json
from pathlib import Path
import zipfile


def load_mapped_info(directory):
    from .info import ModelInfo
    from ..intake.graph import Graph, Node, TensorInfo
    from ..intake.core_ops import CORE_OPS
    from ..experiments.qwen_embedding_validation import map_stored_array
    directory=Path(directory)
    info_path=directory if directory.is_file() else directory/'model_info.json'
    record=json.loads(info_path.read_text())
    if record['graph'] is None:
        raise ValueError('Mapped model information requires the original saved graph')
    graph_dir=info_path.parent/record['graph']
    graph_record=json.loads((graph_dir/'graph.json').read_text())
    archive_path=graph_dir/'weights.npz'
    with zipfile.ZipFile(archive_path) as archive:
        entries=archive.infolist()
    if len({item.filename for item in entries}) != len(entries) or any(not item.filename.endswith('.npy') for item in entries):
        raise ValueError('The saved weight archive requires unique NPY member names')
    weights,metadata={},[]
    for item in entries:
        value,observation=map_stored_array(archive_path,item.filename,_entry=item)
        weights[item.filename[:-4]]=value
        metadata.append(observation)
    nodes=[]
    for node_id,op,inputs,outputs,attrs,source in graph_record['nodes']:
        for key,value in list(attrs.items()):
            if CORE_OPS[op]['attrs'].get(key) in ('tuple','pair'):attrs[key]=tuple(value)
        nodes.append(Node(node_id,op,inputs,outputs,attrs,source))
    tensors={name:TensorInfo(name,tuple(shape),kind) for name,shape,kind in graph_record['tensors']}
    graph=Graph(graph_record['name'],nodes,tensors,weights,graph_record['inputs'],graph_record['outputs'])
    if 'intake' in record['source']:graph.intake_stats=record['source']['intake']
    record['graph']=graph
    info=ModelInfo(**record)
    info.mapped_weights=dict(members=len(metadata),logical_weight_bytes=sum(item['tensor_bytes'] for item in metadata),
        direct_readonly_mapping=True,full_tensor_copies=0,integrity_checks=0,entries=metadata)
    return info
