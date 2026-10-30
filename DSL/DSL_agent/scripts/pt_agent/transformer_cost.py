"""Expose U4 per-tensor ciphertext counts and complete semantic cost groups."""

from collections import Counter

from .dsl.typecheck import check
from .kernels.layout import TokenLayout,ScoreLayout,multi_flat_layout,layout_blocks


def details(program,weights,backend,placement):
    checked=check(program,{k:v.shape for k,v in weights.items()})
    statements=Counter();boots=Counter()
    for nid,latency in placement.node_latency.items():statements[backend.prog_op[nid]]+=latency
    for nid in placement.boot_nodes:boots[backend.prog_op[nid]]+=1
    consumers={}
    for op in program.ops:
        for source in op.inputs:consumers.setdefault(source,[]).append(op)
    def feeds_attention(vid,seen):
        if vid in seen:return False
        seen.add(vid)
        for op in consumers.get(vid,()):
            if op.kind=='matmul':return True
            if op.kind in ('act','layernorm','rmsnorm','linear','softmax'):continue
            if feeds_attention(op.outputs[0],seen):return True
        return False
    groups=Counter();group_boots=Counter();rows=[]
    for op in program.ops:
        if op.kind=='matmul':group='attention_A' if op.attrs['transpose_b'] else 'attention_B'
        elif op.kind=='softmax':group='softmax'
        elif op.kind in ('layernorm','rmsnorm'):group='norm'
        elif op.kind=='linear':
            group=('lm_head' if op.outputs[0] in program.outputs and any(n.kind=='select' for n in program.ops)
                   else 'attention_projections' if feeds_attention(op.outputs[0],set()) else 'FFN')
        else:group='other'
        groups[group]+=statements[op.id];group_boots[group]+=boots[op.id]
        rows.append(dict(id=op.id,line=op.lineno,op=op.kind,group=group,
                         latency_us=statements[op.id],boots=boots[op.id]))
    ciphertexts={}
    for name,shape in checked.shapes.items():
        if shape is None:continue
        if len(shape)==3:layout=TokenLayout(program.header.slots,*shape[1:])
        elif len(shape)==4 and checked.layouts[name]=='score':
            layout=ScoreLayout(program.header.slots,shape[2],shape[1],shape[3])
        elif len(shape)==2:layout=multi_flat_layout(program.header.slots,shape[-1])
        else:continue
        ciphertexts[name]=len(layout_blocks(layout))
    return dict(ciphertext_counts=ciphertexts,statement_costs=rows,
                cost_breakdown={key:dict(latency_us=value,boots=group_boots[key]) for key,value in groups.items()})
