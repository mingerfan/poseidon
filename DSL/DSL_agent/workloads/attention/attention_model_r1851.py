"""B-owned complete two-head cached causal Attention model definition."""
import math,random
def build():
    rng=random.Random(18510037)
    weights={name:[[rng.uniform(-.045,.045)for _ in range(4)]for _ in range(4)]for name in ['wq','wk','wv','wo']}
    constants=dict(weights,scale=1/math.sqrt(2),causal_allow=[[1.,1.,0.],[1.,1.,1.]])
    nodes=[]
    def op(name,kind,refs,attrs=None,outputs=None):
        result=outputs or [name+'_v'];nodes.append(dict(id=name,op=kind,inputs=refs,attrs=attrs or {},outputs=result));return result[0]if len(result)==1 else result
    heads={}
    for kind in ['q','k','v']:
        projected=op(kind+'_projection','linear',['x','w'+kind]);shaped=op(kind+'_heads','reshape',[projected],dict(shape=[2,2,2]));heads[kind]=op(kind+'_head_order','permute',[shaped],dict(dims=[1,0,2]))
    for kind in ['q','k']:
        flat=op(kind+'_rows','reshape',[heads[kind]],dict(shape=[4,2]));rows=op(kind+'_tokens','split',[flat],dict(axis=0,sections=[1,1,1,1]),outputs=[kind+'_row_'+str(i)for i in range(4)]);rotated=[]
        for i,row in enumerate(rows):
            vector=op(kind+'_vec_'+str(i),'reshape',[row],dict(shape=[2]));rope=op(kind+'_rope_'+str(i),'rope',[vector],dict(position=i%2+1,theta=10000.));rotated.append(op(kind+'_rotated_row_'+str(i),'reshape',[rope],dict(shape=[1,2])))
        combined=op(kind+'_rotated_rows','concat',rotated,dict(axis=0));heads[kind]=op(kind+'_rotated_heads','reshape',[combined],dict(shape=[2,2,2]))
    keys=op('keys','concat',['cached_k',heads['k']],dict(axis=1));values=op('values','concat',['cached_v',heads['v']],dict(axis=1));kt=op('key_transpose','permute',[keys],dict(dims=[0,2,1]));scores=op('cipher_scores','matmul',[heads['q'],kt]);scaled=op('scale_scores','multiply',[scores,'scale'])
    exponent=op('score_exp','exp',[scaled],dict(approximation=dict(method='taylor',domain=[-1.,1.],degree=4)));masked=op('causal_mask','multiply',[exponent,'causal_allow']);denominator=op('row_denominator','sum',[masked],dict(axes=[-1],keepdims=True));inverse=op('inverse_denominator','reciprocal',[denominator],dict(approximation=dict(method='goldschmidt',domain=[1.,4.],iterations=3)));probability=op('probabilities','multiply',[masked,inverse]);attended=op('cipher_attend','matmul',[probability,values]);tokens=op('token_order','permute',[attended],dict(dims=[1,0,2]));merged=op('merge_heads','reshape',[tokens],dict(shape=[2,4]));output=op('output_projection','linear',[merged,'wo'])
    return dict(format='poseidon-operator-graph-v1',id='b3_causal_two_head_cached_attention_r1851',inputs=[dict(name='x',shape=[2,4]),dict(name='cached_k',shape=[2,1,2]),dict(name='cached_v',shape=[2,1,2])],constants=constants,nodes=nodes,outputs=[dict(name='output',value=output),dict(name='keys',value=keys),dict(name='values',value=values)])
