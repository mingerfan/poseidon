"""Synthetic variable-period native compositions; fixtures are NOT Agent outputs."""
import ast
from tensor_model_cases import affine,linear
from permutation_model_cases import cases as permutations
from packed_spatial_cases import large_cases

def cases():
    rows=[affine([2,3,5],[1,3,1],'native-alias'),
          dict(schema=5,id='native-packed-loop-rank4',input_shape=[2,2,4,8],constants={},
               nodes=[dict(id='square1',op='square',inputs=['x']),
                      dict(id='out',op='square',inputs=['square1'])],output='out'),
          linear([2,8],3,'native-six-output'),
          linear([2,32],2,'native-zero','zero_rows'),large_cases()[0],permutations()[0]]
    for i,row in enumerate(rows):row['id']='native-packed-'+str(i)+'-'+row['id']
    return rows

def candidate(payload,index,wrong=False):
    """Checked fixture transformation of the deterministic graph; never sent to LLM."""
    if index==0:
        body=ast.parse(payload['hecate_source']).body[0].body
        assert len(body)==3 and isinstance(body[0].value.op,ast.Mult) and isinstance(body[1].value.op,ast.Add)
        gain=body[0].value.right.id;bias=body[1].value.right.id
        alias='values.copy()' if wrong else 'values[:]'
        return ('@hc.func("c,p,p")\ndef affine_native(v,g,b):\n'
                '    values=np.array([v],dtype=object)\n'
                f'    view={alias}\n'
                '    view*=g\n    values+=b\n    return view[0]\n'
                f'@hc.func("c")\ndef golden(x):\n    return affine_native(*(x,{gain},{bias}))\n')
    if index==1:
        return ('@hc.func("c")\ndef polynomial(v):\n'
                '    for i in range(2):\n        v*=v\n    return v\n'
                '@hc.func("c")\ndef golden(x):\n    return polynomial(x)\n')
    tree=ast.parse(payload['hecate_source']);core=tree.body[0];core.name='model_core'
    returned=core.body[-1].value
    if isinstance(returned,ast.List):
        if wrong:returned.elts=list(reversed(returned.elts))
        core.body[-1].value=ast.Call(func=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='array',ctx=ast.Load()),
            args=[returned],keywords=[ast.keyword(arg='dtype',value=ast.Name(id='object',ctx=ast.Load()))])
    signature=core.decorator_list[0].args[0].value
    names=[n.arg for n in core.args.args]
    wrapper=ast.parse('@hc.func('+repr(signature)+')\ndef golden('+','.join(names)+'):\n'
                      '    return model_core(*('+','.join(names)+',))\n').body[0]
    tree.body.append(wrapper)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
