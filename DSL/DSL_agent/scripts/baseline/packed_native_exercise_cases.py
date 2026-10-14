"""Deterministic fixtures for finite packed-native construction exercises."""
import ast
from packed_native_cases import cases as baseline,candidate as base_candidate
from tensor_model_cases import affine

def entries():
    rows=baseline()
    models=[rows[0],rows[1],rows[2],rows[4],affine([2,2,4,4],[1,2,1,1],'zero-item'),
        dict(schema=5,id='scalar-aug',input_shape=[2,3],constants={},nodes=[
            dict(id='square',op='square',inputs=['x']),dict(id='out',op='add',inputs=['square','x'])],output='out')]
    specs=[
      ('pn-array-view',['array.view_inplace'],'Use a distinct shared-memory object-array view and apply a name-target ndarray +=, -= or *= through it; the changed shared storage must affect output. A copy is not a view.'),
      ('pn-loop',['loop.augmented'],'Use a public range loop with at least two executed iterations, performing scalar Expr augmented arithmetic whose results affect output.'),
      ('pn-star',['star.arguments'],'Use a starred positional helper call with at least two Expr arguments, each of which affects the final output.'),
      ('pn-matrix',['return.matrix'],'Return a matrix object array with at least two cells from a reachable decorated helper; at least two returned cells must affect output.'),
      ('pn-zero-item',['item.zero'],'Extract a zero-dimensional object array using item(); the extracted Expr must affect output.'),
      ('pn-scalar',['scalar.augmented'],'Use scalar Expr name-target augmented arithmetic whose result affects output, retaining original input aliases according to native semantics.')]
    return {name:dict(spec=dict(id=name,required_features=features,instruction=instruction),
                      model=dict(model,id='exercise-'+name)) for model,(name,features,instruction) in zip(models,specs)}

def candidate(payload,name,wrong=False):
    if name=='pn-array-view':return base_candidate(payload,0,wrong)
    if name=='pn-loop':return base_candidate(payload,1)
    if name=='pn-star':
        tree=ast.parse(payload['hecate_source'])
        class Lift(ast.NodeTransformer):
            def visit_BinOp(self,node):
                node=self.generic_visit(node)
                if isinstance(node.op,ast.Mult):
                    return ast.Call(func=ast.Name(id='product',ctx=ast.Load()),
                        args=[ast.Starred(value=ast.Tuple(elts=[node.left,node.right],ctx=ast.Load()),ctx=ast.Load())],keywords=[])
                return node
        tree=Lift().visit(tree)
        helper=ast.parse('@hc.func("c,p")\ndef product(v,w):\n    return v*w\n').body[0]
        tree.body.insert(0,helper)
        return ast.unparse(ast.fix_missing_locations(tree))+'\n'
    if name=='pn-matrix':
        tree=ast.parse(base_candidate(payload,4))
        value=tree.body[0].body[-1].value
        tree.body[0].body[-1].value=ast.Call(func=ast.Attribute(value=value,attr='reshape',ctx=ast.Load()),
                                           args=[ast.Constant(4),ast.Constant(4)],keywords=[])
        return ast.unparse(ast.fix_missing_locations(tree))+'\n'
    if name=='pn-zero-item':
        body=ast.parse(payload['hecate_source']).body[0].body
        gain=body[0].value.right.id;bias=body[1].value.right.id
        return ('@hc.func("c,p,p")\ndef wrapped(v,g,b):\n'
                '    return np.array(v*g+b,dtype=object)\n'
                f'@hc.func("c")\ndef golden(x):\n    return wrapped(x,{gain},{bias}).item()\n')
    if name=='pn-scalar':
        return '@hc.func("c")\ndef golden(x):\n    previous=x\n    x*=x\n    return x+previous\n'
    raise ValueError('Unknown fixture')

