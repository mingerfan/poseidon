"""Opt-in directed requirements for arbitrary unified graphs.

Reuse the audited packed native trace + finite-intervention checker. Unsupported
historical contracts remain in the ledger, never accepted by joining rule texts.
"""
import ast
from benchmark_graph import require

SPECS={
"unified-matrix":("return.matrix","Return a matrix Expr object array from a helper; at least two cells must contribute."),
"unified-item":("item.zero","Extract a zero-dimensional Expr object array with item(); it must contribute."),
"unified-star":("star.arguments","Expand at least two helper arguments with starred syntax; every argument must contribute."),
"unified-loop":("loop.augmented","Use at least two iterations of a public range loop with contributing scalar augmented assignment."),
"unified-scalar":("scalar.augmented","Use contributing scalar Expr augmented assignment."),
"unified-view":("array.view_inplace","Mutate an Expr object array sharing storage with a distinct view; changed storage must contribute."),
}

# Each entry has its own finite-influence or explicitly structural witness.
SPECS.update({'unified-native-call-empty_return': ('call.empty_return',
                                      'Reachably call a helper returning an empty tuple or list. '
                                      'Structural-only.'),
 'unified-native-call-forward': ('call.forward',
                                 'Call a helper declared later in source; its result must '
                                 'contribute.'),
 'unified-native-call-identity': ('call.identity',
                                  'Use a typed ciphertext identity helper with a contributing '
                                  'result.'),
 'unified-native-call-nested': ('call.nested',
                                'Invoke a helper from another helper; the nested result must '
                                'contribute.'),
 'unified-native-call-nested_array': ('call.nested_array',
                                      'Call an array-returning helper inside another helper; its '
                                      'cells must contribute.'),
 'unified-native-call-plain_return': ('call.plain_return',
                                      'Call a helper returning a public Expr that contributes to '
                                      'output.'),
 'unified-native-call-public_argument': ('call.public_argument',
                                         'Pass public Expr arguments to a helper; every public '
                                         'argument must contribute.'),
 'unified-native-call-repeated': ('call.repeated',
                                  'Call the same helper at least twice; results of at least two '
                                  'invocations must contribute.'),
 'unified-native-call-scalar_cipher': ('call.scalar_cipher',
                                       'Call a helper returning a scalar ciphertext Expr that '
                                       'contributes to output.'),
 'unified-native-call-tuple_multi': ('call.tuple_multi',
                                     'Use a helper returning at least two ciphertexts in a tuple; '
                                     'at least two results must contribute.'),
 'unified-native-call-two_cipher_arguments': ('call.two_cipher_arguments',
                                              'Use a helper with at least two ciphertext '
                                              'parameters; every ciphertext argument must '
                                              'contribute.'),
 'unified-native-call-zero_arguments': ('call.zero_arguments',
                                        'Use a zero-argument helper with a nonempty result that '
                                        'contributes.'),
 'unified-native-copy': ('copy', 'Copy object storage and use copied cells in a named output.'),
 'unified-native-golden-zero': ('golden.zero',
                                'Return zero-dimensional object storage directly from golden; '
                                'requires one named ciphertext output.'),
 'unified-native-index-reverse': ('index.reverse',
                                  'Use a negative-step object-array slice with at least two '
                                  'contributing cells.'),
 'unified-native-index-zero': ('index.zero',
                               'Extract a zero-dimensional object array using [()] and use its '
                               'cell.'),
 'unified-native-reshape-rank4': ('reshape.rank4',
                                  'Reshape object storage to rank four; at least two cells must '
                                  'contribute.'),
 'unified-native-return-empty': ('return.empty',
                                 'Reachably call a helper returning an empty object array. '
                                 'Structural-only, no numerical contribution claimed.'),
 'unified-native-return-mixed': ('return.mixed',
                                 'Return object storage with both public and ciphertext Expr '
                                 'cells; each kind must contribute.'),
 'unified-native-return-rank4': ('return.rank4',
                                 'Return a rank-four Expr object array from a helper; at least two '
                                 'cells must contribute.'),
 'unified-native-return-zero': ('return.zero',
                                'Return zero-dimensional object storage from a helper and use its '
                                'cell.'),
 'unified-native-star-array': ('star.array',
                               'Expand a nonempty object array into a helper; every positional '
                               'argument must contribute.'),
 'unified-native-star-call_result': ('star.call_result',
                                     'Expand an object-array helper result into another helper; '
                                     'all arguments must contribute.'),
 'unified-native-star-empty': ('star.empty',
                               'Reachably expand empty storage into a zero-argument helper. '
                               'Structural-only.'),
 'unified-native-star-flatten_matrix': ('star.flatten_matrix',
                                        'Flatten matrix object storage then expand into a helper; '
                                        'all arguments must contribute.'),
 'unified-native-star-list': ('star.list',
                              'Expand a nonempty list into a helper; every positional argument '
                              'must contribute.'),
 'unified-native-star-mixed': ('star.mixed',
                               'Mix nonempty starred and ordinary arguments in one call; every '
                               'argument must contribute.'),
 'unified-native-star-multiple': ('star.multiple',
                                  'Use multiple nonempty starred segments in one call; every '
                                  'argument must contribute.'),
 'unified-native-star-reverse': ('star.reverse',
                                 'Expand a reversed object array with at least two arguments; all '
                                 'must contribute.'),
 'unified-native-star-tuple': ('star.tuple',
                               'Expand a nonempty tuple into a helper; every positional argument '
                               'must contribute.'),
 'unified-native-transpose': ('transpose',
                              'Transpose object storage containing at least two contributing Expr '
                              'cells.'),
 'unified-native-unpack-rows': ('unpack.rows',
                                'Unpack at least two rows of object storage and use cells from '
                                'both rows.')})
STRUCTURAL = frozenset(("return.empty", "call.empty_return", "star.empty"))

# Aggregates are conjunctions, not aliases for a single representative syntax.
# Children are independently witnessed in the SAME candidate and real trace.
COMPOSITES={
    "native.call":("call.two_cipher_arguments","call.public_argument","call.nested","call.forward"),
    "native.return":("call.scalar_cipher","call.tuple_multi","call.plain_return","call.empty_return"),
    "native.starred":("star.arguments","star.array","star.call_result","star.flatten_matrix",
                      "star.list","star.mixed","star.multiple","star.reverse","star.tuple","star.empty"),
    "array.storage":("return.matrix","return.zero","index.zero","item.zero","return.mixed"),
    "array.reshape_transpose":("reshape.rank4","transpose","unpack.rows"),
}
for feature,children in COMPOSITES.items():
    SPECS['unified-composite-'+feature.replace('.','-')]=(feature,
        'Satisfy ALL registered subrequirements in the same candidate: '+', '.join(children)+
        '. Each numerical child needs its own executed finite output-influence witness. '
        'Empty return/expansion children are reachable structure only and receive no numerical credit. '
        'Outer object-array transforms rearrange complete Expr cells, never ciphertext slots. '
        'This bounded aggregate does not claim every Python/NumPy variant or all-input equivalence.')


def compose_golden(source,children):
    """Compose parallel identity witnesses with disjoint local/helper bindings.

    This is only a manual fixture generator; candidate acceptance never requires
    these names or this implementation. Normal and forward declaration order stay
    distinct. No exec/eval, model edits or reference rewriting.
    """
    tree=ast.parse(source);golden=next(n for n in tree.body if n.name=='golden')
    ret=golden.body[-1]
    require(type(ret) is ast.Return and type(ret.value) is ast.List,'Composite golden output ABI')
    current=ret.value.elts[0];before=[];after=[];body=[];outputs=[]
    occupied={n.id for n in ast.walk(tree) if type(n) is ast.Name}
    occupied|={n.arg for n in ast.walk(tree) if type(n) is ast.arg}
    occupied|={n.name for n in tree.body if type(n) is ast.FunctionDef}
    # All subrequirements must contribute, but fixture composition must not add
    # an artificial serial chain of rescale levels beyond the model's own depth.
    root='composite_root_input'
    while root in occupied:root='c_'+root
    occupied.add(root)
    body.append(ast.Assign(targets=[ast.Name(id=root,ctx=ast.Store())],value=current))
    for index,feature in enumerate(children):
        name=next(k for k,v in SPECS.items() if v[0]==feature)
        template=golden_variant('@hc.func("c")\ndef golden(composite_input):\n return [composite_input]\n',name)
        fragment=ast.parse(template);fn=next(n for n in fragment.body if n.name=='golden')
        prefix='composite'+str(index)+'_'
        while any(n.startswith(prefix) for n in occupied):prefix='c_'+prefix
        bound={n.id for n in ast.walk(fragment) if type(n) is ast.Name and type(n.ctx) is ast.Store}
        bound|={n.arg for n in ast.walk(fragment) if type(n) is ast.arg}
        bound|={n.name for n in fragment.body if type(n) is ast.FunctionDef and n.name!='golden'}
        bound.discard('composite_input');mapping={n:prefix+n for n in bound}
        class Rename(ast.NodeTransformer):
            def visit_Name(self,n):
                if n.id=='composite_input':return ast.copy_location(ast.Name(id=prefix+'input',ctx=n.ctx),n)
                if n.id in mapping:n.id=mapping[n.id]
                return n
            def visit_arg(self,n):
                n.arg=mapping.get(n.arg,n.arg);return n
            def visit_FunctionDef(self,n):
                n.name=mapping.get(n.name,n.name);return self.generic_visit(n)
        Rename().visit(fragment)
        body.append(ast.Assign(targets=[ast.Name(id=prefix+'input',ctx=ast.Store())],value=ast.Name(id=root,ctx=ast.Load())))
        body.extend(fn.body[:-1]);outputs.append(fn.body[-1].value.elts[0])
        position=fragment.body.index(fn);before.extend(fragment.body[:position]);after.extend(fragment.body[position+1:])
        occupied.update(mapping.values());occupied.add(prefix+'input')
    require(outputs,'Empty aggregate has no semantic requirements')
    combined=outputs[0]
    for value in outputs[1:]:combined=ast.BinOp(left=combined,op=ast.Add(),right=value)
    golden.body[-1:-1]=body
    ret.value.elts[0]=ast.BinOp(left=combined,op=ast.Mult(),right=ast.Constant(1/len(outputs)))
    tree.body[:0]=before;tree.body.extend(after)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'

def spec(name):
    require(name in SPECS,"Unknown unified directed requirement")
    feature,instruction=SPECS[name]
    value=dict(id=name,required_features=list(COMPOSITES.get(feature,(feature,))),instruction=instruction)
    if feature in COMPOSITES:value["aggregate_requirement"]=feature
    return value
def validate_request(request):
    value=request.get("construction_exercise")
    require(type(value) is dict and value==spec(value.get("id")),"Changed unified construction metadata")
def golden_variant(source,name):
    """Manual baseline witness; never counted as an Agent-generated program."""
    spec(name)
    if SPECS[name][0] in COMPOSITES:return compose_golden(source,COMPOSITES[SPECS[name][0]])
    tree=ast.parse(source);golden=next(n for n in tree.body if n.name=="golden")
    ret=golden.body[-1]
    require(isinstance(ret,ast.Return) and isinstance(ret.value,ast.List),"Unexpected baseline return")
    first=ast.unparse(ret.value.elts[0])
    feature=SPECS[name][0]
    body="covered_out = "+first+"\n"
    helpers=[]; forward=False; direct_zero=False
    def helper(signature, lines, name="coverage_helper"):
        helpers.extend(ast.parse('@hc.func("'+signature+'")\ndef '+name+'('+
            {"c":"a","c,c":"a,b","c,c,c":"a,b,c","c,p":"a,p","":""}[signature]+'):\n'+
            ''.join('    '+line+'\n' for line in lines)).body)
    if feature in ('return.matrix','return.rank4','reshape.rank4','transpose','copy','index.reverse','unpack.rows','return.mixed','call.nested_array'):
        if feature=='return.mixed':
            helper('c',['return np.array([a + 0.5, 0.5], dtype=object)'])
            body+='covered_array = coverage_helper(covered_out)\ncovered_out = covered_array[0] - covered_array[1]\n'
        elif feature=='call.nested_array':
            helper('c',['return np.array([a * 0.5, a * 0.5], dtype=object)'], 'coverage_inner')
            helper('c',['cells = coverage_inner(a)','return cells[0] + cells[1]'])
            body+='covered_out = coverage_helper(covered_out)\n'
        else:
            lines=['cells = np.array([[a * 0.5], [a * 0.5]], dtype=object)']
            if feature in ('return.rank4','reshape.rank4'):lines+=['cells = cells.reshape(1, 1, 2, 1)']
            lines+=['return cells'];helper('c',lines)
            body+='covered_array = coverage_helper(covered_out)\n'
            if feature in ('return.rank4','reshape.rank4'):body+='covered_array = covered_array.reshape(2, 1)\n'
            if feature=='transpose':
                body+='covered_array = covered_array.T\ncovered_out = covered_array[0,0] + covered_array[0,1]\n'
            elif feature=='unpack.rows':
                body+='covered_row0, covered_row1 = covered_array\ncovered_out = covered_row0[0] + covered_row1[0]\n'
            else:
                if feature=='copy':body+='covered_array = covered_array.copy()\n'
                if feature=='index.reverse':body+='covered_array = covered_array[::-1]\n'
                body+='covered_out = covered_array[0,0] + covered_array[1,0]\n'
    elif feature in ('return.zero','index.zero','item.zero'):
        helper('c',['return np.array(a, dtype=object)'])
        body+='covered_array = coverage_helper(covered_out)\ncovered_out = covered_array'+('[()]' if feature=='index.zero' else '.item()')+'\n'
    elif feature=='golden.zero':
        require(len(ret.value.elts)==1,'Zero-dimensional golden requires one output')
        body+='covered_out = np.array(covered_out, dtype=object)\n';direct_zero=True
    elif feature in STRUCTURAL:
        helper('', ['return np.array([], dtype=object)' if feature=='return.empty' else 'return ()'])
        body+=('coverage_helper(*[])' if feature=='star.empty' else 'coverage_helper()')+'\n'
    elif feature.startswith('star.'):
        if feature in ('star.multiple','star.mixed'):
            helper('c,c,c',['return a + b + c'])
            body+='covered_part = covered_out * 0.25\ncovered_out = coverage_helper(*[covered_part], covered_out * 0.5, *[covered_part])\n'
        else:
            helper('c,c',['return a + b'])
            if feature=='star.call_result':
                helper('c',['return np.array([a * 0.5, a * 0.5], dtype=object)'],'coverage_parts')
                body+='covered_out = coverage_helper(*coverage_parts(covered_out))\n'
            else:
                form='[covered_out * 0.5, covered_out * 0.5]'
                if feature=='star.tuple':form='(covered_out * 0.5, covered_out * 0.5)'
                elif feature in ('star.array','star.reverse'):form='np.array('+form+', dtype=object)'
                elif feature=='star.flatten_matrix':form='np.array([[covered_out * 0.5, covered_out * 0.5]], dtype=object)'
                body+='covered_array = '+form+'\n'
                suffix='[::-1]' if feature=='star.reverse' else '.flatten()' if feature=='star.flatten_matrix' else ''
                body+='covered_out = coverage_helper(*covered_array'+suffix+')\n'
    elif feature.startswith('call.'):
        if feature in ('call.plain_return','call.zero_arguments'):
            helper('',['return 0.5']);body+='covered_out = covered_out + coverage_helper() - 0.5\n'
        elif feature=='call.public_argument':
            helper('c,p',['return a + p']);body+='covered_out = coverage_helper(covered_out - 0.5, 0.5)\n'
        elif feature=='call.two_cipher_arguments':
            helper('c,c',['return a + b']);body+='covered_out = coverage_helper(covered_out * 0.5, covered_out * 0.5)\n'
        elif feature=='call.tuple_multi':
            helper('c',['return (a * 0.5, a * 0.5)']);body+='covered_a, covered_b = coverage_helper(covered_out)\ncovered_out = covered_a + covered_b\n'
        elif feature=='call.nested':
            helper('c',['return a'],'coverage_inner');helper('c',['return coverage_inner(a)']);body+='covered_out = coverage_helper(covered_out)\n'
        elif feature=='call.repeated':
            helper('c',['return a * 0.5']);body+='covered_out = coverage_helper(covered_out) + coverage_helper(covered_out)\n'
        else:
            helper('c',['return a']);body+='covered_out = coverage_helper(covered_out)\n';forward=feature=='call.forward'
    elif feature=='loop.augmented':
        body+='covered_step = covered_out * 0.25\ncovered_out = covered_out * 0.5\nfor i in range(2):\n    covered_out += covered_step\n'
    elif feature=='scalar.augmented':
        body+='covered_step = covered_out * 0.5\ncovered_out = covered_out * 0.5\ncovered_out += covered_step\n'
    elif feature=='array.view_inplace':
        body+='covered_array = np.array([covered_out * 0.5], dtype=object)\ncovered_view = covered_array.reshape(1)\ncovered_array += covered_out * 0.5\ncovered_out = covered_view[0]\n'
    else:raise ValueError('Unimplemented golden witness: '+feature)
    if direct_zero:ret.value=ast.Name(id='covered_out',ctx=ast.Load())
    else:ret.value.elts[0]=ast.Name(id='covered_out',ctx=ast.Load())
    golden.body[-1:-1]=ast.parse(body).body
    if forward:tree.body.extend(helpers)
    else:tree.body[:0]=helpers
    return ast.unparse(ast.fix_missing_locations(tree))+"\n"
