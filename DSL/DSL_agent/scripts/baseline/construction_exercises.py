"""Versioned targeted construction tests; never execute candidate Python.

An exercise narrows the output implementation form, not the model semantics.
Execution witnesses exclude dead branches. Expression perturbations exclude
unused expression padding; they are finite dependency evidence, not a proof.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

CATALOG = Path(__file__).with_name('construction-exercises-v1.json')
EXERCISES = json.loads(CATALOG.read_text())
TASK = 'hecate-function-synthesis-v19'


def exercise_spec(name):
    if type(name) is str and name.startswith('pn-'):
        from packed_native_exercises import exercise_spec as spec
        return spec(name)
    if type(name) is str and name.startswith('ns-'):
        from native_star_exercises import exercise_spec as spec
        return spec(name)
    if type(name) is str and name.startswith('na-'):
        from native_array_exercises import exercise_spec as spec
        return spec(name)
    if type(name) is str and name.startswith('nf-'):
        from native_function_exercises import exercise_spec as spec
        return spec(name)
    if type(name) is str and name.startswith('ou-'):
        from object_unary_exercises import exercise_spec as spec
        return spec(name)
    if type(name) is str and name.startswith('sc-'):
        from scalar_conversion_exercises import exercise_spec as spec
        return spec(name)
    if type(name) is str and name.startswith('oa-'):
        from object_arithmetic_exercises import exercise_spec as spec
        return spec(name)
    if type(name) is not str or name not in EXERCISES:
        raise ValueError('Unknown construction exercise')
    return copy.deepcopy(EXERCISES[name])


def descriptor(name):
    if type(name) is str and name.startswith('pn-'):
        from packed_native_exercises import descriptor as model
        return model(name)
    if type(name) is str and name.startswith('ns-'):
        from native_star_exercises import descriptor as model
        return model(name)
    if type(name) is str and name.startswith('na-'):
        from native_array_exercises import descriptor as model
        return model(name)
    if type(name) is str and name.startswith('nf-'):
        from native_function_exercises import descriptor as model
        return model(name)
    if type(name) is str and name.startswith('ou-'):
        from object_unary_exercises import descriptor as model
        return model(name)
    if type(name) is str and name.startswith('sc-'):
        from scalar_conversion_exercises import descriptor as model
        return model(name)
    if type(name) is str and name.startswith('oa-'):
        from object_arithmetic_exercises import descriptor as model
        return model(name)
    exercise_spec(name)
    return dict(schema=2, id='exercise-'+name, input_shape=[4],
                constants=dict(weight=[0.5], bias=[0.375]),
                nodes=[dict(id='scaled',op='multiply',inputs=['x','weight']),
                       dict(id='merged',op='add',inputs=['scaled','x']),
                       dict(id='out',op='add',inputs=['merged','bias'])], output='out')


def validate_exercise_request(request):
    spec = request.get('construction_exercise')
    if (request.get('task')=='hecate-periodic-packed-native-synthesis-v2' or
            type(spec) is dict and type(spec.get('id')) is str and spec['id'].startswith('pn-')):
        from packed_native_exercises import validate_exercise_request as validate
        return validate(request)
    if (request.get('task') == 'hecate-native-function-synthesis-v8' or
            type(spec) is dict and type(spec.get('id')) is str and spec['id'].startswith('ns-')):
        from native_star_exercises import validate_exercise_request as validate
        return validate(request)
    array_id = type(spec) is dict and type(spec.get('id')) is str and spec['id'].startswith('na-')
    if request.get('task') == 'hecate-native-function-synthesis-v4' or array_id:
        from native_array_exercises import validate_exercise_request as validate
        return validate(request)
    native_id = type(spec) is dict and type(spec.get('id')) is str and spec['id'].startswith('nf-')
    if request.get('task') == 'hecate-native-function-synthesis-v2' or native_id:
        from native_function_exercises import validate_exercise_request as validate
        return validate(request)
    if request.get('task') == 'hecate-function-synthesis-v22':
        from object_unary_exercises import validate_exercise_request as validate
        return validate(request)
    if request.get('task') == 'hecate-function-synthesis-v21':
        from scalar_conversion_exercises import validate_exercise_request as validate
        return validate(request)
    if request.get('task') == 'hecate-function-synthesis-v20':
        from object_arithmetic_exercises import validate_exercise_request as validate
        return validate(request)
    value = request.get('construction_exercise')
    if (request.get('task') != TASK or type(value) is not dict or
            value != exercise_spec(value.get('id')) or
            request.get('model') != descriptor(value['id'])):
        raise ValueError('Construction exercise specification or model changed')


def node_features(node):
    out = {'node.' + type(node).__name__}
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name):
            out.add('call.'+fn.id)
        elif isinstance(fn, ast.Attribute):
            out.add('call.'+fn.attr)
        if any(isinstance(a, ast.Starred) for a in node.args):
            out.add('expansion.star')
        if any(k.arg is None for k in node.keywords):
            out.add('expansion.kwstar')
    if isinstance(node, ast.Attribute):
        out.add('attr.'+node.attr)
    if isinstance(node, ast.BinOp):
        out.add('binary.'+type(node.op).__name__)
    if isinstance(node, ast.BoolOp):
        out.add('bool.'+type(node.op).__name__)
    if isinstance(node, ast.UnaryOp):
        out.add('unary.'+type(node.op).__name__)
    if isinstance(node, ast.Compare):
        out.update('compare.'+type(op).__name__ for op in node.ops)
    if isinstance(node, ast.Subscript):
        out.add('index.read')
        if isinstance(node.slice, ast.Slice):
            out.add('slice.read')
    if isinstance(node, (ast.Assign,ast.AugAssign)):
        targets = node.targets if isinstance(node,ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target,ast.Subscript):
                out.add('subscript.write')
                if isinstance(target.slice,ast.Slice):
                    out.add('slice.write')
    if isinstance(node, ast.AugAssign):
        out.add('aug.'+type(node.op).__name__)
    if isinstance(node, (ast.FunctionDef,ast.Lambda)):
        for name in ('posonlyargs','kwonlyargs','vararg','kwarg'):
            if getattr(node.args,name):
                out.add('signature.'+{'posonlyargs':'posonly','kwonlyargs':'kwonly'}.get(name,name))
    return out


def fingerprint(expanded):
    """Finite plaintext probe of already checked flat AST; no eval/exec."""
    def vector(value):
        data = value if isinstance(value,list) else [value]
        return [float(data[i % len(data)]) for i in range(4)]
    fn = ast.parse(expanded['source']).body[0]
    outputs = []
    for sample in ([0.,0.,0.,0.],[-0.8,0.25,0.6,1.]):
        values = {k:vector(v) for k,v in expanded['constants'].items()}
        values['x'] = list(sample)
        def expression(node):
            if isinstance(node,ast.Name):
                return values[node.id]
            if isinstance(node,ast.Constant):
                return vector(node.value)
            if isinstance(node,ast.UnaryOp):
                return [-v for v in expression(node.operand)]
            if isinstance(node,ast.BinOp):
                a,b=expression(node.left),expression(node.right)
                if isinstance(node.op,ast.Add): return [x+y for x,y in zip(a,b)]
                if isinstance(node.op,ast.Sub): return [x-y for x,y in zip(a,b)]
                if isinstance(node.op,ast.Mult): return [x*y for x,y in zip(a,b)]
            if isinstance(node,ast.Call):
                from hecate_contract import rotation_literal
                a=expression(node.func.value); step=rotation_literal(node.args[0]) % 4
                return a[step:]+a[:step]
            if isinstance(node,ast.List):
                return [v for n in node.elts for v in expression(n)]
            raise ValueError('Unexpected normalized expression')
        for statement in fn.body:
            if isinstance(statement,ast.Assign):
                values[statement.targets[0].id]=expression(statement.value)
            elif isinstance(statement,ast.Return):
                outputs.extend(expression(statement.value))
            else:
                raise ValueError('Unexpected normalized statement')
    return outputs


def expression_influence(source, constants, expected_outputs, witnesses, reference):
    from function_construction import normalize
    # Bound work per feature. Only already executed expression nodes are eligible.
    for span in sorted(witnesses)[:4]:
        tree=ast.parse(source)
        target=next((n for n in ast.walk(tree) if isinstance(n,ast.expr) and
                     (getattr(n,'lineno',0),getattr(n,'col_offset',0),
                      getattr(n,'end_lineno',0),getattr(n,'end_col_offset',0)) == span),None)
        if target is None:
            continue
        replacements=[ast.Constant(0),ast.Constant(1),
                      ast.BinOp(left=copy.deepcopy(target),op=ast.Mult(),right=ast.Constant(0)),
                      ast.List(elts=[ast.Constant(2),ast.Constant(3)],ctx=ast.Load()),
                      ast.List(elts=[ast.Constant('2'),ast.Constant('3'),ast.Constant('4')],ctx=ast.Load()),
                      ast.Constant('0'),ast.Constant('1'),ast.Constant('2.5')]
        if isinstance(target,ast.Call) and isinstance(target.func,ast.Attribute):
            replacements.append(target.func.value)
        if isinstance(target,ast.Call) and isinstance(target.func,ast.Attribute):
            replacements.append(ast.Call(func=ast.Attribute(value=copy.deepcopy(target),attr='replace',ctx=ast.Load()),
                                         args=[ast.Constant('0'),ast.Constant('1')],keywords=[]))
        if isinstance(target,ast.Call) and isinstance(target.func,ast.Attribute) and target.func.attr=='items':
            names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            suffix=0
            while 'probeKey'+str(suffix) in names or 'probeValue'+str(suffix) in names:
                suffix+=1
            key='probeKey'+str(suffix);value='probeValue'+str(suffix)
            replacements.append(ast.parse('[(%s,%s*0.5) for %s,%s in (%s)]' %
                (key,value,key,value,ast.unparse(target)),mode='eval').body)
        for replacement in replacements:
            class Perturb(ast.NodeTransformer):
                def visit(self,node):
                    if (isinstance(node,ast.expr) and
                        (getattr(node,'lineno',0),getattr(node,'col_offset',0),
                         getattr(node,'end_lineno',0),getattr(node,'end_col_offset',0)) == span):
                        return ast.copy_location(copy.deepcopy(replacement),node)
                    return super().visit(node)
            try:
                changed=ast.unparse(ast.fix_missing_locations(Perturb().visit(ast.parse(source))))
                expanded=normalize(changed,constants,expected_outputs,public_mappings=True)
                probe=fingerprint(expanded)
                if len(probe)==len(reference) and any(abs(a-b)>1e-8 for a,b in zip(probe,reference)):
                    return dict(status='output_changed',line=span[0],column=span[1],
                                replacement=ast.unparse(replacement))
            except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
                continue
    return dict(status='not_demonstrated')



def structural_influence(source, constants, outputs, spans, reference, feature):
    from function_construction import normalize
    for span in sorted(spans)[:4]:
        class Perturb(ast.NodeTransformer):
            def visit(self,node):
                position=(getattr(node,'lineno',0),getattr(node,'col_offset',0),
                          getattr(node,'end_lineno',0),getattr(node,'end_col_offset',0))
                if position==span and isinstance(node,(ast.For,ast.While)):
                    node.orelse=[]
                    return node
                if feature=='call.clear' and isinstance(node,ast.Expr):
                    value=node.value
                    pos=(getattr(value,'lineno',0),getattr(value,'col_offset',0),
                         getattr(value,'end_lineno',0),getattr(value,'end_col_offset',0))
                    if pos==span:
                        return ast.copy_location(ast.Pass(),node)
                return super().visit(node)
        try:
            changed=ast.unparse(ast.fix_missing_locations(Perturb().visit(ast.parse(source))))
            result=normalize(changed,constants,outputs,public_mappings=True)
            values=fingerprint(result)
            if len(values)==len(reference) and any(abs(a-b)>1e-8 for a,b in zip(values,reference)):
                return dict(status='output_changed',line=span[0],column=span[1],
                            replacement='omit_else' if feature.startswith('event.') else 'omit_clear')
        except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
            continue
    return dict(status='not_demonstrated')



def items_consumer_influence(source, constants, outputs, spans, reference):
    """Preserve view creation timing: perturb pair values when consumed."""
    from function_construction import normalize
    def position(node):
        return (getattr(node,'lineno',0),getattr(node,'col_offset',0),
                getattr(node,'end_lineno',0),getattr(node,'end_col_offset',0))
    tree=ast.parse(source)
    aliases={n.targets[0].id for n in ast.walk(tree) if isinstance(n,ast.Assign) and
             len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and position(n.value) in spans}
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and
           (position(n.iter) in spans or isinstance(n.iter,ast.Name) and n.iter.id in aliases) and
           isinstance(n.target,(ast.Tuple,ast.List)) and len(n.target.elts)==2 and
           isinstance(n.target.elts[1],ast.Name)]
    for loop in loops[:4]:
        target=position(loop);value=loop.target.elts[1].id
        class Perturb(ast.NodeTransformer):
            def visit_For(self,node):
                if position(node)==target:
                    assignment=ast.parse(value+' = '+value+' * 0.5').body[0]
                    node.body.insert(0,ast.copy_location(assignment,node))
                    return node
                return self.generic_visit(node)
        try:
            changed=ast.unparse(ast.fix_missing_locations(Perturb().visit(ast.parse(source))))
            expanded=normalize(changed,constants,outputs,public_mappings=True)
            values=fingerprint(expanded)
            if len(values)==len(reference) and any(abs(a-b)>1e-8 for a,b in zip(values,reference)):
                return dict(status='output_changed',line=target[0],column=target[1],
                            replacement='scale_live_items_value_at_consumption')
        except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
            continue
    return dict(status='not_demonstrated')


def check_exercise(source, request):
    if request.get('task')=='hecate-periodic-packed-native-synthesis-v2':
        from packed_native_exercises import check_exercise as check
        return check(source,request)
    if request.get('task') == 'hecate-native-function-synthesis-v8':
        from native_star_exercises import check_exercise as check
        return check(source, request)
    if request.get('task') == 'hecate-native-function-synthesis-v4':
        from native_array_exercises import check_exercise as check
        return check(source, request)
    if request.get('task') == 'hecate-native-function-synthesis-v2':
        from native_function_exercises import check_exercise as check
        return check(source,request)
    if request.get('task') == 'hecate-function-synthesis-v22':
        from object_unary_exercises import check_exercise as check
        return check(source,request)
    if request.get('task') == 'hecate-function-synthesis-v21':
        from scalar_conversion_exercises import check_exercise as check
        return check(source,request)
    if request.get('task') == 'hecate-function-synthesis-v20':
        from object_arithmetic_exercises import check_exercise as check
        return check(source,request)
    from function_construction import normalize
    validate_exercise_request(request)
    required=request['construction_exercise']['required_features']
    observed={}
    def observe(node):
        if isinstance(node,tuple):
            event,node=node
            features={'event.'+event}
        else:
            features=node_features(node)
        span=(getattr(node,'lineno',0),getattr(node,'col_offset',0),
              getattr(node,'end_lineno',0),getattr(node,'end_col_offset',0))
        for feature in features:
            observed.setdefault(feature,set()).add(span)
    expanded=normalize(source,request['public_constants'],request['layout']['output_ciphertexts'],
                       public_mappings=True,observe=observe)
    counters=expanded['construction']
    missing=[f for f in required if
             (counters.get(f[8:],0)<=0 if f.startswith('counter.') else not observed.get(f))]
    if missing:
        raise ValueError('Required construction not executed: '+', '.join(missing))
    reference=fingerprint(expanded)
    influence={}
    for feature in required:
        sensitive_calls = {'float','int','pow','strip','lstrip','rstrip','split','rsplit',
                           'partition','rpartition','replace','join','floor','ceil','log2','items'}
        if (feature.startswith(('binary.','bool.','unary.','compare.')) or
            feature in {'attr.coef','attr.domain','attr.window'} or
            feature.startswith('call.') and feature[5:] in sensitive_calls):
            influence[feature]=expression_influence(source,request['public_constants'],
                                                    request['layout']['output_ciphertexts'],
                                                    observed[feature],reference)
    if 'call.items' in influence and influence['call.items']['status']=='not_demonstrated':
        influence['call.items']=items_consumer_influence(source,request['public_constants'],
            request['layout']['output_ciphertexts'],observed['call.items'],reference)
    for feature in required:
        if feature in ('event.for_else','event.while_else','call.clear'):
            influence[feature]=structural_influence(source,request['public_constants'],
                request['layout']['output_ciphertexts'],observed[feature],reference,feature)
    insensitive=[f for f,r in influence.items() if r['status']!='output_changed']
    if insensitive:
        raise ValueError('Required expression has no demonstrated output influence: '+
                         ', '.join(insensitive)+'. Use it to derive an output operand, not padding.')
    return dict(schema=1,id=request['construction_exercise']['id'],required=required,
                executed={f:[list(span) for span in sorted(observed.get(f,()))]
                          for f in required if not f.startswith('counter.')},
                counters={f:counters[f[8:]] for f in required if f.startswith('counter.')},
                expression_influence=influence,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                interpretation='Executed constructs plus finite expression perturbations; structural '
                    'mutation/control/argument semantics require separate source review. Not all-input proof.')
