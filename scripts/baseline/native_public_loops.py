"""Bounded public range expansion without erasing decorated helper boundaries.

This is trusted AST transformation, not exec/eval. Existing native typing handles
the expanded statements. Python loops are a construction-time feature, never
encrypted control flow. Old native contracts never invoke this module.
"""
import ast
import copy
import hashlib
from function_construction import INTEGER_OPS
from hecate_contract import IDENTIFIER,rotation_literal
from seal_artifact_gate import require
import native_array_core as storage

MAX_ITERATIONS=4096
MAX_RANGE=128
MAX_NODES=4096
MAX_INTEGER=1048576

def expand(tree, constants, *, scalar_augmented=False):
    require(type(tree) is ast.Module and all(type(n) is ast.FunctionDef for n in tree.body),
            'Native loops require top-level function declarations')
    require(len(list(ast.walk(tree)))<=MAX_NODES,'Native loop source node bound')
    declared={n.name for n in tree.body}
    allowed={ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,
             ast.Expr,ast.For,ast.Call,ast.Name,ast.Load,ast.Store,ast.Attribute,
             ast.Constant,ast.List,ast.Tuple,ast.Subscript,ast.Slice,ast.Starred,
             ast.keyword,ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.FloorDiv,
             ast.Mod,ast.USub,ast.UAdd}
    if scalar_augmented: allowed.add(ast.AugAssign)
    for node in ast.walk(tree):
        require(type(node) in allowed,'Unsupported native loop syntax')
        if type(node) is ast.Call:
            if type(node.func) is ast.Name:
                require(node.func.id in declared|{'range'} and not node.keywords,
                        'Unknown/keyword native loop call')
            else:
                require(type(node.func) is ast.Attribute,'Unsupported native loop callable')
                if storage.constructor(node):
                    storage.check_constructor(node)
                elif (node.func.attr=='func' and type(node.func.value) is ast.Name
                      and node.func.value.id=='hc'):
                    pass  # The complete native signature check runs below expansion.
                else:
                    require(node.func.attr in (*storage.METHODS,'rotate') and not node.keywords,
                            'Unsupported native loop method')
        if type(node) is ast.Attribute:
            require(node.attr in (*storage.METHODS,'rotate','array','func','T'),
                    'Unsupported native loop attribute')
    forbidden=set(constants)|declared|{'hc','np','object','range','zero_ct'}
    result=copy.deepcopy(tree)
    records=[]; iterations=0; expanded_nodes=0

    class Substitute(ast.NodeTransformer):
        def __init__(self,indices): self.indices=indices
        def visit_Name(self,node):
            if type(node.ctx) is ast.Load and node.id in self.indices:
                return ast.copy_location(ast.Constant(self.indices[node.id]),node)
            return node
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if (type(node.op) in INTEGER_OPS and type(node.left) is ast.Constant and
                    type(node.right) is ast.Constant and type(node.left.value) is int and
                    type(node.right.value) is int):
                a,b=node.left.value,node.right.value
                require(abs(a)<=MAX_INTEGER and abs(b)<=MAX_INTEGER,'Public integer bound')
                try: value=INTEGER_OPS[type(node.op)](a,b)
                except (ZeroDivisionError,OverflowError) as error:
                    raise ValueError('Invalid public integer arithmetic') from error
                require(type(value) is int and abs(value)<=MAX_INTEGER,'Public integer result bound')
                return ast.copy_location(ast.Constant(value),node)
            return node
        def visit_UnaryOp(self,node):
            node=self.generic_visit(node)
            if (type(node.op) in (ast.USub,ast.UAdd) and type(node.operand) is ast.Constant
                    and type(node.operand.value) is int):
                value=(-1 if type(node.op) is ast.USub else 1)*node.operand.value
                require(abs(value)<=MAX_INTEGER,'Public integer bound')
                return ast.copy_location(ast.Constant(value),node)
            return node

    def rewrite(node,indices):
        return Substitute(indices).visit(copy.deepcopy(node))

    for fn in result.body:
        indices={}; params={a.arg for a in fn.args.args}
        loops=[n for n in ast.walk(fn) if type(n) is ast.For]
        index_names=set()
        for loop in loops:
            require(type(loop.target) is ast.Name and IDENTIFIER.fullmatch(loop.target.id) and
                    loop.target.id not in forbidden|params,'Invalid native induction variable')
            require(not loop.orelse and loop.type_comment is None,'Native for-else/type comments unsupported')
            index_names.add(loop.target.id)
        # Do not silently drop dangerous/unsupported statements even in zero-trip loops.
        for node in ast.walk(fn):
            if isinstance(node,ast.stmt):
                require(type(node) in (ast.FunctionDef,ast.Assign,ast.Expr,ast.For,ast.Return)
                        or scalar_augmented and type(node) is ast.AugAssign,
                        'Unsupported native loop statement')
                if type(node) is ast.FunctionDef: require(node is fn,'Nested native declaration unsupported')
            if type(node) in (ast.Assign,ast.AugAssign):
                targets=node.targets if type(node) is ast.Assign else [node.target]
                if type(node) is ast.AugAssign:
                    require(type(node.target) is ast.Name and type(node.op) in (ast.Add,ast.Sub,ast.Mult),
                            'Only scalar name +=, -= and *= are supported')
                require(all(type(t) in (ast.Name,ast.Tuple,ast.List) for t in targets),
                        'Native loops cannot write array cells or attributes')
                require(not any(type(t) is ast.Name and t.id in forbidden
                                for target in targets for t in ast.walk(target)),
                        'Native loops cannot overwrite reserved bindings')
                require(not any(type(t) is ast.Name and t.id in index_names
                                for target in targets for t in ast.walk(target)),
                        'Native induction variables are read-only')
            if type(node) is ast.Name and type(node.ctx) is ast.Store:
                require(node.id!='range','range is reserved')
        for loop in loops:
            require(not any(type(n) is ast.Return for stmt in loop.body for n in ast.walk(stmt)),
                    'Return inside native loop unsupported')

        def block(statements,active=()):
            nonlocal iterations,expanded_nodes
            require(len(active)<=16,'Native loop nesting bound')
            output=[]
            for stmt in statements:
                if type(stmt) is ast.For:
                    name=stmt.target.id
                    require(name not in active,'Nested native induction variable reuse')
                    call=rewrite(stmt.iter,indices)
                    require(type(call) is ast.Call and type(call.func) is ast.Name and call.func.id=='range'
                            and not call.keywords and 1<=len(call.args)<=3,
                            'Native loop iterable must be range of public integers')
                    values=[rotation_literal(n) for n in call.args]
                    require(all(abs(v)<=MAX_INTEGER for v in values),'Native range integer bound')
                    require(len(values)!=3 or values[2]!=0,'Native range step cannot be zero')
                    sequence=range(*values)
                    require(len(sequence)<=MAX_RANGE,'Native range length bound')
                    records.append(dict(function=fn.name,span=[stmt.lineno,stmt.col_offset,stmt.end_lineno,stmt.end_col_offset],
                                        arguments=values,trip_count=len(sequence)))
                    for value in sequence:
                        iterations+=1
                        require(iterations<=MAX_ITERATIONS,'Native expanded iteration bound')
                        indices[name]=value
                        output.extend(block(stmt.body,active+(name,)))
                    # A zero-trip range leaves a preceding induction binding intact,
                    # otherwise its last public value remains visible after the loop.
                else:
                    rewritten=rewrite(stmt,indices)
                    expanded_nodes+=len(list(ast.walk(rewritten)))
                    require(expanded_nodes<=MAX_NODES,'Native expanded syntax bound')
                    output.append(rewritten)
            return output
        fn.body=block(fn.body)
    require(len(list(ast.walk(result)))<=MAX_NODES,'Native expanded module bound')
    ast.fix_missing_locations(result)
    rendered=ast.unparse(result)
    return result,dict(schema=1,loops=records,iterations=iterations,
                       expanded_ast_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                       encrypted_control_flow=False,candidate_python_executed=False)
