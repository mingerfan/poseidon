"""Lexical cells for the bounded AST interpreter; never execute candidate code."""
import ast
from seal_artifact_gate import require

UNBOUND = object()


class Frame:
    def __init__(self, local_names=(), parent=None, nonlocals=(), *, module=False):
        self.values = dict.fromkeys(local_names, UNBOUND)
        self.parent = parent
        self.nonlocals = frozenset(nonlocals)
        self.module = module

    def owner(self, name):
        if name in self.nonlocals:
            frame = self.parent
            while frame is not None and not frame.module:
                if name in frame.values:
                    return frame
                frame = frame.parent
            raise ValueError('No enclosing nonlocal binding: '+name)
        if name in self.values:
            return self
        return self.parent.owner(name) if self.parent is not None else None

    def __contains__(self, name):
        return self.owner(name) is not None

    def __getitem__(self, name):
        frame = self.owner(name)
        if frame is None:
            raise KeyError(name)
        return frame.values[name]

    def get(self, name, default=None):
        return self[name] if name in self else default

    def __setitem__(self, name, value):
        frame = self.owner(name) if name in self.nonlocals else self
        frame.values[name] = value

    def update(self, values):
        for name, value in dict(values).items():
            self[name] = value


def analyze_scopes(tree, *, call_binding=False, public_iteration=False, function_literals=False):
    """Collect locals without descending into a child function's body.

    Python's compiler is used only for static symbol-table legality (including
    nonlocal use-before-declaration). Its code object is discarded, never run.
    AST size/allowlist/signatures are checked by the caller first.
    """
    try:
        compile(tree, '<bounded-construction-scope-check>', 'exec', dont_inherit=True)
    except (SyntaxError, RecursionError, MemoryError) as error:
        raise ValueError('Invalid lexical scope: '+str(error)) from error
    scopes = {}

    def visit_function(fn):
        locals_ = {a.arg for a in fn.args.args}
        if call_binding:
            from construction_calls import parameter_nodes
            locals_ = {a.arg for a in parameter_nodes(fn.args)}
        nonlocals = set()
        pending = list(fn.body)
        children = []
        while pending:
            node = pending.pop()
            if type(node) is ast.FunctionDef:
                locals_.add(node.name)
                children.append(node)
            elif type(node) is ast.Nonlocal:
                nonlocals.update(node.names)
            elif public_iteration and type(node) in (ast.ListComp, ast.DictComp):
                # Only the first iterable is evaluated in this function frame.
                # Comprehension targets belong to a separate implicit scope.
                pending.append(node.generators[0].iter)
            else:
                if type(node) is ast.Name and type(node.ctx) is ast.Store:
                    locals_.add(node.id)
                pending.extend(ast.iter_child_nodes(node))
        scopes[id(fn)] = (frozenset(locals_ - nonlocals), frozenset(nonlocals))
        for child in children:
            visit_function(child)

    for fn in tree.body:
        visit_function(fn)
    if function_literals:
        from construction_calls import parameter_nodes
        for node in ast.walk(tree):
            if type(node) is ast.Lambda:
                scopes[id(node)] = (frozenset(a.arg for a in parameter_nodes(node.args)), frozenset())
    return scopes
