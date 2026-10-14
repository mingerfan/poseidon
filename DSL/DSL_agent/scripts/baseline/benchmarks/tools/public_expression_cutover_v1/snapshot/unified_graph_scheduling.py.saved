"""Schedule trusted scalar baseline DAGs; never rewrite Provider candidates.

Uses bound-factor reassociation, exact one-hot mask identities, trusted public
constant reuse and common ciphertext factors. Depth is a conservative diagnostic,
not a CKKS capacity proof. Model, request, layout and reference remain unchanged.
"""
import ast
from dataclasses import dataclass
import re
import math
from benchmark_graph import require

VERSION = "poseidon-rule-scheduling-v4"


@dataclass(frozen=True, eq=False)
class Node:
    index: int
    op: str
    inputs: tuple
    value: object
    plain: bool
    depth: int


class Schedule:
    def __init__(self, request, *, binary_masks=False, neutral_factors=False):
        self.request = request
        self.binary_masks = binary_masks
        self.neutral_factors = neutral_factors
        self.neutral_factor_removals = 0
        self.nodes = []
        self.cache = {}
        self.factors = {}
        self.moves = 0
        self.distributions = 0
        self.idempotent_masks = 0
        self.encrypted_public_products = 0
        self.power_two_hoists = 0
        self.common_products = 0
        self.trusted_zero = any(a == dict(dsl_name="zero_ct", kind="encrypted_zero",
            source="trusted_client", slot_period=request["layout"]["input_slot_period"])
            for a in request["layout"].get("auxiliary_ciphertexts", []))
        self.visit_count = 0

    def node(self, op, inputs=(), value=None, plain=False):
        key = (op, tuple(n.index for n in inputs), value, plain)
        if key in self.cache:
            return self.cache[key]
        require(len(self.nodes) < 4096, "Trusted schedule node budget")
        depth = max((n.depth for n in inputs), default=0)
        if op == "multiply" and not plain:
            depth += 1
        n = Node(len(self.nodes), op, tuple(inputs), value, plain, depth)
        self.nodes.append(n)
        self.cache[key] = n
        return n

    def leaf(self, name, plain=False):
        return self.node("name", value=name, plain=plain)

    def mask(self, n):
        if not n.plain or n.op != "name":
            return False
        if not self.binary_masks and re.fullmatch(r"mask[0-9]+", n.value) is None:
            return False
        value = self.request["public_constants"].get(n.value)
        return (type(value) is list and len(value) == self.request["layout"]["input_slot_period"]
                and all(type(v) in (float, int) and v in (0, 1) for v in value)
                and (any(value) if self.binary_masks else sum(value) == 1))

    def identity(self, n):
        if not self.neutral_factors or not n.plain or n.op != "name":
            return False
        value = self.request["public_constants"].get(n.value)
        return (type(value) in (int, float) and value == 1 or
                type(value) is list and len(value) == self.request["layout"]["input_slot_period"]
                and all(type(v) in (int, float) and v == 1 for v in value))

    def encrypted_public(self, n):
        factors = []
        while n.op == "multiply" and n.inputs[1].plain:
            factors.insert(0, n.inputs[1])
            n = n.inputs[0]
        if (self.trusted_zero and n.op == "add" and n.inputs[0].op == "name"
                and n.inputs[0].value == "zero_ct" and n.inputs[1].plain):
            return (n.inputs[1], *factors)
        return None

    def product(self, left, right):
        # Reuse already bound public factors of a trusted constant ciphertext.
        # No public arithmetic or newly derived constants are introduced.
        if not right.plain:
            public = self.encrypted_public(right)
            if public is not None:
                self.encrypted_public_products += 1
                return self.apply_factors(left, public)
            public = self.encrypted_public(left)
            if public is not None:
                self.encrypted_public_products += 1
                return self.apply_factors(right, public)
        return self.factor(left, right) if right.plain else self.node("multiply", (left, right))

    def addition(self, op, left, right):
        def factors(n):
            stack=[n];result=[];work=0
            while stack:
                child=stack.pop();work+=1
                if work>64:return None
                if child.op=="multiply" and all(not x.plain for x in child.inputs):
                    stack.extend(reversed(child.inputs))
                else:
                    result.append(child)
            return result
        if not left.plain and not right.plain:
            a,b=factors(left),factors(right)
            if a and b and len(a)>1 and len(b)>1:
                common=sorted(set(a)&set(b),key=lambda n:(-n.depth,n.index))
                if common:
                    shared=common[0];a.remove(shared);b.remove(shared)
                    def product(items):
                        value=items[0]
                        for item in items[1:]:
                            value=self.node("multiply",(value,item))
                        return value
                    remainder=self.node(op,(product(a),product(b)))
                    self.common_products+=1
                    return self.node("multiply",(remainder,shared))
        return self.node(op,(left,right))

    def power_two(self, p):
        value=self.request["public_constants"].get(p.value) if p.op=="name" else None
        return type(value) in (int,float) and value != 0 and math.frexp(abs(value))[0] == .5

    def apply_factors(self, n, factors, nesting=0):
        # Keep the validated single-factor schedule. Group distribution made
        # small normalization cases exceed AST limits without a depth benefit.
        ordered=tuple(p for p in factors if self.power_two(p))+tuple(p for p in factors if not self.power_two(p))
        if tuple(factors)!=ordered:
            self.power_two_hoists += 1
        result = n
        for p in ordered:
            result = self.factor(result, p, nesting)
        return result

    def factor(self, n, p, nesting=0):
        require(not n.plain and p.plain, "Ciphertext/plaintext factor types")
        require(nesting < 256, "Trusted schedule recursion budget")
        key = (n.index, p.index)
        if key in self.factors:
            return self.factors[key]
        self.visit_count += 1
        require(self.visit_count <= 8192, "Trusted schedule work budget")
        if self.identity(p):
            result = n
            self.neutral_factor_removals += 1
        elif (self.mask(p) and n.op == "multiply" and n.inputs[1] is p):
            # Exact 0/1 bound masks are idempotent. This does not apply to
            # arbitrary weights, overlapping different masks or rotations.
            result = n
            self.idempotent_masks += 1
        elif (n.op == "multiply" and n.inputs[1].plain and self.power_two(p)
              and not self.power_two(n.inputs[1])):
            result=self.node("multiply",(self.factor(n.inputs[0],p,nesting+1),n.inputs[1]))
            self.power_two_hoists += 1
        elif n.op == "multiply" and all(not x.plain for x in n.inputs):
            left, right = n.inputs
            if left.depth < right.depth:
                scaled = self.factor(left, p, nesting+1)
                result = self.node("multiply", (scaled, right))
                self.moves += 1
            elif right.depth < left.depth:
                scaled = self.factor(right, p, nesting+1)
                result = self.node("multiply", (left, scaled))
                self.moves += 1
            else:
                result = self.node("multiply", (n, p))
        elif self.mask(p) and n.op in ("add", "subtract"):
            parts = []
            for operand in n.inputs:
                if operand.plain:
                    # Keep public offsets in encrypted, masked lanes; no derived
                    # public values or unbound plaintext vectors are introduced.
                    encrypted = self.node("add", (self.leaf("zero_ct"), operand))
                    parts.append(self.node("multiply", (encrypted, p)))
                else:
                    parts.append(self.factor(operand, p, nesting+1))
            result = self.node(n.op, tuple(parts))
            self.distributions += 1
        else:
            result = self.node("multiply", (n, p))
        self.factors[key] = result
        return result

    def parse(self, source, *, rewrite):
        tree = ast.parse(source)
        require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef),
                "Trusted baseline function required")
        function = tree.body[0]
        require(function.name == "golden" and not function.args.vararg and
                not function.args.kwarg and not function.args.kwonlyargs and not function.args.defaults,
                "Trusted baseline signature")
        cipher_names = [s["dsl_name"] for s in self.request["layout"]["inputs"]] + ["zero_ct"]
        require([a.arg for a in function.args.args] == cipher_names, "Trusted input binding")
        env = {name:self.leaf(name) for name in cipher_names}
        public = {name:self.leaf(name, plain=True) for name in self.request["public_constants"]}
        env.update(public)

        def expression(value):
            if isinstance(value, ast.Name):
                require(value.id in env, "Trusted baseline undefined name")
                return env[value.id]
            if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                operand = expression(value.operand)
                require(not operand.plain, "Trusted encrypted negation")
                return self.node("negate", (operand,))
            if isinstance(value, ast.BinOp):
                names = {ast.Add:"add", ast.Sub:"subtract", ast.Mult:"multiply"}
                require(type(value.op) in names, "Trusted baseline arithmetic")
                left, right = expression(value.left), expression(value.right)
                require(not left.plain, "Trusted encrypted left operand")
                if isinstance(value.op, ast.Mult) and rewrite:
                    return self.product(left, right)
                if rewrite and isinstance(value.op,(ast.Add,ast.Sub)):
                    return self.addition(names[type(value.op)],left,right)
                return self.node(names[type(value.op)], (left, right))
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "rotate" and not value.keywords and len(value.args) == 1
                    and isinstance(value.args[0], ast.Constant) and type(value.args[0].value) is int):
                operand = expression(value.func.value)
                require(not operand.plain, "Trusted encrypted rotation")
                return self.node("rotate", (operand,), value=value.args[0].value)
            raise ValueError("Unsupported trusted baseline expression")

        require(function.body and isinstance(function.body[-1], ast.Return) and
                isinstance(function.body[-1].value, ast.List), "Trusted output binding")
        require(len(function.body)-1 <= 1024, "Trusted baseline operation budget")
        for statement in function.body[:-1]:
            require(isinstance(statement, ast.Assign) and len(statement.targets) == 1 and
                    isinstance(statement.targets[0], ast.Name), "Trusted baseline assignment")
            name = statement.targets[0].id
            require(re.fullmatch(r"v[0-9]+", name) is not None and name not in env,
                    "Trusted baseline single assignment")
            env[name] = expression(statement.value)
        outputs = [expression(value) for value in function.body[-1].value.elts]
        require(all(not n.plain for n in outputs), "Encrypted baseline returns")
        return outputs

    def source(self, original, outputs):
        assigned = {}
        lines = []
        operators = {"add":"+", "subtract":"-", "multiply":"*"}

        def emit(n, nesting=0):
            require(nesting < 256, "Trusted schedule serialization depth")
            if n.op == "name":
                return n.value
            if n.index in assigned:
                return assigned[n.index]
            children = [emit(c, nesting+1) for c in n.inputs]
            if n.op in operators:
                text = children[0]+" "+operators[n.op]+" "+children[1]
            elif n.op == "negate":
                text = "-"+children[0]
            elif n.op == "rotate":
                text = children[0]+".rotate("+str(n.value)+")"
            else:
                raise ValueError("Unsupported scheduled baseline operation")
            name = "v"+str(len(lines))
            lines.append("    "+name+" = "+text)
            require(len(lines) <= 1024, "Deterministic baseline operation budget")
            assigned[n.index] = name
            return name

        names = [emit(n) for n in outputs]
        source = "\n".join(original.splitlines()[:2]+lines+["    return ["+", ".join(names)+"]"])+"\n"
        require(len(source.encode()) <= 65536, "Rule source budget")
        return source, len(lines)


def schedule_source(source, request):
    """Return an auditable baseline schedule when bound-mask or public-constant rules reduce it."""
    original = Schedule(request)
    old_outputs = original.parse(source, rewrite=False)
    scheduled = Schedule(request)
    outputs = scheduled.parse(source, rewrite=True)
    old_depth = max((n.depth for n in old_outputs), default=0)
    depth = max((n.depth for n in outputs), default=0)
    record = dict(version=VERSION, applied=False, original_depth_bound=old_depth,
                  scheduled_depth_bound=depth, moved_plaintext_factors=scheduled.moves,
                  distributed_bound_masks=scheduled.distributions,
                  removed_duplicate_masks=scheduled.idempotent_masks,
                  reused_encrypted_public_values=scheduled.encrypted_public_products,
                  hoisted_power_of_two_factors=scheduled.power_two_hoists,
                  factored_common_ciphertext_products=scheduled.common_products,
                  request_changed=False, reference_changed=False, provider_candidate_rewritten=False)
    if depth > old_depth or not (depth < old_depth or scheduled.idempotent_masks or scheduled.encrypted_public_products or scheduled.power_two_hoists or scheduled.common_products):
        record["reason"] = "No smaller trusted multiplication schedule"
        return source, record
    result, operations = scheduled.source(source, outputs)
    record.update(applied=True, emitted_operations=operations)
    return result, record
