"""Opt-in public construction/depth guidance; never an answer rewriter."""
import ast

VERSION = "explicit-v9"
NATIVE = (
    "Each decorated native function has at most 16 positional parameters. "
    "Do not pass every public coefficient as a helper argument: supplied named public constants "
    "may be referenced from the helper's trusted lexical scope. Preserve the exact binding at each "
    "call site; specialize the helper or split it when calls need different constants. "
    "Do not shadow public names, invent dictionaries, or use defaults/varargs.")
TYPES = (
    "A supplied Plain Expr is not a public Python scalar. Do not negate a Plain Expr directly: "
    "when mathematically appropriate, negate the resulting ciphertext product instead. "
    "Public arithmetic is restricted to the selected construction contract.")
SLOTS = (
    "Ciphertext indexing is not logical tensor slicing. Object-container cells are whole Expr values. "
    "Implement a logical slice using its frozen C-order input/output slot mapping, admitted rotations "
    "and public masks; exclude tail padding. Never change the layout to accept Python-style ciphertext indexing.")
LOOPS = (
    "In the native contract, use the admitted bounded range of public integers and admitted container "
    "indexing. enumerate, zip, dictionary iteration and list append are not implicit permissions. "
    "Use preallocated allowed containers or a compact statically unrolled computation within the AST/work bounds.")
DEPTH = (
    "Prefer addition for exact ciphertext doubling: reuse the product and add it to itself, rather "
    "than multiplying it by an encoded Plain two at every recurrence level. "
    "Balanced basis construction alone does not remove the extra depth from Plain multiplications. "
    "Preserve every nonzero coefficient, including tiny coefficients, and all node-specific constants.")
FUSION = (
    "For Chebyshev series, x*T0(x)=T1(x), and x*Tn(x)=(T(n+1)(x)+T(n-1)(x))/2 for n>=1. "
    "These identities permit fusing multiplication by the SAME polynomial argument and a public "
    "constant offset into a new exact series before encrypted evaluation. They do not apply to an "
    "unrelated ciphertext or across a nonlinear composition. Derive coefficients solely from public "
    "model data using admitted public computation or finite allowed literals; retain every contribution. "
    "Do not truncate, change the approximation/domain, omit a required helper call, or adjust scale metadata.")
RESPONSE = (
    "Return only the required candidate JSON. Keep helper signatures and polynomial evaluation compact "
    "within the unchanged source/AST/work limits. Do not emit narrative, duplicate unused programs or "
    "attempt an unsupported dynamic coefficient algorithm.")
def specification(request):
    native = "construction_profile" not in request
    return dict(native_helpers=NATIVE if native else
        "Public construction helpers remain undecorated and follow their own unchanged argument/work limits.",
        type_boundary=TYPES, slot_boundary=SLOTS,
        native_loops=LOOPS if native else
        "Only the selected public-construction contract determines admitted iteration and methods.",
        depth=DEPTH, exact_series_fusion=FUSION, response_size=RESPONSE,
        authority="Public identities and contract clarification only; no model implementation, test values or reference.")
def repair_hint(request, source, diagnostic="", stage="static_check"):
    if request.get("generation_guidance",{}).get("version") != VERSION:
        return ""
    if not isinstance(source,str) or len(source.encode())>65536:
        return ""
    try:
        tree=ast.parse(source)
    except (SyntaxError,ValueError,RecursionError):
        return ""
    nodes=list(ast.walk(tree))
    if len(nodes)>4096:return ""
    hints=[]
    native="construction_profile" not in request
    if native and (any(isinstance(n,ast.FunctionDef) and len(n.args.args)>16 for n in nodes)
                   or "parameter limit" in diagnostic):
        hints.append(NATIVE)
    if native and "Native negation" in diagnostic:hints.append(TYPES)
    if "Indexing requires a result container" in diagnostic:hints.append(SLOTS)
    if native and any(s in diagnostic for s in ("native loop","Flat container limit")):
        hints.append(LOOPS)
    if stage in ("compiler","artifact_gate"):
        constants=request.get("public_constants",{})
        def two(n):
            return (isinstance(n,ast.Constant) and type(n.value) in (int,float) and n.value==2 or
                    isinstance(n,ast.Name) and type(constants.get(n.id)) in (int,float) and constants[n.id]==2)
        if any(isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mult) and
               (two(n.left) or two(n.right)) for n in nodes):hints.append(DEPTH)
        if any(n.get("op")=="polynomial" and n.get("attrs",{}).get("basis")=="chebyshev"
               for n in request["model"]["nodes"]):
            hints.append(FUSION)
        hints.append("The failing candidate has not established backend feasibility. Preserve the frozen "
                     "compiler configuration and mathematical graph; do not set level/scale or add bootstrap.")
    return "\n".join(dict.fromkeys(hints))[:5000]
