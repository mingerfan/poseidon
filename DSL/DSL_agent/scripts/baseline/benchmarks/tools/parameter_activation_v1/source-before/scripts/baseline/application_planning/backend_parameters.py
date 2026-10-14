"""Read the fixed Poseidon parameter and address-limit contracts.

This host-side module does not import layout submission, generation controllers,
Providers or historical candidate sandboxes. It never applies process limits.
"""
import ast
import hashlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def require(ok, code):
    if not ok:
        raise ValueError(code)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _integer(node):
    # Only evaluate the literal arithmetic used by the trusted sandbox policy.
    if isinstance(node,ast.Constant):
        require(type(node.value) is int and 0<=node.value<=2**48,"resource_limit_literal")
        return node.value
    require(isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Pow)),
            "resource_limit_expression")
    a,b=_integer(node.left),_integer(node.right)
    require(not isinstance(node.op,ast.Pow) or b<=16,"resource_limit_exponent")
    value=a*b if isinstance(node.op,ast.Mult) else a**b
    require(value<=2**48,"resource_limit_overflow")
    return value

def native_address_limit():
    """Read the existing policy without applying its limits to the host process."""
    from process_resource_limits import limits
    tree=ast.parse(inspect.getsource(limits))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and
           n.func.value.id=="resource" and n.func.attr=="setrlimit" and
           len(n.args)==2 and isinstance(n.args[0],ast.Attribute) and
           n.args[0].attr=="RLIMIT_AS"]
    require(len(calls)==1 and isinstance(calls[0].args[1],ast.Tuple),"resource_address_policy")
    values=tuple(_integer(n) for n in calls[0].args[1].elts)
    require(len(values)==2 and 0<values[0]==values[1],"resource_address_policy")
    return values[0]

def parameters(request):
    from compiler_configuration import request_configuration
    from poseidon_runtime_paths import SPEC
    from model_weight_package import regular_bytes
    from candidate_contract import strict_json
    config=request_configuration(request)
    require(config is not None and config["backend"]=="poseidon_cpu","resource_backend")
    raw=regular_bytes(SPEC,65536)
    require(sha(raw)==config["operator_spec_sha256"],"resource_spec_identity")
    spec=strict_json(raw.decode())
    context=spec["context"]
    degree=context["poly_degree"];count=len(context["rns_moduli_log2"])
    require(spec["spec_id"]=="poseidon-agent-cpu-v1" and
        context["context_id"]==config["parameter_identity"] and
        spec["levels"]==dict(lower_bound=0,upper_bound=count-1) and
        degree==32768 and count==13,"resource_parameter_contract")
    # Ciphertext::resize_internal allocates size * degree * data-modulus count
    # uint64_t coefficients. Input encryption produces two components.
    basis=["src/poseidon/ciphertext.cpp","src/poseidon/ciphertext.h",
           "src/poseidon/tools/agent_runtime/cpu_runner.cpp"]
    basis_hashes={name:sha(regular_bytes(ROOT/name,1024**2)) for name in basis}
    return dict(context_id=context["context_id"],operator_spec_sha256=sha(raw),
        poly_degree=degree,data_modulus_count=count,input_level=count-1,
        input_components=2,coefficient_bytes=8,source_basis_sha256=basis_hashes,
        source_basis=["src/poseidon/ciphertext.cpp:122",
                      "src/poseidon/ciphertext.h:57",
                      "src/poseidon/tools/agent_runtime/cpu_runner.cpp:91"])
