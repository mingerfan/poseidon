"""Versioned explanations of registered directed semantics; no model solutions.

Metadata only. Existing validators, tracing and intervention gates remain the
authority. Do not import golden recipe functions or inspect test/reference data.
"""
import unified_graph_exercises as native
import unified_public_exercises as public

MEANINGS = {
    "attr.T": "Transpose whole cells of an object array. The receiver must be object storage; this does not rotate ciphertext slots.",
    "attr.shape": "Read the public shape tuple of a numeric or object array. A ciphertext Expr itself has no accessible shape here; use the immutable layout for logical tensor dimensions.",
    "attr.size": "Read the public number of cells in object-array storage. The receiver must be an object array, not a ciphertext Expr or a numeric array in this contract.",
    "attr.ndim": "Read the public rank of object-array storage. The receiver must be an object array, not a ciphertext Expr or a numeric array in this contract.",
    "attr.coef": "Read coefficients from bounded public Chebyshev data, not from a ciphertext or arbitrary object.",
    "attr.domain": "Read the domain array of bounded public Chebyshev data. The receiver is a public polynomial object; this is not a ciphertext attribute.",
    "attr.window": "Read the window array of bounded public Chebyshev data. The receiver is a public polynomial object; this is not a ciphertext attribute.",
    "binary.Add": "Use addition in the construction AST with permitted numeric, sequence, object-storage or symbolic-expression operands; their existing type and overload rules still apply.",
    "binary.Sub": "Use subtraction in the construction AST with permitted public or symbolic-expression operands; their existing type and overload rules still apply.",
    "binary.Mult": "Use multiplication in the construction AST with permitted numerical, bounded sequence, object-storage or symbolic-expression operands; existing type and overload rules still apply.",
    "binary.Div": "Use true division on public numerical values; the divisor must be nonzero.",
    "binary.FloorDiv": "Use floor division on public numerical values; the divisor must be nonzero.",
    "binary.Mod": "Use remainder on public numerical values; the divisor must be nonzero.",
    "binary.Pow": "Raise a public numerical value to an admitted bounded public exponent. This does not permit arbitrary ciphertext powers.",
    "bool.And": "Use public short-circuit and: operands are evaluated left to right and the selected operand is returned. No ciphertext-dependent condition is permitted.",
    "bool.Or": "Use public short-circuit or: operands are evaluated left to right and the selected operand is returned. No ciphertext-dependent condition is permitted.",
    "unary.Not": "Negate the truth value of an admitted public value; never test a ciphertext condition.",
    "compare.In": "Test public membership in an admitted public container or sequence.",
    "compare.NotIn": "Test public non-membership in an admitted public container or sequence.",
    "call.Chebyshev": "Construct bounded public Chebyshev data from a nonempty real coefficient vector and admitted public domain/window parameters. This is not an unrestricted numerical library call.",
    "call.Empty": "Construct the admitted symbolic Empty identity. Its operation must contribute; it is not encryption, a ciphertext constant, or bootstrap.",
    "call.array": "Construct an admitted numeric array from finite public data, or explicit object storage under the object-array rules. Never coerce a ciphertext Expr to numeric data.",
    "call.asarray": "Construct admitted numeric public array data. Do not coerce a ciphertext Expr to public numerical data.",
    "call.ceil": "Apply the admitted public numerical ceiling operation to a public real scalar or numeric array.",
    "call.floor": "Apply the admitted public numerical floor operation to a public real scalar or numeric array.",
    "call.log2": "Apply the admitted public numerical base-two logarithm within its valid positive domain.",
    "call.concatenate": "Concatenate admitted object arrays along the allowed public storage axis. Cells are complete Expr objects, not ciphertext slots.",
    "call.copy": "Copy object-array storage and consume the copied cells. A copy is distinct storage.",
    "call.full": "Allocate bounded object-array storage with public positional shape and an admitted fill value, using explicit object dtype.",
    "call.flatten": "Flatten admitted array storage in the supported order. Object-array cells remain whole Expr objects; this does not flatten encrypted slots.",
    "call.reshape": "Reshape admitted public or object-array storage with a compatible public shape. Element count is preserved; ciphertext packing is unchanged.",
    "call.transpose": "Transpose object-array storage with admitted public axes. This rearranges whole cells and does not rotate ciphertext slots.",
    "call.dict": "Construct an admitted construction-time dictionary using supported public keys and binding forms; its values remain subject to the existing type rules.",
    "call.enumerate": "Enumerate a bounded public iterable, producing public index/value pairs.",
    "call.get": "Call the admitted public dictionary get operation and consume the returned value.",
    "call.items": "Obtain and consume an admitted public dictionary items view.",
    "call.keys": "Obtain and consume an admitted public dictionary keys view.",
    "call.values": "Obtain and consume an admitted public dictionary values view.",
    "call.iter": "Create an iterator over an admitted bounded public iterable.",
    "call.next": "Advance an admitted public iterator within its bounded lifetime and consume the yielded value.",
    "call.join": "Call join on a public string receiver with an admitted sequence of public strings.",
    "call.len": "Read the public length of an admitted public sequence/container or object storage; it is not a ciphertext slot-count query.",
    "call.list": "Construct an admitted list from a bounded iterable; do not iterate a ciphertext Expr.",
    "call.tuple": "Construct an admitted tuple from a bounded iterable; do not iterate a ciphertext Expr.",
    "call.int": "Convert an admitted public value to an integer using the supported conversion rules. Ciphertext decryption/coercion is forbidden.",
    "call.float": "Convert an admitted public value to float using the supported conversion rules. Ciphertext decryption/coercion is forbidden.",
    "call.pow": "Apply the admitted public pow operation with bounded public arguments, not arbitrary encrypted exponentiation.",
    "call.pop": "Remove an admitted dictionary entry and consume its returned value; respect the supported public argument binding.",
    "call.popitem": "Remove and consume an item pair from an admitted nonempty public dictionary.",
    "call.setdefault": "Use an admitted public dictionary setdefault operation and consume its result. Mutating unrelated storage without result influence does not qualify.",
    "call.sorted": "Sort an admitted bounded public iterable using comparable public keys. Symbolic cipher/plain Expr values cannot be sort keys.",
    "call.reversed": "Reverse iteration over an admitted bounded public sequence.",
    "call.zip": "Zip admitted bounded public iterables and consume the resulting tuple elements.",
    "call.partition": "Call partition on a public string receiver with a valid public separator, returning the before/separator/after tuple. This is not an array partition operation.",
    "call.rpartition": "Call rpartition on a public string receiver with a valid public separator, splitting at the last occurrence.",
    "call.split": "Call split on a public string receiver with admitted public arguments.",
    "call.rsplit": "Call rsplit on a public string receiver with admitted public arguments.",
    "call.strip": "Strip admitted leading/trailing characters from a public string receiver.",
    "call.lstrip": "Strip admitted leading characters from a public string receiver.",
    "call.rstrip": "Strip admitted trailing characters from a public string receiver.",
    "call.replace": "Replace text in a public string using admitted public string arguments.",
    "index.read": "Read an admitted public container/array using a valid public index. Do not index ciphertext slots through Python subscripting.",
    "slice.read": "Read an admitted container/storage slice using bounded public start/stop/step. A zero step is invalid.",
    "node.ListComp": "Execute a bounded public list comprehension; its produced values must contribute to a named output.",
    "node.DictComp": "Execute a bounded public dictionary comprehension with admitted keys; its produced values must contribute to a named output.",
}

def _description(module, feature):
    matches = sorted({instruction for value, instruction in module.SPECS.values() if value == feature})
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous child semantics: " + feature)
    original = matches[0]
    generic = original.startswith("Use an executed " + feature + " expression")
    if module is public and generic and feature not in MEANINGS:
        raise ValueError("Unexplained public feature: " + feature)
    meaning = MEANINGS.get(feature) if module is public else None
    value = dict(feature=feature, registered_instruction=original)
    if meaning:
        value["meaning"] = meaning
    if module is public and feature.startswith("counter."):
        context = public.COUNTER_CONTEXTS.get(feature.split(".",1)[1])
        if context:
            value["counted_operation"] = _description(public, context)
    return value

def specification(request):
    """Explain exactly the frozen directed requirements; free tasks have none."""
    exercise = request.get("construction_exercise")
    if exercise is None:
        return []
    profile = request.get("construction_profile")
    if profile not in (None, "hecate-unified-public-v1"):
        raise ValueError("Unreviewed construction profile")
    module = public if profile else native
    if type(exercise) is not dict or module.spec(exercise.get("id")) != exercise:
        raise ValueError("Changed directed exercise")
    return [_description(module, feature) for feature in exercise["required_features"]]
