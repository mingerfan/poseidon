"""Opt-in explicit-v6 ABI/type guidance; no recipes, reference arrays or private data."""
VERSION = "explicit-v6"
# These are contract explanations, not solutions to the requested graph.
FEATURE_RULES = {
 "call.values": "Execute the built-in dictionary values() method and consume the returned values in the output dependency. Naming a helper values or leaving the view unused does not qualify.",
 "counter.sequence_operations": "Execute public sequence concatenation or repetition and consume its result. Creating or indexing a list alone is not this operation.",
 "counter.loop_continues": "Execute an actual continue statement in a public loop. Its skipped work must affect an output dependency; a variable named loop_continues is irrelevant.",
 "counter.default_evaluations": "Use an evaluated helper default binding that reaches an output. An explicit argument overriding an unused default does not demonstrate that dependency.",
 "counter.public_numeric_operations": "Execute arithmetic on public numeric values and use that result in construction. Ciphertext arithmetic alone does not increment this counter.",
 "public_cells": "The object-array unary operation must act on public numeric cells. Ciphertext cells and encoded Plain cells do not satisfy public_cells.",
 "fresh.USub": "Observe both the new unary-minus result and the original object storage (or its alias). If the original storage is never observed, fresh allocation cannot be distinguished from mutation.",
 "fresh.UAdd": "Observe both the new unary-plus result and the original object storage (or its alias). Merely returning the new result cannot establish copy independence.",
 "overlap": "Execute an object-array in-place augmented update whose two views overlap in storage. Slice assignment or out-of-place arithmetic on views does not meet this requirement. Observe the affected cells.",
 "item.numeric.positional": "Call numeric-array item with multiple positional indices, matching its rank. A single linear index into a one-dimensional array does not qualify.",
 "empty_left": "Exercise Empty cells on the left of object-array arithmetic. A scalar Empty operation alone is not an object-array event. Empty is a symbolic identity, not numerical zero.",
 "aug.Mult": "Execute the augmented assignment *= on an admitted target, and consume the changed value. A normal multiplication or an unused mutation does not qualify.",
}
TYPE_RULES = {
 "names": "public_constants is request metadata, not a bound runtime dictionary. Only the listed constant keys are prebound. Model-local constant names are not automatically runtime names; map their values through constant_origins. Do not invent pN/cN names.",
 "dtype": "For public numeric np.array/asarray use the admitted float64 dtype spelling (np.float64, np.double, or the literal string float64). Bare dtype=float is not a supported value expression. float(value) is a separate public conversion call. Expr-containing arrays require dtype=object.",
 "shape": "Numeric public arrays must be rectangular. Scalar constants and vector masks have different ranks; inspect the declared shapes before combining them. A Plain operand encodes only a scalar or a 1D vector of length 1 or P; explicitly reshape compatible numeric storage first.",
 "unary": "Unary plus is supported on admitted object-array cells, not on an individual ciphertext Expr. Empty/None cells cannot be used by object-array unary operations. Initialize cells and distinguish public numbers from Plain and ciphertext cells.",
 "compare": "Ordering and equality require the admitted public scalar/sequence types. Do not compare ciphertexts, arrays, functions or iterators as if they were public numbers. Dictionary key and comparison rules remain restricted.",
 "returns": "Return the exact number of complete ciphertext expressions in output order. A helper may internally return containers or mixed values, but those are not additional encrypted outputs. Select its admitted ciphertext result before forming golden's flat return.",
 "broadcast": "Object-array broadcasting is distinct from symbolic scalar arithmetic. A symbolic Expr/Empty left operand does not automatically dispatch object-array broadcasting. Use the admitted array operand forms and compatible ranks.",
 "calls": "Only registered call signatures and attributes are available. Python/NumPy familiarity is not an allowlist. Starred argument expansion must be in an admitted call form, not an arbitrary expression or comprehension.",
}
def specification(request):
    period=request["layout"]["input_slot_period"]
    constants=[]
    for name,value in sorted(request["public_constants"].items()):
        constants.append({"name":name,"type":"prebound_public_plain","shape":[len(value)] if type(value) is list else [],"immutable":True})
    exercise=request.get("construction_exercise",{})
    features=exercise.get("required_features",[])
    details=[]
    for feature in features:
        message=FEATURE_RULES.get(feature)
        if message:details.append({"feature":feature,"acceptance":message})
    return {
      "constant_bindings":constants,
      "type_rules":dict(TYPE_RULES),
      "slot_period":period,
      "rotation_steps":[1<<i for i in range(period.bit_length()-1)],
      "output_ciphertexts":request["layout"]["output_ciphertexts"],
      "directed_acceptance":details,
      "scope":"Public ABI and registered types only. No model implementation, private test values or deterministic DSL answer. Existing validator, contribution probes, tracing and numerical gates remain authoritative.",
    }
def repair_hint(request,diagnostic):
    """Fixed category text; never echo an untrusted diagnostic or source."""
    if type(diagnostic) is not str:return ""
    keys=[]
    patterns={
      "names":("Undefined or unbound",),
      "dtype":("value: float","dtype","numeric np.array","np.array/asarray"),
      "shape":("Ragged","Encoding needs","shape","reshape"),
      "unary":("unary positive","Unary object"),
      "compare":("Ordering requires","Equality requires"),
      "returns":("outputs mismatch",),
      "broadcast":("object-array broadcasting",),
      "calls":("Unsupported construction call","Only permitted","Starred"),
    }
    for key,values in patterns.items():
        if any(v in diagnostic for v in values):keys.append(key)
    parts=[TYPE_RULES[k] for k in keys]
    if "Missing contributing" in diagnostic or "observable object storage" in diagnostic:
        spec=specification(request)
        parts.extend(item["acceptance"] for item in spec["directed_acceptance"])
        parts.append("The required typed operation must execute and influence a named output under bounded intervention. Preserve the original mathematical result. Dead, cancelled or padding-only work is not evidence.")
    if "Rotation requires" in diagnostic:
        parts.append("Rotate a ciphertext receiver using a provisioned positive power-of-two step below the declared period; compose admitted steps for other displacements.")
    return " ".join(parts)[:6000]
