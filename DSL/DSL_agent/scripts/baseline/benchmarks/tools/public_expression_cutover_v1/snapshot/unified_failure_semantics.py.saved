"""Opt-in typed construction guidance. Contains no model solutions or private data."""
VERSION = "explicit-v5"

def specification():
    return {
        "typed_values": [
            "Distinguish public numbers/numeric arrays, object arrays of complete Expr cells, and packed slots inside a ciphertext. Container reshape/transpose does not move ciphertext slots.",
            "Use dtype=object for arrays containing Expr or Empty. Numeric np.array/asarray accepts public numeric data only. float/int/item never decrypt a ciphertext.",
            "Plain encoding accepts a scalar or a one-dimensional vector of length 1 or P. Explicitly reshape public arrays to that encoding shape; matrix shape alone is not a packed Plain.",
            "np.empty takes positional shape and dtype=object; np.full takes positional shape and fill plus dtype=object. Initialize empty cells before arithmetic. Only the request's admitted methods and signatures are available.",
            "Unary plus on an object array and unary plus on an individual ciphertext are different operations. Do not infer arbitrary NumPy/Expr methods from Python familiarity.",
            "At the pinned frontend, Empty - Expr resolves to Expr, not -Expr. Empty is a symbolic construction identity, not encrypted zero; use zero_ct for a client-encrypted numeric zero.",
        ],
        "contribution": [
            "Integrate the required construction into the correct computation: public coefficients, indices, control, storage or expression assembly. Do not add a bias merely to create an observable witness.",
            "The required operation must execute and influence a named output under bounded intervention. Dead code, cancelled values, unused cells and observations confined to padding do not qualify.",
            "Use the actual typed operation. A custom helper named update/items/reshape does not count as the dictionary or array method with that name.",
            "For list/tuple materialization, consume the resulting container in a way that depends on its contents. Syntax occurrence alone is insufficient.",
            "Counter requirements describe an executed operation at a source span, not a numeric counter that the model may add to its answer.",
        ],
        "packing": [
            "After adding constants or evaluating a polynomial, padded slots may become nonzero. Apply the logical validity mask before concatenation, cross-block motion or reduction where padding could influence a named output.",
            "A positive rotate(k) reads source slot j+k modulo P. Derive each destination from the logical layout; compose only provisioned rotations.",
            "Preserve Linear weights, bias, row/column order and output selectors exactly. Construction requirements do not change the mathematical model.",
        ],
        "backend": [
            "Never form encrypted zero using ciphertext times an all-zero Plain or exact self-cancellation. Preserve zero_ct and the SEAL transparent-ciphertext check.",
            "The compiler manages levels, rescale, modswitch and relinearization. A depth/scale capacity rejection is not permission to change safety parameters or model semantics.",
        ],
    }

def repair_hint(diagnostic):
    """Fixed category hints only; never echo arbitrary provider/tool text."""
    if type(diagnostic) is not str:
        return ""
    rules=specification()
    if "Missing contributing" in diagnostic or "observable object storage" in diagnostic:
        return " ".join(rules["contribution"])
    if any(x in diagnostic.lower() for x in ("numeric", "object", "plain", "dtype", "attribute", "operand", "type")):
        return " ".join(rules["typed_values"])
    return ""
