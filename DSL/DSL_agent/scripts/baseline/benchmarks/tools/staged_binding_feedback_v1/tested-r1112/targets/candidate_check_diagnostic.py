"""Typed static failures raised only at the authoritative Agent check."""
import ast

MESSAGES = {
    "undefined_value": "Undefined value",
    "native_array_constructor": "Native arrays require exactly np.array(data, dtype=object)",
}

class CandidateCheckError(ValueError):
    def __init__(self, reason, node):
        if reason not in MESSAGES or not isinstance(node, ast.AST):
            raise ValueError("Candidate check diagnostic")
        line, column = getattr(node,"lineno",None), getattr(node,"col_offset",None)
        if not (type(line) is int and type(column) is int and
                1 <= line <= 65536 and 0 <= column <= 65536):
            raise ValueError("Candidate check location")
        self.reason=reason
        self.location=dict(kind="source",line=line,column=column)
        super().__init__(MESSAGES[reason])
