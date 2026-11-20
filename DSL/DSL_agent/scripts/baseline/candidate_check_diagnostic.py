"""Typed static failures raised only at the authoritative Agent check."""
import ast

MESSAGES = {
    "native_ast_node_budget": "Native candidate AST node budget exceeded",
    "native_expanded_ast_node_budget": "Native expanded AST node budget exceeded",
    "construction_emitted_operation_budget": "Expanded ciphertext operation budget exceeded",
    "construction_call_not_supported": "Unsupported construction call",
    "construction_attribute_not_supported": "Only permitted method/decorator attributes are allowed",
    "construction_syntax_not_supported": "Unsupported construction syntax",
    "undefined_value": "Undefined value",
    "rotation_step_not_allowed": "Unsupported rotation step",
    "native_array_constructor": "Native arrays require exactly np.array(data, dtype=object)",
    "native_signature_mismatch": "c/p signature mismatch",
    "expanded_work_budget": "Unified candidate work budget",
    "native_reserved_binding": "Native loops cannot overwrite reserved bindings",
    "native_control_flow_not_enabled": "Unsupported native loop syntax",
    "native_public_syntax": "Unsupported native loop syntax",
    "native_function_name_invalid": "Invalid or colliding function name",
}

class CandidateCheckError(ValueError):
    def __init__(self, reason, node, *, context=None):
        if reason not in MESSAGES or not isinstance(node, ast.AST):
            raise ValueError("Candidate check diagnostic")
        if reason == "native_ast_node_budget" or (
                reason == "native_expanded_ast_node_budget" and type(node) is ast.Module):
            if (type(node) is not ast.Module or type(context) is not dict or
                    set(context) != {"observed", "limit"} or
                    type(context["limit"]) is not int or context["limit"] != 4096 or
                    type(context["observed"]) is not int or not 4096 < context["observed"] <= 2**53):
                raise ValueError("Candidate AST budget witness")
            self.context = dict(context)
            self.reason = reason
            self.location = dict(kind="candidate")
            super().__init__(MESSAGES[reason])
            return
        line, column = getattr(node,"lineno",None), getattr(node,"col_offset",None)
        if not (type(line) is int and type(column) is int and
                1 <= line <= 65536 and 0 <= column <= 65536):
            raise ValueError("Candidate check location")
        fields = {"native_signature_mismatch": {"expected", "actual"},
                  "native_expanded_ast_node_budget": {"observed", "limit"},
                  "expanded_work_budget": {"observed", "limit"},
                  "construction_emitted_operation_budget": {"observed", "limit"}}.get(reason, set())
        context = {} if context is None else context
        if (type(context) is not dict or set(context) != fields or
                any(type(v) is not int or not 0 <= v <= 2**53 for v in context.values())):
            raise ValueError("Candidate check context")
        if reason == "expanded_work_budget" and not (
                context["limit"] == 1024 and context["observed"] > context["limit"]):
            raise ValueError("Candidate work witness")
        if reason == "native_expanded_ast_node_budget" and not (
                type(node) in (ast.Assign,ast.AugAssign,ast.Expr,ast.Return) and
                context["limit"] == 4096 and 4096 < context["observed"] <= 2**53):
            raise ValueError("Candidate expanded AST budget witness")
        if reason == "construction_emitted_operation_budget" and context != {"observed": 257, "limit": 256}:
            raise ValueError("Candidate emitted work witness")
        self.context=dict(context)
        self.reason=reason
        self.location=dict(kind="source",line=line,column=column)
        super().__init__(MESSAGES[reason])
