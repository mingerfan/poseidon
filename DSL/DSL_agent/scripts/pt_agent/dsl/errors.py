"""Stable R1 parse and static-check diagnostics."""

from dataclasses import dataclass


PARSE_CODES = frozenset({
    "E_SYNTAX", "E_NO_ENTRY", "E_MULTI_ENTRY", "E_HEADER", "E_PARAM", "E_STMT", "E_EXPR",
    "E_UNKNOWN_OP", "E_UNDEFINED", "E_ARITY", "E_ATTR_UNKNOWN", "E_ATTR_MISSING",
    "E_ATTR_TYPE", "E_ATTR_VALUE", "E_TARGET", "E_RETURN", "E_MACRO", "E_REGION",
    "E_NAME", "E_WEIGHT_REF",
})
CHECK_CODES = frozenset({
    "E_SHAPE", "E_WEIGHT_MISSING", "E_WEIGHT_SHAPE", "E_LAYOUT_UNPACKED",
    "E_LAYOUT_REPACK", "E_LAYOUT_MISMATCH", "E_KIND", "W_UNUSED", "W_REFRESH_COMPILER",
})


class PTDSLError(Exception):
    def __init__(self, code: str, lineno: int, msg: str):
        assert code in PARSE_CODES
        self.code, self.lineno, self.msg = code, lineno, msg
        super().__init__(f"{code} line {lineno}: {msg}")


@dataclass(frozen=True)
class Diag:
    code: str
    severity: str
    lineno: int
    op_id: int | None
    msg: str
    call_lines: tuple[int, ...] = ()
    macro_stack: tuple[str, ...] = ()

    def __str__(self):
        location = f"line {self.lineno}"
        if self.call_lines:
            location += f" (in macro {self.macro_stack[-1]} called at line {self.call_lines[-1]})"
        return f"{self.severity.upper()} {self.code} {location}: {self.msg}"
