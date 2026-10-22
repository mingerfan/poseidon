"""Provider-neutral validation of the existing bounded public feedback envelope."""
import copy
import math
from candidate_contract import canonical


class FeedbackError(ValueError):
    """Stable public diagnostic code."""


def check(condition, code):
    if not condition:
        raise FeedbackError(code)


def public_feedback(feedback):
    fields = {"status", "layer", "category", "diagnostic", "numerical_summary", "tool_diagnostic"}
    check(type(feedback) is dict and set(feedback) <= fields and feedback.get("status") == "failed",
          "unapproved_feedback_fields")
    for name in ("layer", "category", "diagnostic"):
        value = feedback.get(name)
        check(value is None or (type(value) is str and len(value.encode()) <= 8000), "feedback_text_limit")
    if "tool_diagnostic" in feedback:
        tool = feedback["tool_diagnostic"]
        check(type(tool) is dict and set(tool) == {"trust", "text"}
              and tool["trust"] == "untrusted_tool_output" and type(tool["text"]) is str
              and len(tool["text"].encode()) <= 16000, "tool_diagnostic_limit")
    if "numerical_summary" in feedback:
        summary = feedback["numerical_summary"]
        check(type(summary) is dict and set(summary) <= {"mae", "max_absolute_error", "compared_values", "passed"},
              "unapproved_numerical_fields")
        for name, value in summary.items():
            if name == "passed":
                check(type(value) is bool, "invalid_numerical_summary")
            else:
                check(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                      "invalid_numerical_summary")
    check(len(canonical(feedback)) <= 32768, "feedback_size_limit")
    return copy.deepcopy(feedback)
