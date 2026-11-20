"""Exact-count Top1 limits: signed loss and absolute accuracy difference."""
from fractions import Fraction
import math

TOP1_KINDS = ("top1_accuracy_drop", "top1_accuracy_difference")

def limit_field(kind):
    if kind not in TOP1_KINDS:
        raise ValueError("Unknown Top1 metric")
    return "max_drop" if kind == "top1_accuracy_drop" else "max_difference"

def threshold(kind, original_correct, compared_correct, total, limit):
    field = limit_field(kind)
    if (type(total) is not int or total <= 0 or
        any(type(x) is not int or not 0 <= x <= total
            for x in (original_correct, compared_correct)) or
        type(limit) not in (int, float) or not math.isfinite(limit) or not 0 <= limit <= 1):
        raise ValueError("Invalid Top1 counts or threshold")
    delta = Fraction(original_correct - compared_correct, total)
    measure = "accuracy_drop" if kind == "top1_accuracy_drop" else "accuracy_difference"
    value = delta if kind == "top1_accuracy_drop" else abs(delta)
    return {measure: float(value), field: limit,
            "passed": value <= Fraction(str(limit))}
