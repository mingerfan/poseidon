"""Select observed quality fields from the model's existing task metric."""
import math


def model_metric(info):
    return getattr(info, 'metric', None) or ('regression_relative' if getattr(info, 'task', None) == 'regression' else None)


def is_regression(metric=None, result=None):
    return metric == 'regression_relative' or (metric is None and result is not None
        and 'relative_error' in result)


def quality_fields(result, metric=None):
    key = 'relative_error' if is_regression(metric, result) else 'agree'
    return {key: result.get(key)}


def quality_matches(reference, observed, metric=None):
    """Classification counts are exact; relative errors use the math policy."""
    key = 'relative_error' if is_regression(metric, reference) else 'agree'
    left, right = reference[key], observed[key]
    if key == 'agree':
        return left == right
    if left is None or right is None:
        return left is None and right is None and not reference['feasible'] and not observed['feasible']
    return (math.isfinite(left) and math.isfinite(right)
        and abs(right-left) <= 1e-12 + 1e-12*abs(left))


def strict_tuner_quality(result, max_acc_drop_pp):
    if is_regression(result=result):
        value = result['relative_error']
        # Validation's feasible flag already includes the model's task limit.
        return result['feasible'] and value is not None and math.isfinite(value)
    return result['feasible'] and result['acc_drop_pp'] <= max_acc_drop_pp
