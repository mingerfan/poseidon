"""Task metrics for original outputs; divergent samples count as incorrect.

The three-argument interface accepts arrays, or an internal output record with
``values``, ``task`` and (for language models) ``lengths``. User inputs never
need a task description. Labelled classification preserves the frozen order
of floating-point operations and its original outward field names.
"""

import numpy as np


def compare(exact_outputs, approx_outputs, labels_or_None):
    record = exact_outputs if isinstance(exact_outputs, dict) else {'values': exact_outputs}
    exact = np.asarray(record['values'])
    approx = np.asarray(approx_outputs)
    if exact.shape != approx.shape or exact.ndim < 1 or not len(exact):
        raise ValueError(f'Metric output shapes differ or are empty: {exact.shape}, {approx.shape}')
    if not np.isfinite(exact).all():
        raise ValueError('Exact model outputs contain nonfinite values')
    task = record.get('task', 'classification' if exact.ndim == 2 and exact.shape[1] >= 2 else 'regression')
    total = len(exact)
    valid = np.isfinite(approx).reshape(total, -1).all(axis=1)
    if task == 'causal_lm':
        lengths = np.asarray(record['lengths'], dtype=np.int64)
        if lengths.shape != (total,) or np.any(lengths < 1) or np.any(lengths > exact.shape[1]):
            raise ValueError('LM metrics require each original nonpadding sequence length')
        exact = exact[np.arange(total), lengths-1]
        approx = approx[np.arange(total), lengths-1]
    if task in ('classification', 'causal_lm'):
        reference = exact.argmax(axis=1)
        predictions = np.where(valid, approx.argmax(axis=1), -1)
        agree = int(np.sum((predictions == reference) & valid))
        result = dict(agree=agree, total=total)
        if task == 'classification' and labels_or_None is not None:
            labels = np.asarray(labels_or_None)
            correct = int(np.sum((predictions == labels) & valid))
            top1 = 100.0*correct/total
            exact_top1 = float(100*np.mean(reference == labels))
            result.update(top1=top1, exact_top1=exact_top1, acc_drop_pp=exact_top1-top1)
        else:
            result['agreement_pct'] = 100.0*agree/total
            if task == 'causal_lm':
                result['logit_rel_err'] = _relative(exact, approx, valid)
                if labels_or_None is not None:
                    result['true_top1'] = 100.0*int(np.sum((predictions == np.asarray(labels_or_None)) & valid))/total
        return result
    return dict(total=total, relative_error=_relative(exact, approx, valid))


def _relative(exact, approx, valid):
    if not valid.all():
        return None
    numerator = float(np.max(np.abs(exact-approx)))
    denominator = float(np.max(np.abs(exact)))
    return numerator/denominator if denominator else (0.0 if numerator == 0 else None)


def passes(result, threshold):
    for key, limit in threshold.items():
        value = result.get(key)
        if value is None or not np.isfinite(value):
            return False
        if value < limit if key == 'agreement_pct' else value > limit:
            return False
    return True
