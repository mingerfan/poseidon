"""Pure activation-configuration sampling for the frozen R9a measurements."""

from collections.abc import Mapping
from copy import deepcopy

from ..dsl.ir import Program


DEGS = (15, 31, 63, 127, 255)
MARGIN_RANGE = (1.0, 1.6)


def act_ops(program):
    """Return expanded activation operations in program order."""
    return [op for op in program.ops if op.kind == 'act']


def apply_config(base_program, calib_ranges, config) -> Program:
    """Copy the program and apply one configuration and calibration range per act."""
    program = deepcopy(base_program)
    acts = act_ops(program)
    if len(acts) != len(calib_ranges) or len(acts) != len(config):
        raise ValueError('Each activation requires one calibration range and configuration')
    for op, bounds, settings in zip(acts, calib_ranges, config):
        lo, hi = ((bounds['min'], bounds['max']) if isinstance(bounds, Mapping) else bounds)
        lo, hi = float(lo), float(hi)
        method, margin = settings['method'], settings['margin']
        if method == 'cheb':
            op.attrs.update(method='cheb', deg=settings['deg'], degrees=None,
                            domain=(lo*margin, hi*margin))
        elif method == 'minimax_comp':
            if op.attrs['fn'] != 'relu':
                raise ValueError('minimax_comp requires a relu activation')
            bound = margin*max(abs(lo), abs(hi))
            op.attrs.update(method='minimax_comp', deg=None, degrees=(15, 15, 27),
                            domain=(-bound, bound))
        else:
            raise ValueError(f'Unknown activation method: {method}')
    return program


def random_config(acts, rng) -> list[dict]:
    """Sample each activation independently using only the supplied generator."""
    result = []
    for op in acts:
        minimax = op.attrs['fn'] == 'relu' and rng.random() < 0.25
        method = 'minimax_comp' if minimax else 'cheb'
        degree = None if minimax else int(rng.choice(DEGS))
        margin = round(float(rng.uniform(*MARGIN_RANGE)), 2)
        result.append(dict(method=method, deg=degree, margin=margin))
    return result


def anchor_configs(acts) -> dict[str, list[dict]]:
    """Return the prescribed uniform Chebyshev and ReLU minimax anchors."""
    result = {}
    for degree in DEGS:
        for margin in (1.2, 1.5):
            result[f'uniform-cheb-{degree}-{margin}'] = [
                dict(method='cheb', deg=degree, margin=margin) for _ in acts]
    if any(op.attrs['fn'] == 'relu' for op in acts):
        for margin in (1.2, 1.5):
            result[f'uniform-minimax-{margin}'] = [
                dict(method='minimax_comp', deg=None, margin=margin) if op.attrs['fn'] == 'relu'
                else dict(method='cheb', deg=63, margin=margin) for op in acts]
    return result
