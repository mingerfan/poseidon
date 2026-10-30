"""Experimental lower-degree rsqrt with the established pre-square RMS layout.

Unlike earlier layout-only rounds, this changes the approximation polynomial.
The true function, domain, epsilon and output error threshold are unchanged.
Each degree must pass an independent truncation/error-budget audit first.
"""
from dataclasses import replace
from qwen24.kernels import qwen_layers as ql, qwen_nonlinear as nonlinear
from qwen24.kernels import rms_affine_candidate as base

STRATEGIES=('degree12p2','degree10p2','degree10p3','degree10p4','degree10p6')
CURRENT='degree12p2'


def rms_norm(x,weight,*,rsqrt,eps=1e-6):
    if CURRENT not in STRATEGIES:raise ValueError('Unknown degree candidate')
    degree,block=map(int,CURRENT.removeprefix('degree').split('p'))
    if not isinstance(rsqrt,nonlinear.ChebyshevApproximation) or rsqrt.function!='rsqrt':
        raise ValueError('Degree experiment requires Chebyshev rsqrt')
    if len(rsqrt.coefficients)!=13 or tuple(rsqrt.domain)!=(.5,2.):
        raise ValueError('Degree experiment is restricted to the audited degree12/domain profile')
    profile=replace(rsqrt,coefficients=tuple(rsqrt.coefficients[:degree+1]))
    return base.rms_norm(x,weight,rsqrt=profile,eps=eps,strategy='prepower'+str(block))


def install(strategy):
    global CURRENT
    if strategy not in STRATEGIES:raise ValueError('Unknown degree candidate')
    CURRENT=strategy;ql.rms_norm=rms_norm
