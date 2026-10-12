"""Audit ModRaise integer-period semantics absent from the Q/scale ledger.

Uses the exact saved Q50 primes and original source EvalMod polynomial. This
is a numerical counterexample tool, not a security estimator or HE execution.
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def polynomial(report, x):
    # EvalModPoly: total K=25, DA=2, interval [-25/4,25/4].
    x -= 0.5/(4*(2*25/4))
    coeffs = [complex(*c) for c in report['bootstrap']['fold']['original_coefficients']]
    a,b = 1.0,x
    out = coeffs[0]+coeffs[1]*b
    for c in coeffs[2:]:
        a,b = b,2*x*b-a
        out += c*b
    for c in report['bootstrap']['fold']['original_constants']:
        out = 2*out*out-c
    return 32*out.real


def audit(report):
    q0 = report['q_bottom_first'][0]*report['q_bottom_first'][1]
    q_diff = q0/2**round(math.log2(q0))
    prepared = 2**report['preparation']['prepared_scale']
    logical = 2**report['preparation']['c2s_in']
    # Exact existing source coefficient: q_div/(K*q_diff) = 2^45/(K*Q0).
    c2s_coefficient = 2**45/(25*q0)
    wrap_step = q0*c2s_coefficient/logical
    input_gain = prepared*c2s_coefficient/logical
    cases = []
    for message in (0.0, 0.05, 0.5):
        for integer_wrap in (0,1,2,5,10,15):
            x = message*input_gain+integer_wrap*wrap_step
            ideal_sine = 32*q_diff/(2*math.pi)*math.sin(2*math.pi*25*x)
            fitted = polynomial(report,x)
            cases.append(dict(input=message,integer_wrap=integer_wrap,c2s_input=x,
                ideal_modular_sine=ideal_sine,degree59_output=fitted,
                absolute_error=abs(fitted-message)))
    # A counterfactual period check only: NOT an implemented or approved fix.
    # Setting logical C2S scale to 2^45 makes one Q0 lift exactly one sine
    # period, but the message gain must then be jointly compensated as well.
    period_aligned_logical = 2**45
    message_gain = 32*q_diff*prepared/q0
    return dict(scope='PUBLIC_MODRAISE_PERIOD_AUDIT',
        q0_product=q0,q_diff=q_diff,prepared_log_scale=math.log2(prepared),
        c2s_logical_log_scale=math.log2(logical),
        normalized_step_per_integer_wrap=wrap_step,
        sine_period=1/25,periods_per_integer_wrap=25*wrap_step,
        period_error_fraction=25*wrap_step-1,
        period_alignment_pass=abs(25*wrap_step-1)<1e-10,
        cases=cases,
        diagnostic_counterfactual=dict(c2s_logical_log_scale=45,
            periods_per_integer_wrap=25*q0*c2s_coefficient/period_aligned_logical,
            remaining_small_signal_message_gain=message_gain,
            precision_validated=False,production_modified=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--witness',type=Path,default=HERE/'q50.json')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=audit(json.loads(args.witness.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
