# Approximation error is not CKKS execution error

Date: 2026-09-07. This is an explicit diagnostic fixture, not automatic activation
rewriting and not a recommendation for a production approximation.

## Three outputs, computed from the same original input

Let f(x) be the original plaintext activation and p(x) the explicitly selected
polynomial. Let y be the decrypted result of evaluating the candidate program.

- Model approximation error: p(x) - f(x).
- Execution residual relative to the polynomial: y - p(x).
- Total error relative to the original: y - f(x).

Signed errors satisfy y-f = (p-f) + (y-p). Absolute errors do **not** generally
add: the components can cancel. The report retains signed errors, absolute
errors, MAE, nonzero-reference relative errors, cosine similarity when defined,
and the decomposition residual. Zero reference values receive null relative
errors, not invented zero errors.

Only after establishing that the candidate implements p may its small execution
residual be interpreted as CKKS numerical error. A wrong program, compiler bug
or runtime bug can also make y-p large. The report explicitly warns against
attributing every execution residual to CKKS noise.

The generic reporting API is `approximation_errors.decompose(original,
polynomial, decrypted)`. Callers must independently evaluate original and
polynomial references from identical inputs. It does not infer an application's
acceptable approximation budget and never claims formal semantic equivalence.

## Deliberately coarse, fully explicit fixture

Original: ReLU(x) = max(0,x).
Executed polynomial: p(x) = (x+x*x)/2.
Declared domain: [-1,1].

With u=abs(x), p(x)-ReLU(x)=(u*u-u)/2. Thus the maximum absolute approximation
error is 1/8 = 0.125, attained at abs(x)=0.5. This is an elementary bound for this
specific scalar polynomial, not a certificate for an arbitrary network.
The polynomial is zero at x=-1 and x=0 and one at x=1; checking endpoints alone
would miss its interior approximation error.

The actual input descriptor is
`scripts/baseline/cases/relu-quadratic-explicit.json`: a schema-2 graph of square,
add and multiplication by public 0.5. It contains no disguised ReLU operator.
The original ReLU remains a separately evaluated plaintext reference.
The standard four input groups (zero, fixed signed, seeded random, boundaries)
are preserved, with four logical input elements per group.

The manual Hecate candidate is:

```python
@hc.func("c")
def golden(x):
    squared = x * x
    combined = x + squared
    result = combined * c0  # trusted constant registry: 0.5
    return result
```

A separate counterexample replaces the addition with subtraction. Both are
statically valid Hecate fragments; only the first implements the declared
polynomial. ReLU syntax itself remains rejected by the generated-program
allowlist. No upstream HE_ReLU/HE_SiLU helper was enabled.

## Actual encrypted results

Evidence: `/home/lhy/poseidon-work/results/approximation-golden-v08aa_ux`.

| Candidate | Max approximation error | Max residual vs polynomial | Max total error vs ReLU | Polynomial numerical test |
|---|---:|---:|---:|---|
| Correct manual DSL | 0.125 | 6.947190934e-9 | 0.125000003263 | Passed |
| Wrong-sign DSL | 0.125 | 1.000000000243 | 1.000000000243 | Rejected |

Both fail the original-target threshold. The original threshold and polynomial
execution threshold were not relaxed: atol=1e-5, rtol=1e-4. A passed experiment
means that the correct candidate passed and the incorrect one was detected;
it does **not** mean that the approximation was approved for an application.

Each candidate ran four actual encrypt/evaluate/decrypt batches (16 output
values) through the existing isolated Hecate/Dacapo/SEAL CPU pipeline. This is
eight encrypted batches and 32 output values across the positive and negative
cases, not eight positive results. Security parameters remain tc128, N=32768,
the existing 60-bit-prime SEAL profile, and input scale 2^40. No bootstrap,
decrypt-and-reencrypt, GPU execution, API call or secret-key exposure occurred.

Correct candidate evidence: `candidate-replay-1lw1_l6p`.
Wrong candidate evidence: `candidate-replay-z99_ybzv`.
The prior development run `approximation-golden-1t8c22f8` is preserved, not merged
into this result. The final runner archives source snapshots so later local
development does not invalidate the ability to inspect the exact experiment.
Reference/input and compiled artifact hashes are also checked.

## Reproduce and audit

Run in WSL from `/mnt/d/Code Space/Poseidon`.

```bash
# New local encrypted experiment; zero model API calls.
python3 scripts/baseline/run_approximation_golden.py

# Audit the saved result without rerunning encryption.
PYTHONDONTWRITEBYTECODE=1 \
POSEIDON_APPROXIMATION_RESULTS=/home/lhy/poseidon-work/results/approximation-golden-v08aa_ux \
python3 -m unittest discover -s scripts/baseline -p test_approximation_errors.py -v
```

Tests cover hand-calculated values, domain rejection, the interior error bound,
finite/shape/tolerance validation, static accept/reject rules, cancellation of
signed errors, actual artifacts and execution, and independent original versus
polynomial acceptance. Six tests passed with the above evidence path.

## Not yet established

- An application-approved approximation and error budget.
- Automatic rewriting of ReLU/SiLU in user models.
- End-to-end error propagation through arbitrary multi-layer networks.
- Online Agent generation of this new capability.
- Upstream bootstrap-based activation helpers or Poseidon GPU execution.

Those gaps remain explicit in the semantic inventory. This fixture does not
expand the current model operator allowlist or silently change user algorithms.
