# Public value FX draft

Agent-owned prototype for authenticated public reshape/view, permute, static slice and Boolean-to-float casts. It folds only registered public expressions into separately bound buffers; encrypted operators stay in the graph.

The full original public state and input graph remain bound. A retained strict v22 trace checks the normalized trusted fixtures. The mathematical evaluator currently limits transformed expressions to 4,096 elements; unchanged weights use the selected existing resource profile.

Run after the paid batch releases its frozen source and Nix launcher lock:

python3 -B scripts/baseline/benchmarks/tools/public_value_fx_draft_v1/check.py

This developer draft is not a production intake policy. The tests do not compile, execute FHE, call a Provider, or accept arbitrary uploaded source.

Latest controls-v2: 21 passed, 0 failures/errors/skips; 12 contexts / 48 math groups / 12 retained strict traces. First fixture failure and original files are retained. See docs/baseline/agent-public-value-fx-draft-r1367.md.
