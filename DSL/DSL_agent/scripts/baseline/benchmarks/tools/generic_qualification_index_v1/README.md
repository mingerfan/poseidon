# Generic qualification index

Development tooling for current source constraints and historical qualification evidence. Reads only declared files and AST/JSON; no model imports, provider, compiler or encrypted execution. Historical observations retain their original identity.

The pinned manifest selects 34 operators, 110 reference contexts and 45 historical Poseidon CPU rule cases. Each original candidate, request, artifact and numerical comparison is checked. The index distinguishes current registration from historical acceptance.

~~~sh
python3 -B scripts/baseline/benchmarks/tools/generic_qualification_index_v1/test_audit.py
python3 -B scripts/baseline/benchmarks/tools/generic_qualification_index_v1/audit.py
python3 -B scripts/baseline/benchmarks/tools/generic_qualification_index_v1/audit.py \
  --verify-index /home/lhohy/Code/Poseidon/docs/baseline/agent-generic-qualification-index-r1378.json
~~~

`--output` creates a new file exclusively inside `docs/baseline`. Published snapshots bind the tool, manifest and declared evidence bytes. Existing snapshots are verified, not rewritten. Changes in source or evidence make old snapshots stale.

The current index is a selected development audit, not the application's capability authority. Future application qualification data should be explicit, versioned data loaded by the host interface; normal requests should not scan benchmark reports.

Original audit errors and control logs are retained beside this tool. Tests use only their own synthetic temporary files and do not validate current encrypted execution.

The historical numeric-link supplement checks retained decoding and frozen NPZ reference files without executing a model. Its explicit supplemental files receive fresh hashes; the original index remains unchanged.
