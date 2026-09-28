"""Explicit historical-evidence context, with a separately checked current tree."""
import hashlib,json
from pathlib import Path
from benchmark_graph import digest
from semantic_benchmark_execution import runtime_sources
ROOT=Path(__file__).resolve().parents[4]
OLD="3e857c649bb4c95a63a272d3ec187405da991dfa7513ced0f39431bba3b58b88"
CURRENT="f6493518f4bf63c257fb7eab9f9a782a89cf5cf4a16c9006f607976f2c58b69b"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(p):
 d=json.loads(p.read_text())
 if d.get("binding")!=digest({k:v for k,v in d.items() if k!="binding"}):raise ValueError("Binding changed: "+str(p))
 return d
def context():
 d=ROOT/"docs/baseline";old=bound(d/"stage2-contract-linkage-r90.json")
 original=bound(d/"stage2-offline-acceptance-r94.json")
 current=bound(d/"stage2-directed-guidance-acceptance-r146.json")
 history=bound(d/"stage2-runtime-lineage-r139.json")
 if digest(old["source_hashes"])!=OLD or original["current_runtime_sha256"]!=OLD:raise ValueError("Historical identity")
 now=runtime_sources()
 if digest(now)!=CURRENT or current["source_hashes"]!=now:raise ValueError("Current source identity")
 changed={k for k in old["source_hashes"].keys()&now.keys() if old["source_hashes"][k]!=now[k]}
 allowed={"scripts/baseline/run_candidate.py","scripts/baseline/candidate_sandbox.py","scripts/baseline/unified_graph_contract.py"}
 added={"scripts/baseline/test_unified_generation_guidance.py","scripts/baseline/test_unified_directed_guidance.py","scripts/baseline/unified_logical_semantics.py","scripts/baseline/unified_directed_semantics.py"}
 if changed!=allowed or now.keys()-old["source_hashes"].keys()!=added or old["source_hashes"].keys()-now.keys():raise ValueError("Unreviewed source change")
 for doc in (old,original,current,history):
  for p,h in doc["parents"].items():
   if sha(Path(p))!=h:raise ValueError("Parent changed: "+p)
 if current["tests"]!={"errors":0,"failed":0,"run":31,"skipped":0} or current["old_requests_identical"]!=2030 or current["scripted_encrypted_passed"]!=2:raise ValueError("Compatibility evidence")
 return dict(historical_source_hashes=old["source_hashes"],current_source_hashes=now,
  historical_runtime_sha256=OLD,current_runtime_sha256=CURRENT,
  changed_files=sorted(changed),added_files=sorted(added),
  retained_regression=current["tests"],retained_requests_identical=2030,
  retained_scripted_fhe_passes=2,old_executions_rebound=False,
  limitation="Current compatibility is finite regression evidence. Historical FHE passes keep their original source identities.")
