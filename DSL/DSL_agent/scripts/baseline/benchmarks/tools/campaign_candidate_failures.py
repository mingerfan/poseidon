"""Recognize a narrow, terminal candidate rejection; never convert failure to pass."""
from pathlib import Path
import hashlib
from stage2_agent_campaign import ROOT
from benchmark_runner import strict_file
from stage2_agent_provenance import provider_accounting
MESSAGE="terminate called after throwing an instance of 'std::logic_error'\n  what():  result ciphertext is transparent\n"

def transparent_failure(folder,report,paid):
 if report.get("status")!="repair_budget_exhausted":return None
 attempts=report.get("attempts",[])
 if not attempts:return None
 last=attempts[-1]
 if (last.get("failure_layer")!="seal_runtime" or last.get("status")!="failed" or not last.get("compiled")
     or last.get("executed") or last.get("diagnostic")!="Isolated SEAL execution failed (exit 134); see execute.log"
     or last.get("trace",{}).get("frontend")!="real_Hecate"):return None
 accounting=provider_accounting(report,paid)
 if len(accounting["received"])!=len(attempts):raise ValueError("Uncertain provider responses")
 if not strict_file(folder/"key-cleanup-outcome.json",131072).get("complete"):raise ValueError("Incomplete key cleanup")
 log=folder/("attempt-%02d"%last["index"])/"execute.log"
 if log.is_symlink() or not log.is_file() or log.stat().st_size>512:raise ValueError("Unsafe runtime diagnostic")
 raw=log.read_bytes()
 if raw.decode("utf-8")!=MESSAGE:return None
 hashes=last.get("artifact_hashes")
 if not hashes:raise ValueError("Missing compiled artifacts")
 output=log.parent/"output"
 for name,hsh in hashes.items():
  path=output/name
  if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=hsh:
   raise ValueError("Changed compiled artifact")
 return dict(kind="candidate_transparent_ciphertext",execute_log_sha256=hashlib.sha256(raw).hexdigest(),
  key_cleanup_sha256=hashlib.sha256((folder/"key-cleanup-outcome.json").read_bytes()).hexdigest(),
  remains_failed=True,automatic_candidate_retry=False)
