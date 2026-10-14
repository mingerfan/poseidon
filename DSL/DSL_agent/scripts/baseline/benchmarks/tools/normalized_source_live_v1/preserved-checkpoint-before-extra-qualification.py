"""Revalidate a stopped unpaid capture; never synthesize a reference receipt."""
import copy
from pathlib import Path
from scope import read,sha,digest,require,check_binding

CONTINUATION=dict(wall_seconds=180,new_result_bytes=64*1024**2,provider_calls=0,compiler_calls=0,
 encrypted_calls=0,global_result_bytes=32*1024**3,minimum_free_bytes=4*1024**3)
FIELDS={"format","status","inventory_binding","source_stage_identity_sha256","failure_log",
 "failure_log_sha256","original_preparation_root","original_deadline_seconds","completed_contexts",
 "remaining_contexts","contexts","files","continuation","human_authority","binding"}

def validate(value,inventory,sdk,*,here,results):
 require(type(value)is dict and set(value)==FIELDS,"capture_checkpoint_fields")
 check_binding(value)
 require(value["format"]=="poseidon-source-capture-checkpoint-v1"and
  value["status"]=="stopped_at_time_boundary","capture_checkpoint_status")
 require(value["inventory_binding"]==inventory["binding"]and
  value["source_stage_identity_sha256"]==sdk["manifest_sha256"],"capture_checkpoint_identity")
 require(value["original_deadline_seconds"]==570 and value["completed_contexts"]==49 and
  value["remaining_contexts"]==2 and value["continuation"]==CONTINUATION,
  "capture_checkpoint_scope")
 require(value["human_authority"]=="docs/baseline/current-plan-test-authorization-2026-10-11.json",
  "capture_checkpoint_authority")
 here=Path(here).resolve();results=Path(results).resolve()
 log=here/"freeze-current-source-first.log"
 require(value["failure_log"]==str(log)and sha(log)==value["failure_log_sha256"]and
  "source_preparation_time_ceiling"in log.read_text(),"capture_checkpoint_original_stop")
 original=Path(value["original_preparation_root"])
 require(original==original.resolve()and original.parent==results and
  original.name.startswith("agent-normalized-source-live-prepare-r1409-")and original.is_dir(),
  "capture_checkpoint_result_root")
 require(type(value["contexts"])is list and
  [r.get("context")for r in value["contexts"]]==[r["context"]for r in inventory["contexts"][:49]],
  "capture_checkpoint_context_order")
 require(type(value["files"])is dict,"capture_checkpoint_files")
 expected=set()
 for row in value["contexts"]:
  require(type(row)is dict and set(row)=={"context","source_folder","task_directory","task_id"},
   "capture_checkpoint_row")
  name=row["context"].replace("/","__");folder=here/"requests"/name;store=original/name/"tasks"
  task=Path(row["task_directory"])
  require(row["source_folder"]==str(folder)and task.parent==store and task.name==row["task_id"]and
   task.name.isalnum()and len(task.name)==32 and task.is_dir(),"capture_checkpoint_task_path")
  require([p for p in store.iterdir()if p.is_dir()]==[task],"capture_checkpoint_task_count")
  for subtree in (folder,original/name):
   require(subtree==subtree.resolve()and subtree.is_dir(),"capture_checkpoint_subtree")
   for p in subtree.rglob("*"):
    require(not p.is_symlink(),"capture_checkpoint_symlink")
    if p.is_file():
     expected.add(str(p));require(value["files"].get(str(p))==sha(p),"capture_checkpoint_file_changed")
  state=read(task/"state.json")
  require(state["status"]=="queued"and state["phase"]=="queued"and
   state["generations"]==state["http_attempts"]==0 and state["ledger"]==[],
   "capture_checkpoint_unpaid_state")
  require(state["intake"].get("reference_verified")is True and state["intake"].get("agent_calls")==0 and
   state["intake"].get("encrypted_execution")is False,"capture_checkpoint_reference_state")
 require(set(value["files"])==expected,"capture_checkpoint_complete_files")
 return {r["context"]:copy.deepcopy(r)for r in value["contexts"]}

def load_metadata(row,sdk,source_origin):
 task=Path(row["task_directory"])
 meta=read(task/"task.json");snapshot=read(task/"source.json")
 check_binding(meta);check_binding(snapshot)
 require(meta["identity"]==sdk and meta["validation_level"]=="numerical",
  "capture_checkpoint_metadata_identity")
 require(source_origin(snapshot,meta["request"],meta.get("model_provenance"))==meta["model_source"],
  "capture_checkpoint_source_origin")
 return meta
