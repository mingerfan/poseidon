"""Pure planning for exact-source replay; never imports a Provider or backend."""
import copy,hashlib,json

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def require(ok,code):
 if not ok:raise ValueError(code)
def aliases(proposal):
 require(type(proposal)is dict and proposal.get("binding")==digest({k:v for k,v in proposal.items()if k!="binding"}),
  "replay_original_proposal_binding")
 require(proposal.get("format")=="poseidon-normalized-source-live-proposal-v1"and
  proposal.get("status")=="frozen_plan","replay_original_proposal_format")
 sources=proposal["source_contexts"];cases=proposal["cases"];groups=proposal["request_groups"]
 require(len(sources)==51 and len(cases)==len(groups)==26,"replay_source_denominator")
 sm={s["context"]:s for s in sources};cm={c["id"]:c for c in cases}
 require(len(sm)==51 and len(cm)==26,"replay_unique_source_and_case")
 require([g["representative_case"]for g in groups]==[c["id"]for c in cases],"replay_group_order")
 mapped=[];out=[]
 for g in groups:
  rep=g["representative_context"];case=cm[g["representative_case"]]
  require(rep in g["contexts"]and rep.replace("/","__")==case["id"],"replay_group_representative")
  require(case["request_id"]==g["request_id"]and case["model_sha256"]==sm[rep]["model_sha256"],
   "replay_representative_request")
  for context in g["contexts"]:
   require(context in sm,"replay_known_source")
   entry=sm[context];mapped.append(context)
   require(entry["request_id"]==g["request_id"]and
    entry["public_request_sha256"]==g["public_request_sha256"]and
    entry["model_sha256"]==case["model_sha256"],"replay_exact_public_request")
   if context==rep:continue
   out.append(dict(context=context,id=context.replace("/","__"),representative_case=case["id"],
    request_id=entry["request_id"],model_sha256=entry["model_sha256"],
    mathematical_request_sha256=entry["public_request_sha256"],source=copy.deepcopy(entry)))
 require(len(mapped)==len(set(mapped))==51 and set(mapped)==set(sm)and
  len(out)==len({v["id"]for v in out})==25,"replay_complete_partition")
 return out

def availability(proposal,report):
 require(type(report)is dict and report.get("proposal_binding")==proposal["binding"],"replay_current_report_binding")
 rows=report.get("rows",[])
 require(type(rows)is list and len({r["case"]for r in rows})==len(rows),"replay_origin_row_uniqueness")
 bycase={r["case"]:r for r in rows}
 require(set(bycase)<={c["id"]for c in proposal["cases"]},"replay_origin_row_scope")
 out=[]
 for item in aliases(proposal):
  origin=bycase.get(item["representative_case"])
  if origin is None:status="waiting_for_representative"
  elif origin.get("status")=="passed":
   require(origin.get("new_agent_generation")is True and origin.get("request_id")==item["request_id"]and
    type(origin.get("package_binding"))is str and len(origin["package_binding"])==64,
    "replay_genuine_passed_origin")
   status="ready_for_origin_receipt_verification"
  elif origin.get("status")=="failed":status="blocked_origin_failure"
  else:raise ValueError("replay_origin_terminal_status")
  out.append(dict(id=item["id"],representative_case=item["representative_case"],status=status))
 return out
