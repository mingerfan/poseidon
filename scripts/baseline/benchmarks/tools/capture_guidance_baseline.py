"""Read-only r97 request/candidate compatibility baseline; never calls a provider."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from semantic_benchmark_execution import runtime_sources
from benchmark_graph import digest
from unified_graph_contract import prepare,validate_candidate
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
out=R/"stage2-guidance-r99"
out.mkdir(exist_ok=False)
sources=runtime_sources()
assert digest(sources)=="3e857c649bb4c95a63a272d3ec187405da991dfa7513ced0f39431bba3b58b88"
for name in sources:
 target=out/"before"/name
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(ROOT/name,target)
rows=[]
for row in json.loads((R/"stage2-agent-pilot-r97-live/report.json").read_text())["rows"]:
 base=Path(row["evidence"])
 request=json.loads((base/"request.json").read_text())
 attempts=[]
 for path in sorted(base.glob("attempt-*/response.txt")):
  raw=path.read_bytes()
  try:
   value=validate_candidate(json.loads(raw),request)
   result={"accepted":True,"metadata_sha256":digest(value)}
  except Exception as error:
   result={"accepted":False,"error_type":type(error).__name__,"diagnostic":str(error)}
  attempts.append({"response":str(path),"response_sha256":hashlib.sha256(raw).hexdigest(),**result})
 rows.append({"id":row["id"],"request":str(base/"request.json"),"request_id":request["request_id"],"attempts":attempts})
body={"format":"poseidon-guidance-compatibility-before-v1","sources":sources,"source_digest":digest(sources),"rows":rows}
body["binding"]=digest(body)
(out/"before.json").write_text(json.dumps(body,indent=2)+"\n")
print(json.dumps({"binding":body["binding"],"requests":len(rows),"responses":sum(len(x["attempts"]) for x in rows)}))
