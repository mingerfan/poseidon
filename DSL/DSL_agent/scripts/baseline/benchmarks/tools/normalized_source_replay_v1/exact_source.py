"""Immutable prior-Agent source injection; no transport and no Agent attribution."""
from candidate_contract import request_file_bytes,strict_json
from generation_receipt import verify
from benchmark_graph import require
class ExactAgentReplay:
    kind="scripted_replay"
    agent_calls=0
    def __init__(self,request,raw,receipt,*,generation_identity_sha256):
        answer=strict_json(raw)
        require(type(answer)is dict and set(answer)=={"schema","request_id","hecate_source"}and
                type(answer["schema"])is int and answer["schema"]==1 and
                answer["request_id"]==request["request_id"]and type(answer["hecate_source"])is str,
                "replay_response_fields")
        self.origin=verify(receipt,request,answer["hecate_source"],
                           generation_identity_sha256=generation_identity_sha256)
        self._request=request_file_bytes(request);self._raw=raw
        self.calls=0
    def generate(self,request,*args,**kwargs):
        require(request_file_bytes(request)==self._request,"replay_exact_mathematical_request")
        require(self.calls==0,"replay_single_immutable_candidate")
        self.calls+=1
        return self._raw
