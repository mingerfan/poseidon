"""Recheck retained decoding/reference linkage only; no model/backend execution."""
import ast, hashlib, io, json, math, struct, zipfile
from pathlib import Path
from audit import ROOT,RESULTS,Reader,require,strict,digest,shape_of,values,verify_index
HERE=Path(__file__).resolve().parent
INDEX=ROOT/"docs/baseline/agent-generic-qualification-index-r1378.json"
INDEX_SHA="f783cf46299fb70a1ec233c0e8e0d0e57fcf62820e0ac9522fcac8fc4a7d00ea"

def read_f64(raw):
    require(raw[:6]==b"\x93NUMPY" and len(raw)>=10 and
            raw[6]in(1,2,3) and raw[7]==0,"npy_magic_version")
    size_len=2 if raw[6]==1 else 4
    header_len=int.from_bytes(raw[8:8+size_len],"little")
    require(0<header_len<=65536 and len(raw)>=8+size_len+header_len,"npy_header_size")
    start=8+size_len;stop=start+header_len
    meta=ast.literal_eval(raw[start:stop].decode("utf-8"if raw[6]==3 else"latin1").strip())
    require(type(meta)is dict and set(meta)=={"descr","shape","fortran_order"} and
            meta["descr"]=="<f8" and meta["fortran_order"]is False,"npy_f64_c_order")
    shape=meta["shape"]
    require(type(shape)is tuple and 1<=len(shape)<=4 and
            all(type(d)is int and 0<d<=65536 for d in shape),"npy_static_shape")
    count=math.prod(shape);body=raw[stop:]
    require(len(body)==8*count and len(body)<=1024**2,"npy_body_size")
    data=struct.unpack("<"+"d"*count,body)
    require(all(math.isfinite(x)for x in data),"npy_finite_data")
    return shape,data

def select(outputs,selectors):
    shape=shape_of(outputs)
    require(len(shape)==3 and all(x>0 for x in shape),"execution_output_shape")
    list(values(outputs))
    require(type(selectors)is list and 0<len(selectors)<=256,"decode_selector_count")
    for selector in selectors:
        require(type(selector)is list and len(selector)==2 and
                all(type(x)is int for x in selector)and
                0<=selector[0]<shape[1] and 0<=selector[1]<shape[2],"decode_selector_bounds")
    return tuple(outputs[g][c][s]for g in range(shape[0])for c,s in selectors)

def reference_from_npz(raw):
    with zipfile.ZipFile(io.BytesIO(raw))as z:
        entries=[x for x in z.infolist()if x.filename=="reference.npy"]
        require(len(entries)==1 and 0<entries[0].file_size<=1024**2,"npz_unique_reference")
        return read_f64(z.read(entries[0]))

def run():
    reader=Reader((ROOT,RESULTS));idx=reader.json(INDEX,INDEX_SHA);verify_index(idx,reader)
    rows=[]
    for row in idx["historical_cpu"]["rows"]:
        folder=Path(row["source_report"]).parent
        def file(name,json_value=False):
            p=folder/name
            require(str(p)in idx["files"],"numeric_link_index_file")
            raw=reader.read(p,idx["files"][str(p)])
            return strict(raw)if json_value else raw
        report=file("report.json",True);attempt=report["attempts"][0]
        request=file("request.json",True)
        require(request["model"]==file("model.json",True),"numeric_link_request_model")
        execution=file("attempt-00/output/execution.json",True)
        require({k:v for k,v in execution.items()if k!="outputs"}==attempt["execution"],
                "numeric_link_execution_metadata")
        actual=tuple(values(attempt["comparison"]["actual"]))
        ref=tuple(values(attempt["comparison"]["reference"]))
        selected=select(execution["outputs"],request["layout"]["output_selectors"])
        shape,decoded=read_f64(file("attempt-00/output/decrypted.npy"))
        ref_shape,reference=reference_from_npz(file("arrays.npz"))
        require(shape==shape_of(attempt["comparison"]["actual"])==
                shape_of(attempt["comparison"]["reference"])==ref_shape and
                actual==selected==decoded and ref==reference,"numeric_link_values")
        rows.append(dict(id=row["id"],shape=list(shape),values=len(actual),
                         execution_decode_report_exact=True,reference_frozen_npz_exact=True,
                         original_reference_reexecuted=False,current_encrypted_execution=False))
    controls=reader.json(HERE/"numeric-link-controls.json")
    require(controls["status"]=="passed"and controls["tests"]==9 and
            controls["source_sha256"]==hashlib.sha256(reader.read(HERE/"numeric_link.py")).hexdigest()and
            controls["test_source_sha256"]==hashlib.sha256(reader.read(HERE/"test_numeric_link.py")).hexdigest(),
            "numeric_link_controls_binding")
    data=dict(format="poseidon-historical-numeric-link-audit-v1",index_binding=idx["binding"],
              index_sha256=INDEX_SHA,rows=rows,planned=len(rows),passed=len(rows),
              failed=0,skipped=0,not_run=0,compared_values=sum(r["values"]for r in rows),
              controls=controls,scope="retained historical files; not reference/FHE re-execution",
              compiler_calls=0,encrypted_calls=0,provider_calls=0,files=reader.known())
    data["binding"]=digest(data)
    output=ROOT/"docs/baseline/agent-historical-numeric-link-r1379.json"
    with output.open("x")as f:json.dump(data,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps({k:data[k]for k in["binding","planned","passed","failed","skipped","not_run","compared_values","compiler_calls","encrypted_calls","provider_calls"]}))

if __name__=="__main__":
    run()
