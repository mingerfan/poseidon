"""Lossless bounded public-vector wire storage; versioned RLE and row dictionaries."""
import json,math

VERSION="capacity-public-vector-rle-v1"
TILES_VERSION="capacity-public-vector-tiles-v2"
BINDINGS_VERSION="capacity-public-bindings-table-v1"
BINDINGS_MAX_ROWS=1024
BINDINGS_MAX_COLUMNS=16

def compact_bindings(bindings):
    if type(bindings) is not list or not 1<=len(bindings)<=BINDINGS_MAX_ROWS:
        return bindings
    if any(type(row) is not dict or not row for row in bindings):return bindings
    columns=sorted(set().union(*(set(row) for row in bindings)))
    if not 1<=len(columns)<=BINDINGS_MAX_COLUMNS or any(type(k) is not str or len(k)>64 for k in columns):
        return bindings
    for row in bindings:
        for value in row.values():
            if not (type(value) is int and abs(value)<=2**31 or type(value) is str and len(value)<=128):
                return bindings
    table=dict(format=BINDINGS_VERSION,columns=columns,rows=[[row.get(key) for key in columns] for row in bindings])
    encode=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
    return table if len(encode(table))<len(encode(bindings)) else bindings

def restore_bindings(descriptor):
    table=descriptor.get("bindings")
    if type(table) is not dict:return
    if set(table)!={"format","columns","rows"} or table["format"]!=BINDINGS_VERSION:
        raise ValueError("Capacity public binding storage metadata")
    columns,rows=table["columns"],table["rows"]
    if (type(columns) is not list or not 1<=len(columns)<=BINDINGS_MAX_COLUMNS
        or any(type(key) is not str or len(key)>64 for key in columns)
        or len(set(columns))!=len(columns)):
        raise ValueError("Capacity public binding storage columns")
    if type(rows) is not list or not 1<=len(rows)<=BINDINGS_MAX_ROWS:
        raise ValueError("Capacity public binding storage rows")
    bindings=[]
    for row in rows:
        if type(row) is not list or len(row)!=len(columns):
            raise ValueError("Capacity public binding storage row")
        decoded={}
        for key,value in zip(columns,row):
            if value is None:continue
            if not (type(value) is int and abs(value)<=2**31 or type(value) is str and len(value)<=128):
                raise ValueError("Capacity public binding storage cell")
            decoded[key]=value
        if not decoded:raise ValueError("Capacity public binding storage empty row")
        bindings.append(decoded)
    descriptor["bindings"]=bindings

def _runs(values):
    result=[]
    for value in values:
        if (result and type(result[-1][1]) is type(value) and result[-1][1]==value
            and (value!=0 or math.copysign(1.,result[-1][1])==math.copysign(1.,value))):
            result[-1][0]+=1
        else:result.append([1,value])
    return result

def _tiled_vector(value):
    encode=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)
    best=None;size=float("inf")
    for width in (8,16,32,64,128,256):
        if width>len(value) or len(value)%width:continue
        blocks=[];seen={};indices=[]
        for start in range(0,len(value),width):
            block=value[start:start+width];identity=encode(block)
            if identity not in seen:
                seen[identity]=len(blocks);blocks.append(_runs(block))
            index=seen[identity]
            if indices and indices[-1][1]==index:indices[-1][0]+=1
            else:indices.append([1,index])
        record=dict(format=TILES_VERSION,length=len(value),block_size=width,blocks=blocks,indices=indices)
        current=len(encode(record))
        if current<size:best=record;size=current
    return best

def _decode_runs(runs,length):
    if type(runs) is not list or not 1<=len(runs)<=length:
        raise ValueError("Capacity public vector storage runs")
    result=[]
    for run in runs:
        if (type(run) is not list or len(run)!=2 or type(run[0]) is not int
            or not 1<=run[0]<=length or type(run[1]) not in (int,float)
            or not math.isfinite(run[1]) or abs(run[1])>1024 or len(result)+run[0]>length):
            raise ValueError("Capacity public vector storage run")
        result.extend([run[1]]*run[0])
    if len(result)!=length:raise ValueError("Capacity public vector storage length")
    return result

def _decode_tiles(value,period):
    if set(value)!={"format","length","block_size","blocks","indices"} or value["format"]!=TILES_VERSION or type(value["length"]) is not int or value["length"]!=period:
        raise ValueError("Capacity public tile storage metadata")
    width=value["block_size"]
    if type(width) is not int or width not in (8,16,32,64,128,256) or width>period or period%width:
        raise ValueError("Capacity public tile storage width")
    blocks=value["blocks"];indices=value["indices"];count=period//width
    if type(blocks) is not list or not 1<=len(blocks)<=count or type(indices) is not list or not 1<=len(indices)<=count:
        raise ValueError("Capacity public tile storage count")
    decoded=[_decode_runs(block,width) for block in blocks];result=[];used=set()
    for run in indices:
        if (type(run) is not list or len(run)!=2 or type(run[0]) is not int
            or not 1<=run[0]<=count or type(run[1]) is not int or not 0<=run[1]<len(decoded)
            or len(result)+run[0]*width>period):
            raise ValueError("Capacity public tile storage index")
        result.extend(decoded[run[1]]*run[0]);used.add(run[1])
    if len(result)!=period or used!=set(range(len(blocks))):
        raise ValueError("Capacity public tile storage length/unused block")
    return result

def compact_request(request):
    if type(request) is not dict:return request
    origins=request.get("constant_origins")
    if type(origins) is not dict:return request
    descriptor=origins.get("capacity_tensor_packing")
    if type(descriptor) is not dict:return request
    if descriptor.get("version") not in ("capacity-public-tensors-v5","capacity-public-tensors-v6","capacity-public-tensors-v7","capacity-public-tensors-v8","capacity-public-tensors-v9","capacity-public-tensors-v10"):return request
    public=request.get("public_constants")
    if type(public) is not dict:return request
    result=dict(request);encoded={}
    for key,value in public.items():
        if type(value) is list and value and all(type(v) in (int,float) for v in value):
            runs=[]
            for v in value:
                if runs and type(runs[-1][1]) is type(v) and runs[-1][1]==v and (v!=0 or math.copysign(1.,runs[-1][1])==math.copysign(1.,v)):runs[-1][0]+=1
                else:runs.append([1,v])
            record=dict(format=VERSION,length=len(value),runs=runs)
            encode=lambda x:json.dumps(x,separators=(",",":"),sort_keys=True,allow_nan=False)
            best=record if len(encode(record))<len(encode(value)) else value
            if descriptor["version"] in ("capacity-public-tensors-v8","capacity-public-tensors-v9","capacity-public-tensors-v10"):
                tiled=_tiled_vector(value)
                if tiled is not None and len(encode(tiled))<len(encode(best)):best=tiled
            encoded[key]=best
        else:encoded[key]=value
    result["public_constants"]=encoded
    bindings=compact_bindings(descriptor.get("bindings"))
    if bindings is not descriptor.get("bindings"):
        result["constant_origins"]=dict(origins)
        result["constant_origins"]["capacity_tensor_packing"]=dict(descriptor,bindings=bindings)
    return result

def restore_public_constants(request):
    """Normalize only the trusted data representation before semantic validation."""
    origins=request.get("constant_origins")
    if type(origins) is not dict:return
    descriptor=origins.get("capacity_tensor_packing")
    if type(descriptor) is not dict or descriptor.get("version") not in ("capacity-public-tensors-v5","capacity-public-tensors-v6","capacity-public-tensors-v7","capacity-public-tensors-v8","capacity-public-tensors-v9","capacity-public-tensors-v10"):return
    restore_bindings(descriptor)
    public=request["public_constants"]
    if type(public) is not dict or len(public)>256:raise ValueError("Capacity public registry storage budget")
    period=request["layout"]["input_slot_period"]
    from packed_input_abi import rotations
    from unified_capacity_layout import WIDE_ABI
    authority=request["layout"] if request["layout"].get("execution_abi")==WIDE_ABI else None
    rotations(period,capacity_layout=authority)
    decoded={}
    for key,value in public.items():
        if type(value) is not dict:decoded[key]=value;continue
        if value.get("format")==TILES_VERSION:
            if descriptor["version"] not in ("capacity-public-tensors-v8","capacity-public-tensors-v9","capacity-public-tensors-v10"):
                raise ValueError("Capacity public tiles require packing v8 or v9")
            decoded[key]=_decode_tiles(value,period);continue
        if set(value)!={"format","length","runs"} or value["format"]!=VERSION or type(value["length"]) is not int or value["length"]!=period:
            raise ValueError("Capacity public vector storage metadata")
        runs=value["runs"]
        if type(runs) is not list or not 1<=len(runs)<=period:raise ValueError("Capacity public vector storage runs")
        vector=[]
        for run in runs:
            if type(run) is not list or len(run)!=2 or type(run[0]) is not int or not 1<=run[0]<=period or type(run[1]) not in (int,float) or not math.isfinite(run[1]) or abs(run[1])>1024 or len(vector)+run[0]>period:
                raise ValueError("Capacity public vector storage run")
            vector.extend([run[1]]*run[0])
        if len(vector)!=period:raise ValueError("Capacity public vector storage length")
        decoded[key]=vector
    request["public_constants"]=decoded
