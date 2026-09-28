"""Read-only diagnosis of ten frozen numerical failures.

This does not execute candidate Python, call a provider/compiler, replace a
reference, or award a benchmark pass. Closed forms were manually derived from
the exact retained final candidates; their source hashes are recorded.
"""
import argparse,ast,hashlib,io,json,math,struct,zipfile
from pathlib import Path

CASES={
 "construct_052_1": ("rotation_reduction", "P=4 with two logical entries: rotate(2) reads padding, yielding z/2 instead of the broadcast mean."),
 "construct_063_2": ("rotation_diagonals", "Composed Linear weights are valid; C1 and C3 are paired with the opposite positive rotation offsets."),
 "construct_074_2": ("rotation_placement", "Positive offsets 3/4/5 place slot-zero outputs at 5/4/3 in P=8, reversing the stacked Linear outputs."),
 "construct_097_1": ("rotation_split_concat", "Final masks select original slots [3,0,0], including padding, instead of logical split/concat order [1,2,0]."),
 "construct_107_2": ("rotation_placement", "Concatenating length-three inputs in P=8 needs a right shift by three; positive rotate(3) moves in the other direction."),
 "construct_109_2": ("polynomial_degree_and_padding", "Shift direction is correct, but the Horner sequence starts with y*c3 and yields degree four instead of three; polynomial padding also contaminates the first stacked output by 1/16."),
 "construct_112_2": ("rotation_and_padding", "Second branch rotates in the wrong direction, and the unmasked polynomial creates nonzero values in padding."),
 "construct_118_0": ("padding_polynomial", "Mean and second-branch placement are correct; first-branch polynomial padding adds 1/16 to the second stacked branch. The second named output is correct."),
 "construct_119_2": ("rotation_placement", "Both branches are masked, but positive rotate(3) places the second branch in slots 5/6/7, not 3/4/5."),
 "construct_152_2": ("rotation_linear", "Using rotate(1) for both rows reads padding in row one rather than the required first input cell.")
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def numpy_array(data):
 if len(data)>2*1024**2:raise ValueError("Diagnostic array size")
 f=io.BytesIO(data)
 if f.read(6)!=b"\x93NUMPY":raise ValueError("Array magic")
 version=f.read(2)
 if version not in (b"\x01\x00",b"\x02\x00"):raise ValueError("Array version")
 length=int.from_bytes(f.read(2 if version[0]==1 else 4),"little")
 if length>4096:raise ValueError("Array header bound")
 header=ast.literal_eval(f.read(length).decode("latin1"))
 if header["descr"]!="<f8" or header["fortran_order"] is not False:raise ValueError("Float64 C-order array required")
 shape=header["shape"]
 if not isinstance(shape,tuple) or not all(type(x)is int and 1<=x<=256 for x in shape):raise ValueError("Diagnostic shape")
 n=math.prod(shape);payload=f.read()
 if len(payload)!=8*n:raise ValueError("Array payload")
 flat=list(struct.unpack("<"+"d"*n,payload))
 if not all(math.isfinite(x) for x in flat):raise ValueError("Nonfinite data")
 def fold(values,dims):
  if len(dims)==1:return values
  size=math.prod(dims[1:])
  return [fold(values[i*size:(i+1)*size],dims[1:]) for i in range(dims[0])]
 return fold(flat,shape)
def arrays(path):
 with zipfile.ZipFile(path) as z:
  if set(z.namelist())!={"inputs.npy","reference.npy"}:raise ValueError("Frozen array keys")
  if any(x.file_size>2*1024**2 for x in z.infolist()):raise ValueError("Expanded size")
  return {n[:-4]:numpy_array(z.read(n)) for n in z.namelist()}
def polynomial(x,coeff,basis="power"):
 if basis=="power":return sum(c*x**i for i,c in enumerate(coeff))
 ts=[1.0,x]
 for i in range(2,len(coeff)):ts.append(2*x*ts[-1]-ts[-2])
 return sum(c*t for c,t in zip(coeff,ts))
def linear(x,w,b):return [sum(a*v for a,v in zip(row,x))+bias for row,bias in zip(w,b)]
def manual(ident,inputs,request):
 model=request["model"];cs=model["constants"];period=request["layout"]["input_slot_period"]
 x=inputs[0];y=inputs[1] if len(inputs)>1 else None
 rot=lambda a,k:[a[(j+k)%period] for j in range(period)]
 poly=lambda a:[polynomial(v,[.0625,-.125,.03125,.015625]) for v in a]
 if ident=="construct_052_1":
  z=inputs[2];rz=rot(z,2)
  return [polynomial(x[j],cs["c0"],"chebyshev")*y[j]**2+.5*(z[j]+rz[j]) for j in range(2)]
 if ident=="construct_063_2":
  w,u=cs["c0"],cs["c2"]
  m=[[sum(u[i][k]*w[k][j] for k in range(2)) for j in range(3)] for i in range(3)]
  b=linear(cs["c1"],u,cs["c3"])+[0]
  c0=[m[0][0],m[1][1],m[2][2],0];c1=[0,m[1][0],m[2][1],0]
  c2=[m[0][2],0,m[2][0],0];c3=[m[0][1],m[1][2],0,0]
  return [-x[j]-sum(rot(y,k)[j]*c[j] for k,c in enumerate([c0,c1,c2,c3]))-b[j] for j in range(3)]
 if ident=="construct_074_2":
  out=[v*v for v in x];v=linear(linear(y[:3],cs["c0"],cs["c1"]),cs["c2"],cs["c3"])
  for pos,value in zip([5,4,3],v):out[pos]+=value
  return out[:6]
 if ident=="construct_097_1":
  mean=(sum(inputs[2])+sum(inputs[3]))/3
  n=[(a+.125)*b*.125+mean for a,b in zip(x,y)]
  return [n[3],n[0],n[0]]
 if ident=="construct_107_2":return [.125*(a+b) for a,b in zip(x,rot(y,3))][:6]
 if ident=="construct_109_2":
  faulty=[.015625*v**4+.03125*v**2-.125*v+.0625 for v in y]
  return [.125*a+b for a,b in zip(x,rot(faulty,5))][:6]
 if ident=="construct_112_2":return [a+b for a,b in zip(poly(x),rot([v*v for v in y],3))][:6]
 if ident=="construct_118_0":
  n=poly(x);mean=sum(inputs[2])/3
  out=[a+.125*b+mean for a,b in zip(n,rot(y,5))]
  return out[:6]+n[:3]
 if ident=="construct_119_2":
  a=poly(x)[:3]+[0.0]*(period-3);b=poly(y)[:3]+[0.0]*(period-3)
  return [v+w for v,w in zip(a,rot(b,3))][:6]
 if ident=="construct_152_2":
  square=[v*v for v in x];shift=rot(square,1)
  return [-(square[0]*(-.125)+shift[0]*(-.03125)+.0625),
          -(square[1]*(-.09375)+shift[1]*(.09375)+.0625)]
 raise ValueError("Unknown diagnostic case")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--results",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError("Preserve prior diagnostic")
 summary_path=a.results/"stage2-agent-progress-r145.json";summary=json.loads(summary_path.read_text())
 selected={r["id"]:r for r in summary["rows"] if r.get("terminal_failure_layer")=="numerical_comparison"}
 if set(selected)!=set(CASES):raise ValueError("Frozen numerical failure denominator")
 parents={str(summary_path):sha(summary_path),str(Path(__file__).resolve()):sha(Path(__file__))}
 rows=[]
 for ident,row in selected.items():
  folder=Path(row["audited_result"]["evidence"]);rp=folder/"report.json"
  if sha(rp)!=row["audited_result"]["report_sha256"]:raise ValueError("Changed audited report")
  report=json.loads(rp.read_text());request=json.loads((folder/"request.json").read_text())
  attempt=report["attempts"][-1]
  if attempt["index"]!=3 or attempt["failure_layer"]!="numerical_comparison" or not attempt["executed"]:raise ValueError("Final encrypted failure expected")
  for name,h in report["frozen_hashes"].items():
   if sha(folder/name)!=h:raise ValueError("Frozen evidence changed")
  comparison=attempt["comparison"];data=arrays(folder/"arrays.npz")
  if data["reference"]!=comparison["reference"]:raise ValueError("Reference report mismatch")
  input_sets=data["inputs"]
  if len(request["model"]["inputs"])==1:input_sets=[[x] for x in input_sets]
  if len(input_sets)!=4 or any(len(x)!=len(request["model"]["inputs"]) or any(len(v)!=request["layout"]["input_slot_period"] for v in x) for x in input_sets):raise ValueError("Packed input dimensions")
  predicted=[manual(ident,v,request) for v in input_sets];actual=comparison["actual"]
  if [len(x) for x in predicted]!=[len(x) for x in actual]:raise ValueError("Output shape")
  errors=[abs(x-y) for aa,bb in zip(actual,predicted) for x,y in zip(aa,bb)]
  matches=all(abs(x-y)<=1e-5+1e-4*abs(y) for aa,bb in zip(actual,predicted) for x,y in zip(aa,bb))
  wrong=max(abs(x-y) for aa,bb in zip(predicted,data["reference"]) for x,y in zip(aa,bb))
  if not matches or wrong<1e-3:raise ValueError("Manual fault explanation not supported: "+ident)
  for p in [rp,folder/"request.json",folder/"arrays.npz",folder/"attempt-03/candidate.py",folder/"attempt-03/report.json"]:
   parents[str(p)]=sha(p)
  rows.append(dict(id=ident,model_id=request["model"]["id"],failure_kind=CASES[ident][0],explanation=CASES[ident][1],
   actual_matches_manual_fault_semantics=matches,manual_vs_decrypted_max_absolute_error=max(errors),
   manual_vs_reference_max_absolute_error=wrong,original_decrypted_vs_reference_max_absolute_error=comparison["max_absolute_error"],
   actual_compared_values=sum(map(len,actual)),evidence=str(folder),candidate_source_sha256=sha(folder/"attempt-03/candidate.py"),
   status="original_numerical_failure_preserved"))
 result=dict(format="poseidon-numerical-failure-diagnosis-r149",rows=rows,parents=parents,
  diagnosed=10,all_manual_predictions_match=True,original_statuses_modified=False,new_agent_passes=0,
  new_paid_calls=0,new_compilations=0,new_encrypted_executions=0,stage2_complete=False,
  limitation="Human-derived finite diagnostic formulas, not an independent DSL interpreter or a proof for arbitrary inputs. Matches explain observed failures without changing references or awarding passes.")
 result["binding"]=hashlib.sha256(canonical(result)).hexdigest()
 with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
 print(json.dumps(dict(binding=result["binding"],diagnosed=10,max_prediction_error=max(r["manual_vs_decrypted_max_absolute_error"] for r in rows),new_paid_calls=0)))
if __name__=="__main__":main()
