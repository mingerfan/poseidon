"""Generate bounded self-contained trusted-source/NPY fixtures using stdlib only."""
import ast,copy,hashlib,json,math,struct
from pathlib import Path
from cases import CASES,APPROXIMATION,execution_candidates,signatures,output_names,masks
HERE=Path(__file__).resolve().parent

def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def flattened(v):
 if isinstance(v,(tuple,list)):return [n for item in v for n in flattened(item)]
 return [v]
def npy_bytes(values,shape,dtype):
 count=math.prod(shape)
 if len(values)!=count or dtype not in ("<f8","|b1"):raise ValueError("fixture NPY type/shape")
 if dtype=="<f8":
  if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError("fixture finite weights")
  payload=struct.pack("<"+"d"*count,*values)
 else:
  if any(type(v) is not bool for v in values):raise ValueError("fixture Boolean mask")
  payload=bytes(values)
 header=repr(dict(descr=dtype,fortran_order=False,shape=tuple(shape))).encode("ascii")
 header+=b" "*((64-(10+len(header)+1)%64)%64)+b"\n"
 return b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header+payload

def source(case):
 e=case["e"];h=case["h"]
 ctor=f"nn.MultiheadAttention({e}, {h}, batch_first={case['batch_first']!r}, kdim={case.get('kdim')!r}, vdim={case.get('vdim')!r}, add_bias_kv={case.get('bias_kv',False)!r}, add_zero_attn={case.get('zero_attn',False)!r}, dtype=torch.float64)"
 lines=['"""Static eval native MHA composition; weights loaded by the trusted tool."""','import torch','from torch import nn','','class Model(nn.Module):','    def __init__(self):','        super().__init__()',f'        self.attention = {ctor}']
 if case.get("pre_width"):lines.append(f'        self.pre = nn.Linear({case["pre_width"]}, {e}, dtype=torch.float64)')
 if case.get("post_width"):lines.append(f'        self.post = nn.Linear({e}, {case["post_width"]}, dtype=torch.float64)')
 am,pm=masks(case)
 for name,mask in (("attn_mask",am),("padding_mask",pm)):
  expression='None' if mask is None else 'torch.tensor('+repr(mask['values'])+', dtype=torch.'+('bool' if mask['dtype']=='bool' else 'float64')+')'
  lines.append(f'        self.register_buffer({name!r}, {expression})')
 args='q' if case.get('self_attention') else 'q, k, v';lines+=['',f'    def forward(self, {args}):']
 if case.get('pre_width'):lines.append('        projected = self.pre(q)')
 else:lines.append('        projected = q')
 kv='projected, projected' if case.get('self_attention') else 'k, v'
 lines.append(f'        output, weights = self.attention(projected, {kv}, need_weights={case["need"]!r}, average_attn_weights={case["average"]!r}, attn_mask=self.attn_mask, key_padding_mask=self.padding_mask)')
 if case.get('post_width'):lines.append('        output = self.post(output)')
 if case.get('residual'):lines.append('        output = output + projected')
 if case['tree']=='named':lines.append('        return {"result": output, "weights": weights}')
 elif case['tree']=='tuple_none':lines.append('        return (output, [weights, None])')
 elif case.get('projected_output'):lines.append('        return {"result": output, "aux": [None, projected]}')
 else:lines.append('        return (output, None)')
 lines+=['','def make():','    return Model().eval()',''];text='\n'.join(lines);ast.parse(text);return text

def records(case,directory):
 e=case['e'];names=[]
 if case.get('kdim',e)==e and case.get('vdim',e)==e:names=[('attention.in_proj_weight',[3*e,e])]
 else:names=[('attention.q_proj_weight',[e,e]),('attention.k_proj_weight',[e,case.get('kdim',e)]),('attention.v_proj_weight',[e,case.get('vdim',e)])]
 names.append(('attention.in_proj_bias',[3*e]))
 if case.get('bias_kv'):names.extend([('attention.bias_k',[1,1,e]),('attention.bias_v',[1,1,e])])
 names.extend([('attention.out_proj.weight',[e,e]),('attention.out_proj.bias',[e])])
 if case.get('pre_width'):names.extend([('pre.weight',[e,case['pre_width']]),('pre.bias',[e])])
 if case.get('post_width'):names.extend([('post.weight',[case['post_width'],e]),('post.bias',[case['post_width']])])
 values={};offset=0
 for name,shape in names:
  n=math.prod(shape);values[name]=([math.sin(offset+i+1)*.08 for i in range(n)],shape,'<f8');offset+=n
 am,pm=masks(case)
 for name,mask in [('attn_mask',am),('padding_mask',pm)]:
  if mask is None:continue
  shape=[];v=mask['values']
  while isinstance(v,list):shape.append(len(v));v=v[0]
  values[name]=(flattened(mask['values']),shape,'|b1' if mask['dtype']=='bool' else '<f8')
 result={}
 for index,name in enumerate(sorted(values)):
  data,shape,dtype=values[name];raw=npy_bytes(data,shape,dtype);filename=f'weight_{index:03d}.npy';(directory/filename).write_bytes(raw)
  result[name]=dict(file=filename,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),shape=shape,dtype=dtype)
 return result

def main():
 output=HERE/'source_fixtures'
 if output.exists():raise ValueError('fresh source fixture directory required')
 output.mkdir();rows=[]
 for case in tuple(CASES)+execution_candidates():
  directory=output/case['id'];directory.mkdir();text=source(case);(directory/'model.py').write_text(text);weights=records(case,directory);profile=copy.deepcopy(case.get('approximation',APPROXIMATION))
  manifest=dict(format='poseidon-trusted-pytorch-v16',files={'model.py':hashlib.sha256(text.encode()).hexdigest()},entry='model.make',factory=dict(args=[],kwargs={}),dtype_policy='strict-float64-v1',inputs=signatures(case),outputs=output_names(case),weights=weights,approximations=dict(format='poseidon-pytorch-approximations-v1',defaults=profile,modules=[],nodes=[]))
  (directory/'manifest.json').write_bytes(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n')
  rows.append(dict(id=case['id'],manifest=str((directory/'manifest.json').relative_to(HERE)),manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),source_sha256=manifest['files']['model.py'],approximation=profile,configuration_kind='separate_declared_shallow_feasibility' if 'approximation' in case else 'fixed_original_math_profile',weight_files=len(weights),weight_elements=sum(math.prod(w['shape']) for w in weights.values()),original_accuracy_certified=False,source_capture_status='not_run',backend_status='not_run',agent_status='not_run'))
 index=dict(format='poseidon-ordered-mha-source-fixtures-v1',cases=rows,api_calls=0,compiler_calls=0,encrypted_executions=0,production_activation=False,reference_report='/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-ordered-composition-reference-v1/report.json',reference_report_binding='dbce73028d2835ce17a82d839e903e0e61d50a1f7a92ca7b1880f2241fdd090f',implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 index['binding']=hashlib.sha256(canonical(index)).hexdigest();(output/'index.json').write_bytes(json.dumps(index,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n');print(json.dumps(dict(status='prepared',cases=len(rows),binding=index['binding'],directory=str(output))))
if __name__=='__main__':main()
