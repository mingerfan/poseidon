import ast,json,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'test-results/resnet20-boot-placement-audit-20261026/profile-probe'
OUT.mkdir(exist_ok=False)
CT='tensor<1x!earth.ci<0 * 0>>';PT='tensor<1x!earth.pl<0 * 0>>'
lines=[];constants={}
class Expr:
 def __init__(self,name,plain=False):self.name=name;self.plain=plain
 def ty(self):return PT if self.plain else CT
 @staticmethod
 def c(value):
  value=float(value)
  if value not in constants:
   name=f'p{len(constants)}';constants[value]=Expr(name,True)
   lines.append(f'%{name} = "earth.constant"() {{value = dense<{value:.17e}> : tensor<32768xf64>, rms_var = 1.0 : f64}} : () -> {PT}')
  return constants[value]
 def op(self,kind,b=None):
  args=[self]+([] if b is None else [b if isinstance(b,Expr) else Expr.c(b)])
  name=f'v{len(lines)}'
  lines.append(f'%{name} = "earth.{kind}"('+','.join('%'+x.name for x in args)+') : ('+','.join(x.ty() for x in args)+f') -> {CT}')
  return Expr(name)
 def __add__(self,b):return self.op('add',b)
 def __radd__(self,b):return self.__add__(b)
 def __mul__(self,b):return self.op('mul',b)
 def __rmul__(self,b):return self.__mul__(b)
 def __neg__(self):return self.op('negate')
 def __sub__(self,b):return self+-b
source=ROOT/'test-results/model-regression-20261021/source/ResNet20.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='component')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
base=ROOT/'third_party/ckks-runtime/third_party/dacapo/python/poly/poly/data'
values=[float(v) for v in (ROOT/'bench/resnet20_s2c_first/support/data/resnet20/relu_param/d13.txt').read_text().split()]
p1=ns['component']((base/'tree15.txt').read_text().splitlines(),values[:16],15,4,2.,Expr.c)
p2=ns['component']((base/'tree15.txt').read_text().splitlines(),values[16:32],15,4,1.7,Expr.c)
p3=ns['component']((base/'tree27.txt').read_text().splitlines(),values[32:],27,8,2.,Expr.c)
x=Expr('x');y=x*(p3(p2(p1(x)))+Expr.c(.5))
text='module { func.func @relu(%x: '+CT+') -> '+CT+' {\n'+'\n'.join(lines)+f'\nreturn %{y.name} : {CT}\n'+'}}\n'
(OUT/'relu.mlir').write_text(text)
baseprofile=json.loads((ROOT/'test-results/relu-compiler-20261024/compiler-profile.json').read_text())
rows=[]
for budget,boot in [(18,45),(25,45),(25,40),(30,40)]:
 cfg=json.loads(json.dumps(baseprofile));cfg.update(levelUpperBound=budget+1,bootstrapLevelUpperBound=budget+1)
 for name,lat in cfg['latencyTable'].items():cfg['latencyTable'][name]=(lat+[lat[-1]]*(budget+2))[:budget+2]
 name=f'budget{budget}-boot{boot}';profile=OUT/f'{name}.json';profile.write_text(json.dumps(cfg))
 command=[str(ROOT/'build-relu-20261024/hecate-opt-final'),'--dacapo-precompile','--polynomial-optimizations=double-align','--boot-placement=optimized',f'--ckks-config={profile}','--waterline=40',f'--bootstrap-scale={boot}','--mlir-disable-threading',str(OUT/'relu.mlir'),'-o',str(OUT/f'{name}.mlir')]
 t=time.monotonic()
 r=subprocess.run(command,capture_output=True,text=True,timeout=60)
 (OUT/f'{name}.log').write_text(r.stdout+r.stderr)
 assert r.returncode==0,r.stderr[-3000:]
 ir=(OUT/f'{name}.mlir').read_text();count=ir.count('"ckks.bootstrapc"')
 ret=next(l.strip() for l in ir.splitlines() if l.strip().startswith('return'))
 row=dict(budget=budget,boot_scale=boot,boots=count,return_type=ret,elapsed=time.monotonic()-t)
 rows.append(row);print(json.dumps(row),flush=True)
(OUT/'summary.json').write_text(json.dumps(dict(scope='Exact source d13 ReLU expression compiled alone; nominal 30-bit logical planning only; no full-model compilation or native CKKS execution. Input scale40. Latency arrays extended by repeating final entries solely for feasibility probing.',rows=rows),indent=2)+'\n')
