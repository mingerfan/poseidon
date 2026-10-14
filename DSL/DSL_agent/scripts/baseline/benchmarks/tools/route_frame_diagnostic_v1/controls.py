"""Small mathematical controls for the public symbolic route diagnostic."""
import ast,hashlib,json,os,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import inspect_routes as a
checks=0
def same(x,y):
 global checks
 assert x==y,(x,y);checks+=1
same(a.permutation_indices([2,3],[1,0]),[0,3,1,4,2,5])
same([a.flat(a.coordinate(j,[2,3,2]),[2,3,2])for j in range(12)],list(range(12)))
v=a.basis(4,3);f=a.Frame(4,{"x":v,"mask":a.coefficients([1,0,1,0],4)})
same(f.evaluate(ast.parse("x.rotate(1)",mode="eval").body),[v[1],v[2],v[3],v[0]])
same(f.evaluate(ast.parse("x.rotate(-1)",mode="eval").body),[v[3],v[0],v[1],v[2]])
same(f.evaluate(ast.parse("mask*x",mode="eval").body),[v[0],{},v[2],{}])
same(f.evaluate(ast.parse("x+x",mode="eval").body),[{(0,):2.},{(1,):2.},{(2,):2.},{}])
same(f.evaluate(ast.parse("x*x",mode="eval").body),[{(0,0):1.},{(1,1):1.},{(2,2):1.},{}])
p=8;g=a.basis(p,6);pre=[0.]*p;pre[0]=1.
f=a.Frame(p,{"g":g,"pre":a.coefficients(pre,p)})
correct=f.evaluate(ast.parse("(pre*g.rotate(1)).rotate(6)",mode="eval").body)
bad=f.evaluate(ast.parse("pre*g.rotate(7)",mode="eval").body)
expected=[{}for _ in range(p)];expected[2]={(1,):1.}
same(correct,expected)
same(a.compare(correct,expected,6)["mismatched_slots"],0)
same(a.compare(bad,expected,6)["mismatched_slots"],1)
try:f.evaluate(ast.parse("g.open('/tmp')",mode="eval").body)
except ValueError:checks+=1
else:raise AssertionError("forbidden_call_not_rejected")
scope=json.loads((pathlib.Path(__file__).resolve().parent/"scope.json").read_bytes());a.bound(scope)
for name,want in scope["evidence_files"].items():assert a.sha(name)==want
out=dict(format="poseidon-route-frame-controls-v1",status="passed",passed=checks,failed=0,skipped=0,
 scope_binding=scope["binding"],script_sha256=a.sha(__file__),inspector_sha256=a.sha(a.__file__),
 private_inputs_or_reference_read=False,new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0)
out["binding"]=a.digest(out);dest=pathlib.Path(__file__).resolve().parent/"controls.json"
with dest.open("xb")as stream:os.chmod(dest,0o600);stream.write(a.canonical(out))
print(json.dumps({k:out[k]for k in["status","passed","failed","skipped","binding"]}))
