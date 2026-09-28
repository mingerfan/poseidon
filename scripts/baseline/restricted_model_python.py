"""Static, bounded Python function frontend. Never imports or executes input code."""
import ast
import hashlib
import copy
from pathlib import PurePosixPath
from benchmark_graph import require, NAME, canonical, shape
from model_decomposition import FORMAT, REGISTRY, OPS, decompose

FORMAT_PY="poseidon-python-model-v1"
CALLS={"torch.add":"add","torch.sub":"subtract","torch.mul":"multiply","torch.neg":"negate",
       "torch.square":"square","torch.matmul":"matmul","torch.dot":"dot",
       "torch.reshape":"reshape","torch.flatten":"flatten","torch.sum":"sum","torch.mean":"mean",
       "torch.stack":"stack","torch.cat":"concat","torch.transpose":"transpose",
       "torch.nn.functional.linear":"linear","poseidon.rms_norm":"rms_norm",
       "torch.softmax":"softmax","torch.nn.functional.softmax":"softmax",
       "torch.exp":"exp","torch.rsqrt":"rsqrt","torch.reciprocal":"reciprocal",
       "poseidon.rope":"rope"}
class Ref:
    def __init__(self,name):self.name=name

class Frontend:
    def __init__(self,manifest,files):
        require(type(manifest) is dict and set(manifest)=={"format","id","files","entry","inputs","constants","outputs"}
                and manifest["format"]==FORMAT_PY,"Python manifest fields/version")
        require(type(files) is dict and 1<=len(files)<=8 and set(files)==set(manifest["files"]),"Explicit file inventory")
        require(sum(len(s.encode()) for s in files.values())<=65536,"Python package budget")
        self.functions={};self.imports={};self.stack=[];self.steps=0;self.counter=0
        self.graph=dict(format=FORMAT,id=manifest["id"],inputs=copy.deepcopy(manifest["inputs"]),
                        constants=copy.deepcopy(manifest["constants"]),nodes=[],outputs=[])
        self.reserved={i["name"] for i in manifest["inputs"]}|set(manifest["constants"])
        require(len(self.reserved)==len(manifest["inputs"])+len(manifest["constants"]),"Duplicate input/constant")
        self.public={canonical(v):n for n,v in manifest["constants"].items()}
        self.globals={n:Ref(n) for n in manifest["constants"]}
        self.manifest=manifest
        modules={}
        for path,source in files.items():
            p=PurePosixPath(path)
            require(not p.is_absolute() and ".." not in p.parts and p.suffix==".py"
                    and len(p.parts)==1 and p.stem.isidentifier() and not p.stem.startswith("_"),"Flat safe module path")
            require(type(source) is str and hashlib.sha256(source.encode()).hexdigest()==manifest["files"][path],"Source hash mismatch")
            module=p.stem;tree=ast.parse(source)
            require(sum(1 for _ in ast.walk(tree))<=4096,"Python AST budget")
            modules[module]=tree
        for module,tree in modules.items():
            aliases={};self.imports[module]=aliases
            for node in tree.body:
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and type(node.value.value) is str:continue
                if isinstance(node,ast.Import):
                    for item in node.names:
                        require(item.name in ("torch","torch.nn.functional","poseidon"),"Import not allowed")
                        alias=item.asname or item.name.split(".")[0]
                        require(alias not in aliases,"Duplicate import")
                        aliases[alias]=item.name if item.asname else item.name.split(".")[0]
                elif isinstance(node,ast.ImportFrom):
                    require(node.level==0 and node.module in modules,"Only listed local helper imports")
                    for item in node.names:
                        require(item.name!="*" and not item.name.startswith("_"),"Wildcard/private import")
                        alias=item.asname or item.name
                        require(alias not in aliases,"Duplicate import")
                        aliases[alias]=node.module+"."+item.name
                elif isinstance(node,ast.FunctionDef):
                    require(not node.decorator_list and not node.returns and not node.type_comment,"Decorators/annotations unsupported")
                    require(not node.name.startswith("_"),"Private entry/function")
                    key=module+"."+node.name
                    require(key not in self.functions,"Duplicate function")
                    self.functions[key]=node
                else:raise ValueError("Top-level executable code/classes are not allowed")
            names={k.split(".")[-1] for k in self.functions if k.startswith(module+".")}
            require(not names.intersection(aliases) and not names.intersection(self.globals)
                    and not set(aliases).intersection(self.globals),"Ambiguous module namespace")
    def fresh(self):
        while True:
            n="py_value_"+str(self.counter);self.counter+=1
            if n not in self.reserved:self.reserved.add(n);return n
    def ref(self,v):
        if isinstance(v,Ref):return v.name
        shape(v);key=canonical(v)
        if key not in self.public:
            require(len(self.graph["constants"])<32,"Public constant budget")
            n=self.fresh();self.graph["constants"][n]=v;self.public[key]=n
        return self.public[key]
    def emit(self,op,args,attrs=None):
        require(op in OPS or op in REGISTRY,"Unregistered Python operator")
        require(len(self.graph["nodes"])<64,"Python expanded node budget")
        out=self.fresh()
        self.graph["nodes"].append(dict(id=out,op=op,inputs=[self.ref(a) for a in args],
            attrs=attrs or {},outputs=[out]))
        return Ref(out)
    def expr(self,n,env,module):
        self.steps+=1;require(self.steps<=4096,"Python evaluation budget")
        if isinstance(n,ast.Name):
            require(n.id in env,"Undefined value: "+n.id);return env[n.id]
        if isinstance(n,ast.Constant):
            require(type(n.value) in (int,float,bool,str,type(None)),"Unsupported literal")
            return n.value
        if isinstance(n,(ast.List,ast.Tuple)):
            require(len(n.elts)<=256,"Container budget")
            return [self.expr(x,env,module) for x in n.elts]
        if isinstance(n,ast.Dict):
            keys=[self.expr(k,env,module) for k in n.keys]
            require(all(type(k) is str for k in keys) and len(keys)==len(set(keys)),"Literal dictionary keys")
            return {k:self.expr(v,env,module) for k,v in zip(keys,n.values)}
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            v=self.expr(n.operand,env,module)
            if isinstance(v,Ref):return self.emit("negate",[v]) if isinstance(n.op,ast.USub) else v
            require(type(v) in (int,float),"Numeric unary")
            return -v if isinstance(n.op,ast.USub) else v
        if isinstance(n,ast.BinOp):
            a=self.expr(n.left,env,module);b=self.expr(n.right,env,module)
            if isinstance(a,Ref) or isinstance(b,Ref):
                if isinstance(n.op,ast.Pow):
                    require(isinstance(a,Ref) and type(b) is int and b in (2,4),"Power exponent")
                    return self.emit("power",[a],dict(exponent=b))
                op={ast.Add:"add",ast.Sub:"subtract",ast.Mult:"multiply",ast.MatMult:"matmul"}.get(type(n.op))
                require(op is not None,"Unsupported encrypted binary operator")
                if not isinstance(a,Ref) and op in ("add","multiply"):a,b=b,a
                if not isinstance(a,Ref) and op=="subtract":return self.emit("add",[self.emit("negate",[b]),a])
                return self.emit(op,[a,b])
            require(type(a) in (int,float) and type(b) in (int,float),"Public numeric operation")
            fn={ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b}
            require(type(n.op) in fn,"Public operator unsupported")
            value=fn[type(n.op)]();shape(value);return value
        if isinstance(n,ast.Subscript):
            v=self.expr(n.value,env,module);idx=self.expr(n.slice,env,module)
            require(type(v) in (list,dict) and type(idx) in (int,str),"Public container indexing only")
            try:return v[idx]
            except (KeyError,IndexError,TypeError):raise ValueError("Public index bounds")
        if isinstance(n,ast.Call):
            require(not any(isinstance(a,ast.Starred) for a in n.args) and all(k.arg is not None for k in n.keywords),"Argument expansion unsupported")
            require(len({k.arg for k in n.keywords})==len(n.keywords),"Duplicate keyword")
            name=self.resolve(n.func,module)
            args=[self.expr(a,env,module) for a in n.args]
            kw={k.arg:self.expr(k.value,env,module) for k in n.keywords}
            if name in self.functions:return self.call(name,args,kw)
            require(name in CALLS,"Unregistered call: "+name)
            op=CALLS[name]
            if op in ("add","subtract","multiply","matmul","dot"):
                require(len(args)==2 and not kw,"Binary call signature");return self.emit(op,args)
            if op=="linear":
                require(2<=len(args)<=3 and set(kw)<= {"bias"} and not(len(args)==3 and kw),"Linear signature")
                if kw.get("bias") is not None:args.append(kw["bias"])
                return self.emit(op,args)
            if op in ("negate","square","flatten"):
                require(len(args)==1 and not kw,"Unary call signature");return self.emit(op,args)
            if op in ("sum","mean"):
                require(len(args)==1 and set(kw)<= {"dim","keepdim"} and "dim" in kw,"Static reduction signature")
                dims=kw["dim"] if type(kw["dim"]) is list else [kw["dim"]]
                return self.emit(op,args,dict(axes=dims,keepdims=kw.get("keepdim",False)))
            if op in ("stack","concat"):
                require(len(args)==1 and type(args[0]) is list and set(kw)<= {"dim"},"Join signature")
                return self.emit(op,args[0],dict(axis=kw.get("dim",0)))
            if op=="reshape":
                require(len(args)==2 and not kw and type(args[1]) is list,"Reshape signature")
                return self.emit(op,args[:1],dict(shape=args[1]))
            if op=="transpose":
                require(len(args)==3 and not kw,"Transpose signature")
                return self.emit(op,args[:1],dict(dim0=args[1],dim1=args[2]))
            # Approximation metadata is an explicit frontend extension, never a silent torch substitution.
            if op in ("exp","rsqrt","reciprocal"):
                require(len(args)==1 and set(kw)=={"approximation"},"Explicit approximation keyword required")
                return self.emit(op,args,kw)
            if op=="softmax":
                require(len(args)==1 and set(kw)=={"dim","exp","reciprocal"},"Explicit softmax approximation required")
                return self.emit(op,args,dict(axis=kw["dim"],exp=kw["exp"],reciprocal=kw["reciprocal"]))
            if op=="rms_norm":
                require(len(args)==2 and set(kw)=={"eps","approximation"},"Restricted RMSNorm signature")
                return self.emit(op,args,kw)
            if op=="rope":
                require(len(args)==1 and set(kw)=={"position","theta"},"RoPE public signature")
                return self.emit(op,args,kw)
            raise ValueError("Unsupported call signature")
        raise ValueError("Unsupported Python expression: "+type(n).__name__)
    def resolve(self,n,module):
        if isinstance(n,ast.Name):
            return self.imports[module].get(n.id,module+"."+n.id)
        if isinstance(n,ast.Attribute):
            require(not n.attr.startswith("_"),"Private attribute")
            return self.resolve(n.value,module)+"."+n.attr
        raise ValueError("Dynamic callable unsupported")
    def statements(self,body,env,module):
        for n in body:
            self.steps+=1;require(self.steps<=4096,"Python statement budget")
            if isinstance(n,ast.Assign):
                require(len(n.targets)==1 and isinstance(n.targets[0],ast.Name),"Only name assignment")
                name=n.targets[0].id
                require(name not in self.imports[module] and module+"."+name not in self.functions,"Cannot shadow callable")
                env[name]=self.expr(n.value,env,module)
            elif isinstance(n,ast.Return):return True,self.expr(n.value,env,module)
            elif isinstance(n,ast.For):
                require(isinstance(n.target,ast.Name) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name)
                        and n.iter.func.id=="range" and not n.iter.keywords and not n.orelse,"Public range loop only")
                require("range" not in env and "range" not in self.imports[module] and module+".range" not in self.functions,"Shadowed range")
                require(n.target.id not in self.imports[module] and module+"."+n.target.id not in self.functions,"Loop target shadows callable")
                args=[self.expr(x,env,module) for x in n.iter.args]
                require(1<=len(args)<=3 and all(type(x) is int and abs(x)<=256 for x in args),"Range bounds")
                try:r=range(*args)
                except ValueError:raise ValueError("Zero range step")
                require(len(r)<=64,"Loop expansion budget")
                for i in r:
                    env[n.target.id]=i
                    returned,value=self.statements(n.body,env,module)
                    if returned:return True,value
            elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and type(n.value.value) is str:pass
            else:raise ValueError("Unsupported Python statement: "+type(n).__name__)
        return False,None
    def call(self,name,args,kw):
        require(name in self.functions and name not in self.stack and len(self.stack)<8,"Unknown or recursive helper")
        n=self.functions[name];a=n.args
        require(not a.posonlyargs and not a.vararg and not a.kwarg and not a.kwonlyargs and not a.defaults,"Simple static function signature required")
        names=[p.arg for p in a.args]
        require(all(p.annotation is None for p in a.args) and len(args)<=len(names),"Function signature")
        require(set(kw)<=set(names) and not set(names[:len(args)]).intersection(kw),"Argument binding")
        values=dict(zip(names,args));values.update(kw);require(set(values)==set(names),"Missing arguments")
        module=name.rsplit(".",1)[0]
        require(not set(names).intersection(self.imports[module]) and not any(module+"."+x in self.functions for x in names),"Shadowed callable parameter")
        env=dict(self.globals);env.update(values);self.stack.append(name)
        try:returned,value=self.statements(n.body,env,module)
        finally:self.stack.pop()
        require(returned,"Function must return");return value

def parse(manifest,files):
    f=Frontend(manifest,files)
    result=f.call(manifest["entry"],[Ref(i["name"]) for i in manifest["inputs"]],{})
    values=result if type(result) is list else [result]
    require(type(manifest["outputs"]) is list and len(values)==len(manifest["outputs"])
            and all(isinstance(v,Ref) for v in values),"Named encrypted outputs")
    f.graph["outputs"]=[dict(name=n,value=v.name) for n,v in zip(manifest["outputs"],values)]
    lowered,binding=decompose(f.graph)
    binding["python_package_sha256"]=__import__("benchmark_graph").digest(manifest)
    binding["python_source_hashes"]=copy.deepcopy(manifest["files"])
    from pathlib import Path
    binding["python_frontend_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    binding["binding_sha256"]=__import__("benchmark_graph").digest({k:v for k,v in binding.items() if k!="binding_sha256"})
    return f.graph,lowered,binding
