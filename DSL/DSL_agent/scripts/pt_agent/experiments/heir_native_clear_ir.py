"""Strict reader and sparse-copy analysis for trusted native HEIR arithmetic IR.

No model data or cryptographic objects are loaded by this module. The optional
bounded integer checks reuse the independently installed Orbit Gurobi runtime.
"""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re


class Unsupported(ValueError):
    pass


def split_top(text, delimiter=','):
    result=[]; start=0; stack=[]; quoted=False; escaped=False
    matching={')':'(',']':'[','}':'{','>':'<'}
    for i,c in enumerate(text):
        if quoted:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':quoted=False
            continue
        if c=='"':quoted=True
        elif c in '([{<':stack.append(c)
        elif c in ')]}>':
            if stack and stack[-1]==matching[c]:stack.pop()
        elif c==delimiter and not stack:
            result.append(text[start:i].strip());start=i+1
    result.append(text[start:].strip())
    return result


def shape_type(text):
    m=re.search(r'memref<([^,>]+)',text)
    if not m:raise Unsupported('A static memref type is required: '+text[:100])
    parts=m[1].split('x');shape=[]
    while parts and parts[0].isdigit():shape.append(int(parts.pop(0)))
    if not shape or not parts:raise Unsupported('Dynamic or unsupported memref shape')
    element='x'.join(parts)
    if element not in ('f64','!pt','!ct','!linear_transformation'):
        raise Unsupported('Unsupported memref element '+element)
    return shape,element


@dataclass
class Node:
    kind:str
    line:int
    text:str=''
    defs:list=field(default_factory=list)
    attrs:dict=field(default_factory=dict)
    body:list=field(default_factory=list)


@dataclass
class Function:
    name:str
    line:int
    args:list
    returns:list
    body:list


@dataclass
class Program:
    globals:dict
    functions:dict
    operation_counts:dict
    ssa_alpha_renames:dict=field(default_factory=dict)


def canonical_ssa_names(text):
    """Alpha-rename valid hyphenated SSA names for the small reader only.

    Quoted attribute/resource text is preserved. Source lines and all operand
    relationships remain unchanged; the original saved native IR is untouched.
    """
    pattern=r'"(?:[^"\\]|\\.)*"|%[-\w]+'
    tokens=re.findall(pattern,text)
    used={token for token in tokens if token.startswith('%')}
    mapping={}
    for token in tokens:
        if not token.startswith('%') or '-' not in token or token in mapping:continue
        index=len(mapping)
        while '%__native_alpha_'+str(index) in used:index+=1
        mapping[token]='%__native_alpha_'+str(index);used.add(mapping[token])
    return re.sub(pattern,lambda match:mapping.get(match[0],match[0]),text),mapping


def instruction(text,line):
    m=re.match(r'(%[\w]+:\d+|%[\w]+(?:\s*,\s*%[\w]+)*)\s*=\s*(.*)',text)
    if m and ':' in m[1]:
        base,size=m[1].split(':');defs=[base+'#'+str(i) for i in range(int(size))]
    else:defs=re.findall(r'%[\w]+',m[1]) if m else []
    raw=m[2] if m else text
    op=raw.split(' ',1)[0].split('(',1)[0]
    if op=='return':op='return'
    return Node(op,line,raw,defs)


def parse(text):
    text,ssa_alpha_renames=canonical_ssa_names(text)
    lines=text.splitlines();globals={};functions={};counts={}
    starts=[i for i,line in enumerate(lines) if re.match(r'\s*func.func (?:private )?@',line)]
    for i,line in enumerate(lines[:starts[0] if starts else len(lines)]):
        if 'memref.global ' not in line:continue
        name=re.search(r'@([\w]+)',line)[1]
        shape,element=shape_type(line)
        m=re.search(r'= dense<([^<>]+)>',line)
        if element!='f64' or not m or m[1].startswith('['):raise Unsupported('Only finite scalar-splat globals are supported')
        if m[1].startswith('"0x') and m[1].endswith('"'):
            raw=m[1][3:-1]
            if not re.fullmatch('[0-9A-Fa-f]+',raw) or len(raw)!=math.prod(shape)*16:
                raise Unsupported('Malformed native raw f64 global at line '+str(i+1))
            globals[name]={'shape':shape,'raw_hex':raw,'line':i+1}
        else:
            try:value=float(m[1])
            except ValueError:raise Unsupported('Unsupported scalar global at line '+str(i+1)) from None
            if not math.isfinite(value):raise Unsupported('Nonfinite global')
            globals[name]={'shape':shape,'fill':value,'line':i+1}
    for ordinal,start in enumerate(starts):
        header=lines[start].strip();m=re.match(r'func.func (?:private )?@([\w]+)\(',header);name=m[1]
        pos=m.end();depth=1;end=pos
        while depth and end<len(header):
            if header[end]=='(':depth+=1
            elif header[end]==')':depth-=1
            end+=1
        if depth:raise Unsupported('Multiline function header')
        args=[]
        for part in split_top(header[pos:end-1]):
            if part:
                a=re.match(r'(%[\w]+)\s*:\s*(.*)',part)
                if not a:raise Unsupported('Malformed function argument')
                args.append((a[1],a[2].split(' {',1)[0]))
        tail=header[end:]
        ret=tail.split(' attributes ',1)[0].strip()
        if not ret.startswith('->'):returns=[]
        else:
            ret=ret[2:].strip();ret=ret[:-1].rstrip() if ret.endswith('{') else ret
            returns=split_top(ret[1:-1]) if ret.startswith('(') and ret.endswith(')') else [ret]
        stop=starts[ordinal+1] if ordinal+1<len(starts) else len(lines)
        bodylines=[(i+1,lines[i].strip()) for i in range(start+1,stop) if lines[i].strip()]
        # Each function ends with its own closing brace; the last also has the module brace.
        while bodylines and bodylines[-1][1]=='}':bodylines.pop()
        root=[];stack=[root]
        for line,t in bodylines:
            if t.startswith('//'):continue
            if t=='}':
                if len(stack)==1:raise Unsupported('Unexpected region close')
                stack.pop();continue
            if t.startswith('scf.for '):
                q=re.fullmatch(r'scf.for (%[\w]+) = (%[\w]+) to (%[\w]+) step (%[\w]+)(?:\s*:\s*i32)?\s*\{',t)
                if not q:raise Unsupported('Unsupported for form at '+str(line))
                node=Node('for',line,attrs=dict(zip(('iv','lo','hi','step'),q.groups())))
                stack[-1].append(node);stack.append(node.body);continue
            if t.startswith('scf.if '):
                q=re.fullmatch(r'scf.if (%[\w]+)\s*\{',t)
                if not q:raise Unsupported('Unsupported conditional region')
                node=Node('if',line,attrs={'pred':q[1]});stack[-1].append(node);stack.append(node.body);continue
            if t=='} else {':
                if len(stack)<2 or not stack[-2] or stack[-2][-1].kind!='if':
                    raise Unsupported('Else without a matching conditional region')
                conditional=stack[-2][-1]
                if 'else_body' in conditional.attrs:raise Unsupported('Repeated conditional else region')
                conditional.attrs['else_body']=[];stack[-1]=conditional.attrs['else_body'];continue
            if t.startswith('} else'):raise Unsupported('Unsupported conditional else form')
            node=instruction(t,line);stack[-1].append(node);counts[node.kind]=counts.get(node.kind,0)+1
        # Closing braces were removed only at the end, so open trailing nested loops are valid here.
        functions[name]=Function(name,start+1,args,returns,root)
    if not functions:raise Unsupported('No trusted functions')
    return Program(globals,functions,counts,ssa_alpha_renames)


def walk(nodes):
    for n in nodes:
        yield n
        yield from walk(n.body)
        yield from walk(n.attrs.get('else_body',[]))


def refs(text):return re.findall(r'%[\w]+(?:#\d+)?',text)


def constant(node):
    m=re.fullmatch(r'arith.constant ([^ ]+) : (index|i32|i64|f64)',node.text)
    if not m:raise Unsupported('Unsupported constant '+node.text[:100])
    return (float(m[1]) if m[2]=='f64' else int(m[1])),m[2]


def scalar_expr(node,env):
    op=node.kind
    args=refs(node.text)
    if op=='arith.constant':return ('c',constant(node)[0])
    if op=='arith.index_cast':return env[args[0]]
    if op in ('arith.addi','arith.subi','arith.muli','arith.floordivsi','arith.remsi','arith.maxsi','arith.minsi'):
        return (op.rsplit('.',1)[1],env[args[0]],env[args[1]])
    if op=='arith.cmpi':
        pred=node.text.split(' ',1)[1].split(',',1)[0]
        return ('cmp',pred,env[args[0]],env[args[1]])
    raise Unsupported('Unsupported scalar integer expression '+node.text[:100])


def interval(e,bounds):
    if e[0]=='c':return e[1],e[1]
    if e[0]=='v':return bounds[e[1]]
    a,b=interval(e[1],bounds),interval(e[2],bounds)
    if e[0]=='addi':return a[0]+b[0],a[1]+b[1]
    if e[0]=='subi':return a[0]-b[1],a[1]-b[0]
    if e[0]=='muli':v=[x*y for x in a for y in b];return min(v),max(v)
    if e[0] in ('floordivsi','remsi'):
        if b[0]!=b[1] or b[0]<=0:raise Unsupported('A positive constant divisor is required')
        if e[0]=='floordivsi':return a[0]//b[0],a[1]//b[0]
        return -(b[0]-1),b[0]-1
    if e[0]=='maxsi':return max(a[0],b[0]),max(a[1],b[1])
    if e[0]=='minsi':return min(a[0],b[0]),min(a[1],b[1])
    raise Unsupported('Unknown interval expression')


def transfer_region(node,outer):
    nodes=list(walk([node]));loads=[n for n in nodes if n.kind=='memref.load'];stores=[n for n in nodes if n.kind=='memref.store']
    if len(loads)!=1 or len(stores)!=1:raise Unsupported('Sparse transfer requires exactly one scalar load and store at line '+str(node.line))
    load,store=loads[0],stores[0]
    if shape_type(load.text)[1]!='f64':raise Unsupported('Sparse transfer scalar source must be f64')
    lm=re.match(r'memref.load (%[\w]+)\[([^]]*)\]',load.text)
    sm=re.match(r'memref.store (%[\w]+), (%[\w]+)\[([^]]*)\]',store.text)
    if not lm or not sm or sm[1]!=load.defs[0]:raise Unsupported('Sparse transfer requires an unchanged loaded f64 value')
    env=dict(outer);loops=[];guards=[];bounds={};expressions={}
    def visit(n):
        if n.kind=='for':
            a=n.attrs;lo,hi,step=(env[a[x]] for x in ('lo','hi','step'))
            slo,shi=interval(step,bounds)
            if slo!=shi or slo<=0:raise Unsupported('Positive constant loop step required')
            lower,upper=interval(lo,bounds)[0],interval(hi,bounds)[1]-1
            bounds[a['iv']]=(lower,upper);env[a['iv']]=('v',a['iv']);loops.append(dict(iv=a['iv'],lo=lo,hi=hi,step=slo))
        elif n.kind=='if':
            if n.attrs.get('else_body'):raise Unsupported('Sparse transfer else region requires a separate relation proof')
            guards.append(env[n.attrs['pred']])
        elif n.kind.startswith('arith.'):
            e=scalar_expr(n,env);env[n.defs[0]]=e;expressions[n.defs[0]]=e
            if e[0]!='cmp':
                low,high=interval(e,bounds)
                if ': i32' in n.text and not (-(2**31)<=low<=high<2**31):
                    raise Unsupported('i32 overflow range is unproved at line '+str(n.line))
        elif n.kind not in ('memref.load','memref.store'):
            raise Unsupported('Effect inside sparse transfer '+n.kind)
        for child in n.body:visit(child)
    visit(node)
    source_indices=[env[x.strip()] for x in lm[2].split(',')]
    destination_indices=[env[x.strip()] for x in sm[3].split(',')]
    if any(e[0] not in ('v','c') for e in source_indices):raise Unsupported('Source-coordinate inversion is unsupported at line '+str(node.line))
    known={e[1]:i for i,e in enumerate(source_indices) if e[0]=='v'}
    return dict(line=node.line,node=node,source=lm[1],destination=sm[2],load=load.defs[0],source_indices=source_indices,
                destination_indices=destination_indices,loops=loops,guards=guards,bounds=bounds,known=known,
                source_shape=shape_type(load.text)[0],destination_shape=shape_type(store.text)[0])


class _Encoder:
    def __init__(self,model,gp,bounds,suffix):
        self.model,self.gp=model,gp;self.cache={};self.vars={name:model.addVar(lb=lo,ub=hi,vtype=gp.GRB.INTEGER,name=name[1:]+suffix) for name,(lo,hi) in bounds.items()};self.suffix=suffix
    def aux(self):return self.model.addVar(lb=-(2**31),ub=2**31-1,vtype=self.gp.GRB.INTEGER)
    def expr(self,e):
        if e in self.cache:return self.cache[e]
        if e[0]=='c':v=e[1]
        elif e[0]=='v':v=self.vars[e[1]]
        else:
            a,b=self.expr(e[1]),self.expr(e[2]);op=e[0]
            if op=='addi':v=a+b
            elif op=='subi':v=a-b
            elif op=='muli':
                if e[1][0]=='c':v=e[1][1]*b
                elif e[2][0]=='c':v=e[2][1]*a
                else:raise Unsupported('Nonlinear source address')
            elif op in ('minsi','maxsi'):
                av,bv=self.aux(),self.aux();self.model.addConstr(av==a);self.model.addConstr(bv==b);v=self.aux()
                if op=='minsi':self.model.addGenConstrMin(v,[av,bv])
                else:self.model.addGenConstrMax(v,[av,bv])
            elif op in ('floordivsi','remsi'):
                if e[2][0]!='c' or e[2][1]<=0:raise Unsupported('Constant positive divisor required')
                divisor=e[2][1];q=self.aux();rem=self.aux();self.model.addConstr(a==divisor*q+rem)
                if op=='floordivsi':self.model.addConstr(rem>=0);self.model.addConstr(rem<=divisor-1);v=q
                else:
                    sign=self.model.addVar(vtype=self.gp.GRB.BINARY)
                    self.model.addGenConstrIndicator(sign,True,a>=0);self.model.addGenConstrIndicator(sign,False,a<=-1)
                    self.model.addGenConstrIndicator(sign,True,rem>=0);self.model.addGenConstrIndicator(sign,True,rem<=divisor-1)
                    self.model.addGenConstrIndicator(sign,False,rem>=-divisor+1);self.model.addGenConstrIndicator(sign,False,rem<=0);v=rem
            else:raise Unsupported('Unsupported address expression '+op)
        self.cache[e]=v;return v
    def constrain(self,region):
        for loop in region['loops']:
            v=self.vars[loop['iv']];lo,hi=self.expr(loop['lo']),self.expr(loop['hi']);self.model.addConstr(v>=lo);self.model.addConstr(v<=hi-1)
            q=self.aux();self.model.addConstr(v-lo==loop['step']*q)
        for guard in region['guards']:
            if guard[0]!='cmp':raise Unsupported('Unsupported transfer condition')
            _,pred,left,right=guard;a,b=self.expr(left),self.expr(right)
            if pred=='eq':self.model.addConstr(a==b)
            elif pred=='slt':self.model.addConstr(a<=b-1)
            elif pred=='sle':self.model.addConstr(a<=b)
            elif pred=='sgt':self.model.addConstr(a>=b+1)
            elif pred=='sge':self.model.addConstr(a>=b)
            else:raise Unsupported('Unsupported guard comparison '+pred)


def check_functional_destination(region,*,seconds=5):
    """Check bounded integer address functionality with the existing solver.

    Two valid native writes to one target must read the same source coordinate.
    This is an integer relation check, not a cryptographic security assessment.
    """
    import gurobipy as gp
    license_file=Path(gp.__file__).parent/'.libs/gurobi.lic'
    if not license_file.is_file():raise Unsupported('The existing bundled local Gurobi license is required')
    saved={k:v for k,v in os.environ.items() if k.startswith(('GRB_','GUROBI_'))}
    for k in saved:del os.environ[k]
    os.environ['GRB_LICENSE_FILE']=str(license_file)
    env=None;model=None;address_model=None
    try:
        env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
        # Skipping zero-valued stores must also preserve the native program's
        # address validity. Check source and destination bounds for all legal
        # iterations, including iterations whose numerical value will be zero.
        address_model=gp.Model(env=env);address_model.Params.OutputFlag=0
        address_model.Params.Threads=1;address_model.Params.TimeLimit=seconds
        address_model.Params.IntFeasTol=1e-9;address_model.Params.FeasibilityTol=1e-9
        addresses=_Encoder(address_model,gp,region['bounds'],'Bounds');addresses.constrain(region)
        outside=[]
        for indices,shape in ((region['source_indices'],region['source_shape']),
                              (region['destination_indices'],region['destination_shape'])):
            if len(indices)!=len(shape):raise Unsupported('Native copy address rank differs')
            for expression,size in zip(indices,shape):
                if expression[0]=='c':
                    if not 0<=expression[1]<size:raise Unsupported('Native copy has a constant address outside its shape')
                    continue
                value=addresses.expr(expression)
                for condition in (value<=-1,value>=size):
                    bit=address_model.addVar(vtype=gp.GRB.BINARY)
                    address_model.addGenConstrIndicator(bit,True,condition);outside.append(bit)
        if outside:
            address_model.addConstr(gp.quicksum(outside)>=1);address_model.optimize()
            if address_model.Status!=gp.GRB.INFEASIBLE:
                raise Unsupported('Native copy address bounds unproved at line '+str(region['line']))
        address_seconds=address_model.Runtime if outside else 0.0
        model=gp.Model(env=env);model.Params.OutputFlag=0;model.Params.Threads=1;model.Params.TimeLimit=seconds
        model.Params.IntFeasTol=1e-9;model.Params.FeasibilityTol=1e-9
        a=_Encoder(model,gp,region['bounds'],'A');b=_Encoder(model,gp,region['bounds'],'B');a.constrain(region);b.constrain(region)
        for x in region['destination_indices']:model.addConstr(a.expr(x)==b.expr(x))
        alternatives=[]
        for x in region['source_indices']:
            if x[0]=='c':continue
            delta=a.expr(x)-b.expr(x)
            for sign in (-1,1):
                bit=model.addVar(vtype=gp.GRB.BINARY);model.addGenConstrIndicator(bit,True,sign*delta>=1);alternatives.append(bit)
        if not alternatives:passed=True;status='constant_source'
        else:
            model.addConstr(gp.quicksum(alternatives)>=1);model.optimize();status=str(model.Status);passed=model.Status==gp.GRB.INFEASIBLE
        result=dict(line=region['line'],passed=passed,solver='existing gurobipy',solver_status=status,
                    variables=model.NumVars,constraints=model.NumConstrs,general_constraints=model.NumGenConstrs,
                    source_coordinate_functional=True if passed else None,
                    all_native_addresses_within_shape=True,address_check_seconds=address_seconds,
                    integer_solver_queries=int(bool(outside))+int(bool(alternatives)),
                    scope='bounded integer address and functionality infeasibility checks; no data, key or security estimator',seconds=model.Runtime if alternatives else 0.0)
        if not passed:raise Unsupported('Destination source-coordinate relation unproved at line '+str(region['line'])+' (solver status '+status+')')
        return result
    finally:
        if model is not None:model.dispose()
        if address_model is not None:address_model.dispose()
        if env is not None:env.dispose()
        for k in list(os.environ):
            if k.startswith(('GRB_','GUROBI_')):del os.environ[k]
        os.environ.update(saved)
