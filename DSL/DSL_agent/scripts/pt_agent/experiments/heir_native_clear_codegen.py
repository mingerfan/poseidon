"""Generate a bounded, independent mathematical replay of trusted HEIR IR."""
import json
import math
import re

from .heir_native_clear_ir import Unsupported,constant,refs,shape_type,transfer_region,check_functional_destination,walk


RUNTIME=r'''package main
import("encoding/binary";"encoding/hex";"encoding/json";"fmt";"math";"os";"runtime";"runtime/debug";"sort";"strings";"syscall";"time")
type KV struct{index int64;value float64}
type S struct{fill float64;dense []float64;sparse map[int64]float64;building,readonly bool;builder,packed []KV}
type T struct{shape,stride []int64;off int64;s *S;initialized bool;projectedRows map[int64]bool}
type V struct{data []float64;level int}
type A struct{data []any}
type E struct{diag,slot int32;value float64}
type LT struct{entries []E;slots,n1 int}
var count=map[string]int64{};var globals=map[string]*T{};var resources=map[string]*T{};var active="startup";var checkpoint=func(){};var frozenLayouts=[]any{};var captured=map[string]*V{}
func hit(s string){count[s]++}
func mark(s string){active=s;checkpoint()}
func procValue(raw,key string)int64{for _,line:=range strings.Split(raw,"\n"){if strings.HasPrefix(line,key+":"){fields:=strings.Fields(line);if len(fields)<2{panic("resource metadata fields")};var value int64;if _,e:=fmt.Sscan(fields[1],&value);e!=nil{panic(e)};return value}};panic("resource metadata key missing: "+key)}
func physicalGuard(){limit:=int64(4);if raw:=os.Getenv("HEIR_NATIVE_CLEAR_RSS_GIB");raw!=""{if _,e:=fmt.Sscan(raw,&limit);e!=nil{panic(e)}};limitKiB:=limit*1024*1024;if raw:=os.Getenv("HEIR_NATIVE_CLEAR_RSS_MIB");raw!=""{if _,e:=fmt.Sscan(raw,&limitKiB);e!=nil{panic(e)};limitKiB*=1024};for{time.Sleep(100*time.Millisecond);status,e:=os.ReadFile("/proc/self/status");if e!=nil{panic(e)};host,e:=os.ReadFile("/proc/meminfo");if e!=nil{panic(e)};rss,available:=procValue(string(status),"VmRSS"),procValue(string(host),"MemAvailable");if rss>limitKiB||available<1024*1024{record:=map[string]any{"reason":"native mathematical replay resource guard","rss_kib":rss,"rss_limit_kib":limitKiB,"host_available_kib":available,"minimum_host_available_gib":1,"time_utc":time.Now().UTC().Format(time.RFC3339Nano)};raw,_:=json.Marshal(record);os.WriteFile(os.Args[2]+".resource-guard.json",append(raw,'\n'),0600);fmt.Fprintln(os.Stderr,string(raw));os.Exit(125)}}}
func product(a []int64)int64{n:=int64(1);for _,x:=range a{if x<=0||n>math.MaxInt64/x{panic("invalid shape")};n*=x};return n}
func tensor(shape []int64)*T{product(shape);strides:=make([]int64,len(shape));v:=int64(1);for i:=len(shape)-1;i>=0;i--{strides[i]=v;v*=shape[i]};return &T{shape:shape,stride:strides,s:&S{sparse:map[int64]float64{}}}}
func splat(shape []int64,f float64)*T{t:=tensor(shape);t.s.fill=f;t.initialized=true;return t}
func rawTensor(shape []int64,h string)*T{raw,e:=hex.DecodeString(h);if e!=nil{panic(e)};if int64(len(raw))!=product(shape)*8{panic("native dense raw length")};t:=tensor(shape);t.s.dense=make([]float64,len(raw)/8);for i:=range t.s.dense{v:=math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]));if math.IsNaN(v)||math.IsInf(v,0){panic("nonfinite native global")};t.s.dense[i]=v};t.initialized=true;return t}
func (t *T) flat(ix []int64)int64{if len(ix)!=len(t.shape){panic("index rank")};v:=t.off;for i,x:=range ix{if x<0||x>=t.shape[i]{panic(fmt.Sprintf("native index outside shape %v: %v",t.shape,ix))};v+=x*t.stride[i]};return v}
func (t *T) atFlat(i int64)float64{if !t.initialized{panic("uninitialized native tensor read")};if t.s.building{panic("read from unfinished native sparse builder")};if t.s.dense!=nil{return t.s.dense[i]};if t.s.readonly{j:=sort.Search(len(t.s.packed),func(j int)bool{return t.s.packed[j].index>=i});if j<len(t.s.packed)&&t.s.packed[j].index==i{return t.s.packed[j].value};return t.s.fill};if v,ok:=t.s.sparse[i];ok{return v};return t.s.fill}
func (t *T) get(ix ...int64)float64{if t.projectedRows!=nil&&!t.projectedRows[ix[0]]{panic("read outside proven native transform row projection")};return t.atFlat(t.flat(ix))}
func (t *T) project(rows []int){if rows==nil{return};if len(t.shape)!=2{panic("transform row projection rank")};t.projectedRows=map[int64]bool{};t.s.building=true;for _,row:=range rows{if row<0||int64(row)>=t.shape[0]{panic("transform projection bounds")};t.projectedRows[int64(row)]=true}}
func (t *T) set(v float64,ix ...int64){if math.IsNaN(v)||math.IsInf(v,0){panic("nonfinite native tensor store")};i:=t.flat(ix);t.initialized=true;if t.s.readonly{panic("write to frozen native sparse layout")};if t.projectedRows!=nil&&!t.projectedRows[ix[0]]{hit("unconsumed_layout_stores_skipped");return};if t.s.building{t.s.builder=append(t.s.builder,KV{i,v});return};if t.s.dense!=nil{t.s.dense[i]=v;return};if v==t.s.fill{delete(t.s.sparse,i)}else{t.s.sparse[i]=v}}
func (t *T) compact(){if !t.s.building{return};if !t.initialized||t.s.fill!=0||t.s.dense!=nil||len(t.s.sparse)!=0{panic("native sparse builder must own initialized zero storage")};values:=t.s.builder;sort.SliceStable(values,func(i,j int)bool{return values[i].index<values[j].index});written:=0;for begin:=0;begin<len(values);{end:=begin+1;for end<len(values)&&values[end].index==values[begin].index{if values[end].value!=values[begin].value{panic("native sparse builder has different values for one destination")};end++};last:=values[end-1];if last.value!=0{values[written]=last;written++};begin=end};t.s.packed=append([]KV(nil),values[:written]...);t.s.builder=nil;t.s.sparse=nil;t.s.building=false;t.s.readonly=true;hit("native_layouts_frozen");count["native_layout_entries_frozen"]+=int64(written);frozenLayouts=append(frozenLayouts,map[string]any{"function":active,"builder_entries":len(values),"unique_nonzero_entries":written,"stored_index_value_bytes":written*16,"consumer_rows":len(t.projectedRows)});runtime.GC();checkpoint()}
func (t *T) expectZero(){if !t.initialized||t.s.fill!=0||t.s.dense!=nil||len(t.s.sparse)!=0||len(t.s.builder)!=0||t.s.readonly{panic("sparse transfer requires a pristine initialized zero destination")}}
func (t *T) contiguous()bool{v:=int64(1);for i:=len(t.shape)-1;i>=0;i--{if t.stride[i]!=v{return false};v*=t.shape[i]};return t.off==0}
func coordinates(flat int64,shape []int64,out []int64){for i:=len(shape)-1;i>=0;i--{out[i]=flat%shape[i];flat/=shape[i]}}
func (t *T) nonzero(fn func([]int64,float64)){if !t.initialized{panic("uninitialized sparse source")};coords:=make([]int64,len(t.shape));size:=product(t.shape)
 if t.s.building{panic("iterate unfinished native sparse builder")};if t.s.readonly{if !t.contiguous(){panic("unsupported frozen native sparse view")};for _,entry:=range t.s.packed{coordinates(entry.index,t.shape,coords);fn(coords,entry.value);hit("sparse_source_entries")};return}
 if !t.contiguous(){if size>1<<20{panic("unsupported large strided sparse source")};for i:=int64(0);i<size;i++{coordinates(i,t.shape,coords);v:=t.get(coords...);if v!=0{fn(coords,v)}};return}
 if t.s.dense!=nil{for i,v:=range t.s.dense{if v!=0{coordinates(int64(i),t.shape,coords);fn(coords,v)}};return}
 if t.s.fill!=0{if size>1<<20{panic("unsupported large nonzero splat traversal")};for i:=int64(0);i<size;i++{v:=t.atFlat(i);if v!=0{coordinates(i,t.shape,coords);fn(coords,v)}};return}
 keys:=make([]int64,0,len(t.s.sparse));for k:=range t.s.sparse{keys=append(keys,k)};sort.Slice(keys,func(i,j int)bool{return keys[i]<keys[j]});for _,k:=range keys{coordinates(k,t.shape,coords);fn(coords,t.s.sparse[k]);hit("sparse_source_entries")}}
func copyTensor(src,dst *T){if product(src.shape)!=product(dst.shape)||!src.contiguous()||!dst.contiguous(){panic("unsupported native memref copy view")};if !src.initialized||src.s.building{panic("uninitialized/unfinished native memref copy")};if dst.projectedRows!=nil{if len(src.shape)!=2||src.shape[0]!=dst.shape[0]||src.shape[1]!=dst.shape[1]{panic("projected native copy shape")};building:=dst.s.building;dst.s=&S{building:building,sparse:map[int64]float64{}};dst.initialized=true;src.nonzero(func(ix []int64,v float64){dst.set(v,ix...)});return};dst.s=&S{fill:src.s.fill,sparse:map[int64]float64{}};if src.s.readonly{dst.s.readonly=true;dst.s.packed=append([]KV(nil),src.s.packed...)}else if src.s.dense!=nil{dst.s.dense=append([]float64{},src.s.dense...)}else{for k,v:=range src.s.sparse{dst.s.sparse[k]=v}};dst.initialized=true}
func subview(src *T,offset,size,steps []int64)*T{if len(offset)!=len(src.shape)||len(size)!=len(src.shape)||len(steps)!=len(src.shape){panic("subview rank")};t:=&T{s:src.s,initialized:src.initialized,off:src.off};for i,n:=range size{t.off+=offset[i]*src.stride[i];if n!=1{t.shape=append(t.shape,n);t.stride=append(t.stride,src.stride[i]*steps[i])}};if len(t.shape)==0{t.shape=[]int64{1};t.stride=[]int64{1}};return t}
func loadResource(path string,t *T){if previous:=resources[path];previous!=nil{copyTensor(previous,t);hit("resource_cache_hits");return};raw,e:=os.ReadFile(path);if e!=nil{panic(e)};if int64(len(raw))!=product(t.shape)*8{panic("resource shape/length")};t.s.dense=make([]float64,len(raw)/8);for i:=range t.s.dense{v:=math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]));if math.IsNaN(v)||math.IsInf(v,0){panic("nonfinite native resource")};t.s.dense[i]=v};t.initialized=true;resources[path]=t;hit("resource_files_read")}
func vec(t *T,level int)*V{n:=product(t.shape);if n>1<<20{panic("oversized clear vector")};v:=&V{data:make([]float64,n),level:level};ix:=make([]int64,len(t.shape));for i:=range v.data{coordinates(int64(i),t.shape,ix);v.data[i]=t.get(ix...)};return v}
func clone(a *V)*V{return &V{data:append([]float64{},a.data...),level:a.level}}
func identity(a,out *V,kind string)*V{hit("identity_"+kind);if out==nil{return clone(a)};if out!=a{out.data=append(out.data[:0],a.data...)};out.level=a.level;return out}
func binaryVec(a,b,out *V,kind string)*V{if len(a.data)!=len(b.data){panic("native vector length")};if out==nil{out=&V{data:make([]float64,len(a.data))}};if len(out.data)!=len(a.data){out.data=make([]float64,len(a.data))};out.level=min(a.level,b.level);for i,x:=range a.data{y:=b.data[i];switch kind{case "add":out.data[i]=x+y;case "sub":out.data[i]=x-y;case "mul":out.data[i]=float64(x*y);default:panic("unsupported vector binary")};if math.IsNaN(out.data[i])||math.IsInf(out.data[i],0){panic("nonfinite native arithmetic")}};hit(kind);return out}
func rotate(a,out *V,step int64)*V{n:=len(a.data);tmp:=make([]float64,n);for i:=range tmp{j:=(i+int(step))%n;if j<0{j+=n};tmp[i]=a.data[j]};if out==nil{out=&V{}};out.data=tmp;out.level=a.level;hit("rotate");return out}
func decode(a *V,t *T)*T{if int64(len(a.data))!=product(t.shape){panic("decode shape")};t.s=&S{dense:append([]float64{},a.data...)};t.initialized=true;return t}
func nativeBSGS(diags []int,slots,logRatio int)int{if logRatio<0{return 0};maximum:=math.Exp2(float64(logRatio));for n1:=1;n1<slots;n1<<=1{giants,babies:=map[int]bool{},map[int]bool{};for _,d:=range diags{d&=slots-1;giants[(d/n1)*n1]=true;babies[d&(n1-1)]=true};ratio:=float64(len(babies)-1)/float64(len(giants)-1);if ratio==maximum{return n1};if ratio>maximum{return n1/2}};return 1}
func prepareTransform(t *T,diags,rows []int,slots int,logRatios ...int)*LT{if len(t.shape)!=2||t.shape[1]!=int64(slots)||len(diags)!=len(rows)||slots<=0||int64(slots)>math.MaxInt32{panic("native transform layout")};which:=map[int64]int{};for i,row:=range rows{if row<0||int64(row)>=t.shape[0]{panic("native source row")};if t.projectedRows!=nil&&!t.projectedRows[int64(row)]{panic("transform requests unmaterialized row")};which[int64(row)]=diags[i]&(slots-1)};ratio:=0;if len(logRatios)>0{ratio=logRatios[0]};lt:=&LT{slots:slots,n1:nativeBSGS(diags,slots,ratio)};if t.s.readonly&&t.contiguous(){lt.entries=make([]E,0,len(t.s.packed))};t.nonzero(func(ix []int64,v float64){if diag,ok:=which[ix[0]];ok{lt.entries=append(lt.entries,E{diag:int32(diag),slot:int32(ix[1]),value:v})}});sort.Slice(lt.entries,func(i,j int)bool{a,b:=lt.entries[i],lt.entries[j];if a.diag!=b.diag{return a.diag<b.diag};return a.slot<b.slot});hit("linear_transforms_prepared");return lt}
func applyTransform(a *V,lt *LT)*V{if len(a.data)!=lt.slots{panic("transform input slots")};out:=&V{data:make([]float64,lt.slots),level:a.level};partial:=make([]float64,lt.slots);giant:=-1;flush:=func(){if giant<0{return};for i,v:=range partial{out.data[i]=out.data[i]+v;partial[i]=0}};for _,e:=range lt.entries{slot,diag:=int(e.slot),int(e.diag);j:=(slot+diag)%lt.slots;if j<0{j+=lt.slots};term:=float64(e.value*a.data[j]);if lt.n1==0{out.data[slot]=out.data[slot]+term;continue};group:=(diag/lt.n1)*lt.n1;if group!=giant{flush();giant=group};partial[slot]=partial[slot]+term};flush();for _,v:=range out.data{if math.IsNaN(v)||math.IsInf(v,0){panic("nonfinite native linear transformation")}};hit("linear_transforms_applied");return out}
func floorDiv(a,b int64)int64{if b<=0{panic("nonpositive divisor")};q,r:=a/b,a%b;if r<0{q--};return q}
func objectArray(size int64)*A{if size>1<<20{panic("oversized object array")};return &A{data:make([]any,size)}}
func getGlobal(name string)*T{t:=globals[name];if t==nil{panic("unknown global")};return t}
'''


def go_name(name):return 'v_'+name.lstrip('%').replace('#','_part_')
def go_fn(name):return 'f_'+name

def go_type(text):
    if text.startswith('memref'):
        _,element=shape_type(text)
        return '*T' if element=='f64' else '*A'
    if text in ('!pt','!ct'):return '*V'
    if text=='!linear_transformation':return '*LT'
    if text in ('i32','i64','index'):return 'int64'
    if text=='f64':return 'float64'
    return 'any'

def vector(values):return '[]int64{'+','.join(str(x) for x in values)+'}'
def int_vector(values):return '[]int{'+','.join(str(x) for x in values)+'}'


def native_transform_rows(program,go_text):
    matches={m[1]:[int(x) for x in m[2].split(',') if x.strip()] for m in re.finditer(r'([\w]+)_source_rows := \[\]int\{([^}]+)\}',go_text)}
    rows=[]
    go_diags=list(re.finditer(r'([\w]+)_diags_idx := \[\]int\{([^}]+)\}',go_text))
    specs=[n for f in program.functions.values() for n in walk(f.body) if n.kind=='lattigo.ckks.prepare_linear_transform']
    if len(specs)!=len(go_diags):raise Unsupported('Native IR/Go transform count mismatch')
    for n,m in zip(specs,go_diags):
        diags=[int(x) for x in re.search(r'diagonal_indices = \[([^]]*)\]',n.text)[1].split(',') if x.strip()]
        gd=[int(x) for x in m[2].split(',') if x.strip()];sr=matches.get(m[1])
        mapping='explicit_source_rows'
        if sr is None:
            implicit=re.search(r'for i, diagIndex := range '+re.escape(m[1])+r'_diags_idx \{\s*sourceRow := i\s*'+re.escape(m[1])+r'_diags\[diagIndex\]',go_text[m.end():m.end()+500])
            if implicit:sr=list(range(len(gd)));mapping='native_sourceRow_i'
        if diags!=gd or sr is None or len(sr)!=len(diags):raise Unsupported('Native transform row identity is unverified')
        if len(set(diags))!=len(diags) or len(set(sr))!=len(sr):raise Unsupported('Repeated native transform diagonal or source row is unsupported')
        ratio=re.search(r'logBabyStepGiantStepRatio = (-?\d+)',n.text)
        rows.append(dict(line=n.line,diagonals=diags,source_rows=sr,source_go_prefix=m[1],source_row_mapping=mapping,logSlots=int(re.search(r'logSlots = (\d+)',n.text)[1]),logBabyStepGiantStepRatio=int(ratio[1]) if ratio else 0))
    return rows


class Emitter:
    def __init__(self,program,transform_rows,*,prove=True,taps=None):
        self.program=program;self.transforms={r['line']:r for r in transform_rows};self.prove=prove;self.proofs=[];self.release_count=0
        self.taps=taps or {}
        self.projected_calls={};self.projected_helpers=set();self.projection_evidence=[]
        if program is not None:
            for function in program.functions.values():
                uses={};writers={}
                for node in walk(function.body):
                    for value in refs(node.text):uses.setdefault(value,[]).append(node)
                    for value in node.defs:writers[value]=node
                for node in walk(function.body):
                    if node.kind!='lattigo.ckks.prepare_linear_transform':continue
                    value=refs(node.text)[2];writer=writers.get(value)
                    if writer is None or writer.kind not in ('call','func.call'):continue
                    name=re.search(r'@([\w]+)',writer.text)[1]
                    if not name.startswith('_assign_layout_'):continue
                    if uses[value]!=[node]:raise Unsupported('Projected native layout must have exactly one transform consumer')
                    rows=self.transforms[node.line]['source_rows'];helper=program.functions[name]
                    shape,element=shape_type(helper.returns[0])
                    if len(shape)!=2 or element!='f64':raise Unsupported('Projected native layout requires a rank2 f64 helper result')
                    self.projected_calls[(function.name,writer.line)]=rows;self.projected_helpers.add(name)
                    self.projection_evidence.append(dict(helper=name,call_line=writer.line,transform_line=node.line,
                                                        source=value,full_shape=shape,required_source_rows=len(set(rows)),
                                                        complete_access_set_verified=True,
                                                        scope='Only the sole native transform consumer selected source rows are materialized'))
    def emit(self,entry,*,input_shape=None,output_shape=None):
        lines=[RUNTIME]
        for fn in self.program.functions.values():
            if fn.name.endswith('__configure'):continue
            lines.append(self.function(fn))
        initializers=[]
        for name,g in self.program.globals.items():
            rhs='rawTensor('+vector(g['shape'])+','+json.dumps(g['raw_hex'])+')' if 'raw_hex' in g else 'splat('+vector(g['shape'])+','+repr(g['fill'])+')'
            initializers.append('globals['+json.dumps(name)+']='+rhs)
        parts=[initializers[i:i+100] for i in range(0,len(initializers),100)]
        lines.append('func initGlobals(){'+';'.join('initGlobals'+str(i)+'()' for i in range(len(parts)))+'}')
        for i,part in enumerate(parts):
            lines.append('func initGlobals'+str(i)+'(){');lines.extend(part);lines.append('}')
        lines.append(self.main(entry,input_shape=input_shape,output_shape=output_shape))
        return '\n'.join(lines),dict(index_relation_checks=self.proofs,SSA_pointer_releases=self.release_count,
                                     ssa_alpha_renames=self.program.ssa_alpha_renames,
                                     functions_emitted=[f.name for f in self.program.functions.values() if not f.name.endswith('__configure')],
                                     transform_row_projection=self.projection_evidence,
                                     tensor_storage='zero/splat/dense/sparse; source-verified fresh layout append builder frozen to readonly sorted index/value pairs',
                                     linear_transform_arithmetic='native sorted BSGS groups and separate f64 products/additions',
                                     default_go_heap_soft_limit_bytes=2*1024**3,
                                     level_scale_scope='rescale/relinearize/drop/bootstrap are mathematical identities in this diagnostic only',
                                     native_clear_code_generated=True,actual_cryptographic_operations=0)
    def function(self,f):
        self.types={n:go_type(t) for n,t in f.args};self.scalar={};self.returned=set();self.sparse=f.name.startswith('_assign_layout_');self.current_function=f.name
        args=', '.join(go_name(n)+' '+go_type(t) for n,t in f.args)
        projected=f.name in self.projected_helpers
        self.projected_return=None
        if projected:
            returns=[n for n in f.body if n.kind=='return']
            if len(returns)!=1 or len(refs(returns[0].text))!=1:raise Unsupported('Projected helper must return exactly one top-level tensor')
            self.projected_return=refs(returns[0].text)[0]
            if not any(n.kind=='memref.alloc' and n.defs==[self.projected_return] for n in f.body):
                raise Unsupported('Projected helper result must own a fresh native tensor')
            for node in walk(f.body):
                used=refs(node.text)
                if self.projected_return not in used:continue
                if node.kind=='return' and used==[self.projected_return]:continue
                if node.kind=='memref.copy' and len(used)==2 and used[1]==self.projected_return and used[0]!=self.projected_return:continue
                if node.kind=='memref.store' and len(used)>=2 and used[1]==self.projected_return and used[0]!=self.projected_return:continue
                raise Unsupported('Projected helper result has an internal read or alias before its native transform')
            args+=(', ' if args else '')+'projection []int'
        rettypes=[go_type(t) for t in f.returns];ret=rettypes[0] if len(rettypes)==1 else '('+', '.join(rettypes)+')'
        groups=[]
        last={}
        for i,n in enumerate(f.body):
            for q in walk([n]):
                for v in refs(q.text):last[v]=i
                if q.kind=='return':self.returned.update(refs(q.text))
        for i,node in enumerate(f.body):
            group=self.node(node)
            tap=self.taps.get((f.name,node.line))
            if tap is not None:
                if len(node.defs)!=1 or self.types.get(node.defs[0])!='*V':raise Unsupported('Native numerical cut requires exactly one vector SSA result')
                group.append('captured['+json.dumps(tap)+']=clone('+go_name(node.defs[0])+')')
            for v,pos in last.items():
                if pos==i and self.types.get(v) in ('*T','*A','*V','*LT') and v not in self.returned:
                    group.append(go_name(v)+'=nil');self.release_count+=1
            groups.append(group)
        # Large native straight-line functions exceed the local Go compiler's
        # address-space budget. Separate bounded source blocks share the exact
        # native SSA objects; assignments and last-use deletions retain order.
        if len(groups)>400 and not any(n.kind in ('for','if') for n in f.body):
            return self.chunked_function(f,args,ret,rettypes,groups)
        lines=['func '+go_fn(f.name)+'('+args+') '+ret+' {','mark('+json.dumps(f.name)+')']
        lines.extend(line for group in groups for line in group)
        lines.append('}');return '\n'.join(lines)
    def chunked_function(self,f,args,ret,rettypes,groups):
        names={go_name(name):name for name in self.types}
        def reference(match):
            original=names.get(match[0])
            if original is None:raise Unsupported('Unbound chunked native SSA '+match[0])
            expression='env['+json.dumps(original)+']';ty=self.types[original]
            return expression if ty=='any' else expression+'.('+ty+')'
        def rewrite(line):
            if line.startswith('_='):return None
            released=re.fullmatch(r'(v_\w+)=nil',line)
            if released:return 'delete(env,'+json.dumps(names[released[1]])+')'
            if line.startswith('return '):
                values=line[7:].split(',')
                return ';'.join('env['+json.dumps('return:'+str(i))+']='+re.sub(r'\bv_\w+\b',reference,value) for i,value in enumerate(values))
            assigned=re.fullmatch(r'((?:v_\w+,)*v_\w+):=(.*)',line)
            if assigned:
                targets=','.join('env['+json.dumps(names[name])+']' for name in assigned[1].split(','))
                return targets+'='+re.sub(r'\bv_\w+\b',reference,assigned[2])
            return re.sub(r'\bv_\w+\b',reference,line)
        blocks=[]
        for start in range(0,len(groups),200):
            block=[rewrite(line) for group in groups[start:start+200] for line in group]
            blocks.append([line for line in block if line is not None])
        name=go_fn(f.name)
        lines=['func '+name+'('+args+') '+ret+' {','mark('+json.dumps(f.name)+')','env:=map[string]any{'+','.join(json.dumps(n)+':'+go_name(n) for n,_ in f.args)+'}']
        lines.extend(name+'_part'+str(i)+'(env)' for i in range(len(blocks)))
        lines.append('return '+','.join('env['+json.dumps('return:'+str(i))+'].('+ty+')' for i,ty in enumerate(rettypes)))
        lines.append('}')
        for i,block in enumerate(blocks):
            lines.append('func '+name+'_part'+str(i)+'(env map[string]any){');lines.extend(block);lines.append('}')
        return '\n'.join(lines)
    def assign(self,n,expression,kind):
        if len(n.defs)!=1:raise Unsupported('Expected single native result at line '+str(n.line))
        name=n.defs[0];self.types[name]=kind;return [go_name(name)+':='+expression,'_='+go_name(name)]
    def node(self,n,bound=None):
        op=n.kind;vs=refs(n.text);v=lambda i:go_name(vs[i]);out=[]
        if op=='for':
            if self.sparse and bound is None:
                region=transfer_region(n,self.scalar)
                if self.prove:self.proofs.append(check_functional_destination(region))
                out=[go_name(region['destination'])+'.expectZero()',go_name(region['source'])+'.nonzero(func(co []int64,value float64){']
                for axis,e in enumerate(region['source_indices']):
                    if e[0]=='c':out.append('if co['+str(axis)+']!='+str(e[1])+' {return}')
                    elif region['known'][e[1]]!=axis:out.append('if co['+str(axis)+']!=co['+str(region['known'][e[1]])+'] {return}')
                out.extend(self.node(n,region));out.append('})');return out
            a=n.attrs;iv=go_name(a['iv']);lo,hi,step=(go_name(a[x]) for x in ('lo','hi','step'))
            if bound is not None and a['iv'] in bound['known']:
                out=[iv+':=co['+str(bound['known'][a['iv']])+']','if '+iv+'>='+lo+'&&'+iv+'<'+hi+'&&('+iv+'-'+lo+')%'+step+'==0 {']
            else:out=['for '+iv+':='+lo+';'+iv+'<'+hi+';'+iv+'+='+step+' {']
            for child in n.body:out.extend(self.node(child,bound))
            out.append('}');return out
        if op=='if':
            out=['if '+go_name(n.attrs['pred'])+' {']
            for child in n.body:out.extend(self.node(child,bound))
            if 'else_body' in n.attrs:
                out.append('}else{')
                for child in n.attrs['else_body']:out.extend(self.node(child,bound))
            out.append('}');return out
        if op=='arith.constant':
            value,ty=constant(n);self.scalar[n.defs[0]]=('c',value)
            return self.assign(n,('float64('+repr(value)+')') if ty=='f64' else 'int64('+str(value)+')','float64' if ty=='f64' else 'int64')
        if op.startswith('arith.'):
            a,b=(v(0),v(1)) if len(vs)>1 else (v(0),None)
            if op=='arith.index_cast':expr=a;ty='int64'
            elif op=='arith.cmpi':
                pred=n.text.split(' ',1)[1].split(',',1)[0];sign={'eq':'==','ne':'!=','slt':'<','sle':'<=','sgt':'>','sge':'>='}.get(pred)
                if sign is None:raise Unsupported('Integer predicate '+pred)
                expr=a+sign+b;ty='bool'
            elif op in ('arith.addi','arith.subi','arith.muli','arith.remsi'):
                expr=a+{'arith.addi':'+','arith.subi':'-','arith.muli':'*','arith.remsi':'%'}[op]+b;ty='int64'
            elif op=='arith.floordivsi':expr='floorDiv('+a+','+b+')';ty='int64'
            elif op in ('arith.maxsi','arith.minsi'):expr=('max' if op.endswith('maxsi') else 'min')+'('+a+','+b+')';ty='int64'
            elif op in ('arith.andi','arith.ori') and n.text.endswith(' : i1'):
                expr=a+('&&' if op=='arith.andi' else '||')+b;ty='bool'
            else:raise Unsupported('Integer operation '+op)
            return self.assign(n,expr,ty)
        if op=='memref.get_global':return self.assign(n,'getGlobal('+json.dumps(re.search(r'@([\w]+)',n.text)[1])+')','*T')
        if op=='memref.alloc':
            shape,elem=shape_type(n.text);out=self.assign(n,'tensor('+vector(shape)+')' if elem=='f64' else 'objectArray('+str(math.prod(shape))+')','*T' if elem=='f64' else '*A')
            if n.defs==[self.projected_return]:out.append(go_name(n.defs[0])+'.project(projection)')
            return out
        if op=='memref.copy':return ['copyTensor('+v(0)+','+v(1)+')']
        if op=='preprocessing.load_resource':return ['loadResource('+json.dumps(re.search(r'"([^"]+)"',n.text)[1])+','+v(0)+')']
        if op=='memref.cast':return self.assign(n,v(0),self.types[vs[0]])
        if op=='memref.load':
            ix=re.search(r'\[([^]]*)\]',n.text)[1];indices=[go_name(x.strip()) for x in ix.split(',')];_,elem=shape_type(n.text)
            if bound is not None:return self.assign(n,'value','float64')
            if elem=='f64':return self.assign(n,v(0)+'.get('+','.join(indices)+')','float64')
            ty='*LT' if elem=='!linear_transformation' else '*V';return self.assign(n,v(0)+'.data['+indices[0]+'].('+ty+')',ty)
        if op=='memref.store':
            m=re.match(r'memref.store (%\w+), (%\w+)\[([^]]*)\]',n.text);indices=[go_name(x.strip()) for x in m[3].split(',')]
            _,elem=shape_type(n.text)
            return [go_name(m[2])+('.set('+go_name(m[1])+','+','.join(indices)+')' if elem=='f64' else '.data['+indices[0]+']='+go_name(m[1]))]
        if op=='memref.subview':
            m=re.match(r'memref.subview (%\w+)\[([^]]*)\] \[([^]]*)\] \[([^]]*)\]',n.text)
            if not m:raise Unsupported('Subview syntax')
            seq=lambda s:'[]int64{'+','.join(go_name(x.strip()) if x.strip().startswith('%') else x.strip() for x in s.split(','))+'}'
            return self.assign(n,'subview('+go_name(m[1])+','+','.join(seq(m[i]) for i in (2,3,4))+')','*T')
        if op in ('call','func.call'):
            m=re.search(r'@([\w]+)\(([^)]*)\)',n.text);f=self.program.functions.get(m[1])
            if f is None:raise Unsupported('Unknown native call')
            arguments=[go_name(x) for x in refs(m[2])]
            if f.name in self.projected_helpers:
                projected=self.projected_calls.get((self.current_function,n.line))
                arguments.append('nil' if projected is None else int_vector(projected))
            expr=go_fn(f.name)+'('+','.join(arguments)+')'
            if len(n.defs)==1:return self.assign(n,expr,go_type(f.returns[0]))
            if len(n.defs)!=len(f.returns):raise Unsupported('Call return arity')
            for name,ty in zip(n.defs,f.returns):self.types[name]=go_type(ty)
            return [','.join(go_name(x) for x in n.defs)+':='+expr]+['_='+go_name(x) for x in n.defs]
        if op=='return':return ([go_name(vs[0])+'.compact()'] if vs==[self.projected_return] else [])+['return '+','.join(go_name(x) for x in vs)]
        if op=='lattigo.ckks.new_plaintext':return self.assign(n,'&V{level:'+re.search(r'level = (\d+)',n.text)[1]+'}','*V')
        if op=='lattigo.ckks.encode':return self.assign(n,'vec('+v(1)+','+v(2)+'.level)','*V')
        if op=='lattigo.rlwe.encrypt':return self.assign(n,'identity('+v(1)+',nil,"client_pack")','*V')
        if op=='lattigo.rlwe.decrypt':return self.assign(n,'identity('+v(1)+',nil,"client_unpack")','*V')
        if op=='lattigo.ckks.decode':return self.assign(n,'decode('+v(1)+','+v(2)+')','*T')
        if op=='lattigo.ckks.prepare_linear_transform':
            spec=self.transforms.get(n.line)
            if spec is None:raise Unsupported('No actual native source-row mapping')
            return self.assign(n,'prepareTransform('+v(2)+','+int_vector(spec['diagonals'])+','+int_vector(spec['source_rows'])+','+str(2**spec['logSlots'])+','+str(spec.get('logBabyStepGiantStepRatio',0))+')','*LT')
        if op=='lattigo.ckks.apply_linear_transform':return self.assign(n,'applyTransform('+v(1)+','+v(2)+')','*V')
        if op in ('lattigo.ckks.add','lattigo.ckks.sub','lattigo.ckks.mul','lattigo.ckks.mul_new'):
            kind=op.rsplit('.',1)[1].removesuffix('_new');target='nil' if op.endswith('_new') else v(3)
            return self.assign(n,'binaryVec('+v(1)+','+v(2)+','+target+','+json.dumps(kind)+')','*V')
        if op=='lattigo.ckks.rotate':return self.assign(n,'rotate('+v(1)+','+v(2)+','+v(3)+')','*V')
        if op=='lattigo.ckks.rotate_new':return self.assign(n,'rotate('+v(1)+',nil,'+v(2)+')','*V')
        if op in ('lattigo.ckks.relinearize','lattigo.ckks.rescale','lattigo.rlwe.drop_level','lattigo.rlwe.drop_level_new','lattigo.ckks.bootstrap'):
            name=op.rsplit('.',1)[1];target=v(2) if len(vs)==3 and name!='bootstrap' else 'nil'
            return self.assign(n,'identity('+v(1)+','+target+','+json.dumps(name)+')','*V')
        raise Unsupported('Unsupported native operation '+op+' at line '+str(n.line))
    def main(self,entry,*,input_shape=None,output_shape=None):
        names=[entry+'__preprocessing',entry+'__preprocessed',entry+'__encrypt__arg0',entry+'__decrypt__result0']
        if any(n not in self.program.functions for n in names):raise Unsupported('Complete native interface functions are required')
        source=r'''
type Sample struct{SampleID string `json:"sample_id"`;Input []float64 `json:"input"`;Reference []float64 `json:"reference"`}
func main(){
 runtime.GOMAXPROCS(1);heapGiB:=int64(2);if raw:=os.Getenv("HEIR_NATIVE_CLEAR_HEAP_GIB");raw!=""{if _,e:=fmt.Sscan(raw,&heapGiB);e!=nil{panic(e)}};debug.SetMemoryLimit(heapGiB<<30)
 result:=map[string]any{"status":"running","scope":"independent clear mathematical replay of saved native IR; level changes/bootstrap identities; actual CKKS0","keygen_runs":0,"encryptions":0,"bootstraps":0,"decryptions":0,"native_math_attempts":0,"native_math_completed":0,"math_probes_completed":0,"samples":[]any{},"counts":count,"active_phase":active,"go_heap_soft_limit_bytes":heapGiB<<30}
 started:=time.Now();if len(os.Args)!=3{panic("saved samples and result paths required")}
 checkpoint=func(){var usage syscall.Rusage;syscall.Getrusage(syscall.RUSAGE_SELF,&usage);result["ru_maxrss_kib"]=usage.Maxrss;result["elapsed_seconds"]=time.Since(started).Seconds();result["active_phase"]=active;result["frozen_layouts"]=frozenLayouts;raw,e:=json.Marshal(result);if e!=nil{panic(e)};temporary:=os.Args[2]+".tmp";if e=os.WriteFile(temporary,append(raw,'\n'),0600);e!=nil{panic(e)};if e=os.Rename(temporary,os.Args[2]);e!=nil{panic(e)}}
 defer func(){if value:=recover();value!=nil{result["status"]="error";result["error"]=fmt.Sprint(value)};checkpoint()}();checkpoint();go physicalGuard()
 raw,e:=os.ReadFile(os.Args[1]);if e!=nil{panic(e)};var samples []Sample;if e=json.Unmarshal(raw,&samples);e!=nil{panic(e)};if len(samples)!=3{panic("exactly three saved training probes required")}
 mark("native_globals");initGlobals();mark("native_preprocessing")
 pts,lts:=PREPROCESS(nil,nil);result["preprocessing_completed"]=true
 rows:=[]any{}
 for _,sample:=range samples{
  result["native_math_attempts"]=result["native_math_attempts"].(int)+1;result["active_sample_id"]=sample.SampleID;mark(sample.SampleID)
  if len(sample.Input)!=3072||len(sample.Reference)!=10{panic("saved probe shape")}
  input:=tensor([]int64{1,3,32,32});input.s.dense=sample.Input;input.initialized=true
  packed:=PACK(nil,nil,nil,nil,input)
  output:=EVALUATE(nil,nil,nil,nil,packed,pts,lts)
  logits:=UNPACK(nil,nil,nil,nil,output)
  result["native_math_completed"]=result["native_math_completed"].(int)+1
  maximum,ratio:=0.0,0.0;passed:=true;observed:=make([]float64,10)
  for i,want:=range sample.Reference{got:=logits.get(0,int64(i));observed[i]=got;if math.IsNaN(got)||math.IsInf(got,0){panic("nonfinite native math logits")};delta:=math.Abs(got-want);bound:=1e-12+1e-12*math.Abs(want);maximum=math.Max(maximum,delta);ratio=math.Max(ratio,delta/bound);passed=passed&&delta<=bound}
  rows=append(rows,map[string]any{"sample_id":sample.SampleID,"reference":sample.Reference,"observed":observed,"max_absolute_error":maximum,"max_tolerance_ratio":ratio,"passed":passed});result["samples"]=rows
  checkpoint()
  if !passed{result["status"]="math_failed";return};result["math_probes_completed"]=result["math_probes_completed"].(int)+1
  runtime.GC()
 }
 active="complete";result["status"]="ok"
}
'''.replace('PREPROCESS',go_fn(names[0])).replace('EVALUATE',go_fn(names[1])).replace('UNPACK(',go_fn(names[3])+'(').replace('PACK(',go_fn(names[2])+'(')
        if input_shape is not None or output_shape is not None:
            if (not input_shape or not output_shape or input_shape[0]!=1
                    or len(output_shape)!=2 or output_shape[0]!=1
                    or any(type(n) is not int or n<=0 for n in (*input_shape,*output_shape))):
                raise Unsupported('Common-model runtime requires positive static batch-one shapes')
            import math
            source=source.replace('len(sample.Input)!=3072||len(sample.Reference)!=10',
                'len(sample.Input)!='+str(math.prod(input_shape))+'||len(sample.Reference)!='+str(output_shape[1]))
            source=source.replace('input:=tensor([]int64{1,3,32,32})','input:=tensor('+vector(input_shape)+')')
            source=source.replace('observed:=make([]float64,10)','observed:=make([]float64,'+str(output_shape[1])+')')
        return source
