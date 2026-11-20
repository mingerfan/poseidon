// Package qwengeneration compiles a constructor-free native component client.
// Original RuntimePlans/bundles are loaded separately and never rewritten.
// This library owns no parameter/key/encryption/decryption constructor.
package qwengeneration

import (
 "encoding/binary"
 "encoding/json"
 "fmt"
 "math"
 "os"
 "path/filepath"
 "regexp"
 "strings"

 "github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
 "github.com/tuneinsight/lattigo/v6/core/rlwe"
 "github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const Slots = 16384
const ContextID = "qwen-bootstrap48"
const BootstrapProfile = "cpu-bootstrap48-level1-to5"

type Place struct { Kind string `json:"kind"`; Rank int `json:"rank"` }
type Value struct {
 ID string `json:"id"`; Kind string `json:"kind"`; Context string `json:"context"`
 Level int `json:"level"`; Scale int `json:"scale_log2"`; Components int `json:"components"`
 NTT bool `json:"ntt"`; Place Place `json:"place"`
}
type Payload struct { Kind string `json:"kind"`; Content string `json:"content"`; Values []float64 `json:"values"` }
type Instruction struct {
 Kind string `json:"kind"`; Ordinal int `json:"ordinal"`; Op string `json:"op"`
 Inputs []string `json:"inputs"`; Output string `json:"output"`; Place Place `json:"place"`
 Attrs struct {
  Steps int `json:"steps"`; Level int `json:"target_level"`; Scale int `json:"target_scale_log2"`
  Components int `json:"target_components"`; Implementation string `json:"implementation"`
  OperatorProfile string `json:"operator_profile"`
 } `json:"attrs"`
 Payload Payload `json:"payload"`
}
type Plan struct {
 Version int `json:"format_version"`; Target json.RawMessage `json:"target"`
 Values []Value `json:"values"`; Inputs []string `json:"external_inputs"`
 Initialization []Instruction `json:"initialization"`; Execution []Instruction `json:"execution"`
 Finalization []Instruction `json:"finalization"`; Outputs []string `json:"final_outputs"`
 Bundle struct { ID string `json:"id"`; Version int `json:"version"` } `json:"plaintext_bundle"`
}
type Blob struct { Content string `json:"content"`; Bytes int `json:"byte_length"` }
type Manifest struct {
 Format int `json:"bundle_format_version"`; ID string `json:"bundle_id"`
 Version int `json:"version"`; Blobs []Blob `json:"blobs"`
}
// Source binding proof is admitted by the future external27-artifact gate.
// A nonempty source path here is structural identity, not independent C2 proof.
type ComponentRef struct {
 Role string; Index int; PlanPath string; BundlePath string; SourceCase string
}
type Component struct {
 reference ComponentRef; plan Plan; values map[string]Value
 encodes map[string]Payload; last map[string]int
}
type Program struct {
 Layers [24]ComponentRef; Head ComponentRef; First ComponentRef; Terminal ComponentRef
 SourceCase string; PrefixLength int
}
type SharedContext struct {
 params ckks.Parameters; encoder *ckks.Encoder; evaluator *ckks.Evaluator
 bootstrapper *bootstrapping.Evaluator; nextSerial uint64
}
type Handle struct { value *rlwe.Ciphertext; owner *SharedContext; descriptor Value; serial uint64 }
type inputSnapshot struct { metadata rlwe.MetaData; level,degree int }
func snapshotInput(ct *rlwe.Ciphertext) inputSnapshot {
 m:=*ct.MetaData;m.Scale=rlwe.NewScale(ct.Scale)
 return inputSnapshot{metadata:m,level:ct.Level(),degree:ct.Degree()}
}
func inputUnchanged(ct *rlwe.Ciphertext,before inputSnapshot)bool{
 return ct.Level()==before.level&&ct.Degree()==before.degree&&ct.Scale.Equal(before.metadata.Scale)&&
  ct.IsNTT==before.metadata.IsNTT&&ct.IsMontgomery==before.metadata.IsMontgomery&&
  ct.IsBatched==before.metadata.IsBatched&&ct.LogDimensions==before.metadata.LogDimensions
}

// BorrowSharedContext stores already supplied native objects. It creates none.
// The existing50-bit adapter is deliberately outside this raw43 library.
func BorrowSharedContext(p ckks.Parameters, encoder *ckks.Encoder, evaluator *ckks.Evaluator, bootstrapper *bootstrapping.Evaluator) (*SharedContext,error) {
 if p.N()!=65536 || p.MaxLevel()<5 || p.DefaultScale().Log2()!=43 || encoder==nil || evaluator==nil || bootstrapper==nil {
  return nil,fmt.Errorf("unsupported: require borrowed raw43 N16 evaluator/encoder/native bootstrapper")
 }
 return &SharedContext{params:p,encoder:encoder,evaluator:evaluator,bootstrapper:bootstrapper},nil
}
func sameType(a,b Value) bool {
 return a.Kind==b.Kind && a.Context==b.Context && a.Level==b.Level && a.Scale==b.Scale && a.Components==b.Components && a.NTT==b.NTT && a.Place==b.Place
}
func validPlace(p Place) bool { return p.Kind=="host" && p.Rank==0 }
func validValue(v Value) bool {
 return v.ID!="" && v.Context==ContextID && validPlace(v.Place) && v.NTT && v.Level>=0 && v.Level<=8 && v.Scale>=0 && v.Scale<=1023 &&
  ((v.Kind=="plaintext" && v.Components==1)||(v.Kind=="ciphertext" && (v.Components==2||v.Components==3)))
}
func boundaryValue(id string) Value { return Value{ID:id,Kind:"ciphertext",Context:ContextID,Level:5,Scale:43,Components:2,NTT:true,Place:Place{Kind:"host",Rank:0}} }
func validateCipher(ct *rlwe.Ciphertext, v Value, predicted rlwe.Scale) error {
 if ct==nil || ct.Level()!=v.Level || ct.Degree()+1!=v.Components || !ct.IsNTT || ct.LogDimensions.Cols!=14 {
  return fmt.Errorf("ciphertext metadata differs at %s",v.ID)
 }
 if ct.Scale.Cmp(predicted)!=0 { return fmt.Errorf("actual native scale recurrence differs at %s",v.ID) }
 return nil
}
func (s *SharedContext) issue(ct *rlwe.Ciphertext,v Value) Handle {
 s.nextSerial++;return Handle{value:ct,owner:s,descriptor:v,serial:s.nextSerial}
}
func (s *SharedContext) require(h Handle,v Value) error {
 if h.owner!=s || h.serial==0 || !sameType(h.descriptor,v) { return fmt.Errorf("foreign or incompatible typed ciphertext handle") }
 return validateCipher(h.value,v,h.value.Scale)
}
// BindInitial borrows caller ciphertexts. No plaintext input or key is loaded.
func (s *SharedContext) BindInitial(raw [34]*rlwe.Ciphertext) (out [34]Handle,err error) {
 for i,ct:=range raw {
  v:=boundaryValue(fmt.Sprintf("original:%d",i))
  if err=validateCipher(ct,v,s.params.DefaultScale());err!=nil{return}
  out[i]=s.issue(ct,v)
 }
 return
}
func (h Handle) Ciphertext() *rlwe.Ciphertext { return h.value }

func arity(role string,index int)(inputs,outputs int,err error) {
 switch role {
 case "layer":if index<0||index>=24{return 0,0,fmt.Errorf("invalid original layer index")};return 34,32,nil
 case "head":return 34,19,nil
 case "first":return 20,22,nil
 case "terminal":return 19,19,nil
 default:return 0,0,fmt.Errorf("unsupported component role")
 }
}
func readJSON(path string,dest any) error {
 file,err:=os.Open(path);if err!=nil{return err};defer file.Close()
 return json.NewDecoder(file).Decode(dest)
}
var safeLeaf=regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
func blobPath(bundle,content string)(string,error) {
 parts:=strings.SplitN(content,":",2)
 if len(parts)!=2 || !safeLeaf.MatchString(parts[1]) { return "",fmt.Errorf("unsafe opaque bundle locator") }
 return filepath.Join(bundle,"data",parts[1]+".bin"),nil
}
// Original r31 full-slot float64 representation; no digest or payload rewrite.
func readVector(path string)([]float64,error) {
 raw,err:=os.ReadFile(path);if err!=nil{return nil,err}
 if len(raw)!=Slots*8{return nil,fmt.Errorf("unsupported: require original full-slot float64 bundle vector")}
 vector:=make([]float64,Slots)
 for i:=range vector {
  vector[i]=math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:]))
  if math.IsNaN(vector[i])||math.IsInf(vector[i],0){return nil,fmt.Errorf("nonfinite original bundle vector")}
 }
 return vector,nil
}
// Validate every original symbolic operation before a native call is allowed.
func validateOperation(op Instruction,values map[string]Value) error {
 arities:=map[string]int{"mod_switch":1,"rotate":1,"mul_cp":2,"rescale":1,"add_cc":2,"add_cp":2,"boot":1,"mul_cc":2,"relinearize":1,"negate":1}
 n,ok:=arities[op.Op]
 if !ok||op.Kind!="compute"||len(op.Inputs)!=n||!validPlace(op.Place){return fmt.Errorf("unsupported compute instruction")}
 a,ok:=values[op.Inputs[0]];out,outOK:=values[op.Output]
 if !ok||!outOK||a.Kind!="ciphertext"||out.Kind!="ciphertext" {return fmt.Errorf("missing ciphertext operand/output")}
 level,scale,components:=a.Level,a.Scale,a.Components
 if n==2 {
  b,exists:=values[op.Inputs[1]];kind:="ciphertext";if strings.HasSuffix(op.Op,"_cp"){kind="plaintext"}
  if !exists||b.Kind!=kind||b.Level!=level||b.Context!=a.Context{return fmt.Errorf("binary operand types differ")}
  if strings.HasPrefix(op.Op,"add") {
   if b.Scale!=scale{return fmt.Errorf("nominal addition scales differ")};components=max(components,b.Components)
  } else {scale+=b.Scale;components+=b.Components-1}
 }
 switch op.Op {
 case "mod_switch":if op.Attrs.Level<0||op.Attrs.Level>level{return fmt.Errorf("invalid level drop")};level=op.Attrs.Level
 case "rescale":if op.Attrs.Level!=level-1||op.Attrs.Scale!=scale-43{return fmt.Errorf("original rescale must consume one nominal43-bit prime")};level--;scale-=43
 case "relinearize":if components!=3{return fmt.Errorf("relinearize requires3 components")};components=2
 case "rotate":if components!=2{return fmt.Errorf("rotate requires2 components")}
 case "boot":
  if level!=1||scale!=43||components!=2||op.Attrs.Implementation!="native"||op.Attrs.OperatorProfile!=BootstrapProfile||op.Attrs.Level!=5||op.Attrs.Scale!=43||op.Attrs.Components!=2{return fmt.Errorf("original native bootstrap contract differs")}
  level,scale,components=5,43,2
 }
 if out.Level!=level||out.Scale!=scale||out.Components!=components{return fmt.Errorf("symbolic output equation differs at ordinal%d",op.Ordinal)}
 return nil
}

func LoadComponent(ref ComponentRef)(*Component,error) {
 ni,no,err:=arity(ref.Role,ref.Index);if err!=nil{return nil,err}
 if ref.SourceCase==""||ref.PlanPath==""{return nil,fmt.Errorf("original component source identity required")}
 c:=&Component{reference:ref,values:map[string]Value{},encodes:map[string]Payload{},last:map[string]int{}}
 if err=readJSON(ref.PlanPath,&c.plan);err!=nil{return nil,err}
 p:=&c.plan
 if p.Version!=1||len(p.Inputs)!=ni||len(p.Outputs)!=no||len(p.Finalization)!=0{return nil,fmt.Errorf("actual component ABI differs")}
 for _,v:=range p.Values {
  if _,exists:=c.values[v.ID];exists||!validValue(v){return nil,fmt.Errorf("invalid value metadata")};c.values[v.ID]=v
 }
 seen:=map[string]bool{};ordinals:=map[int]bool{};bootOutputs:=map[string]bool{}
 for _,id:=range p.Inputs {
  v,exists:=c.values[id]
  if !exists||seen[id]||!sameType(v,boundaryValue(id)){return nil,fmt.Errorf("raw43 level5 component input differs")};seen[id]=true
 }
 manifest:=map[string]Blob{};needsBundle:=false
 for _,op:=range p.Initialization{needsBundle=needsBundle||op.Payload.Kind=="bundle"}
 if needsBundle {
  var m Manifest;if err=readJSON(filepath.Join(ref.BundlePath,"manifest.json"),&m);err!=nil{return nil,err}
  if m.Format!=1||m.Version!=1||p.Bundle.Version!=1||m.ID!=p.Bundle.ID{return nil,fmt.Errorf("bundle metadata differs")}
  for _,b:=range m.Blobs {
   if _,exists:=manifest[b.Content];exists||b.Bytes!=Slots*8{return nil,fmt.Errorf("unsupported or duplicate original blob shape")}
   path,e:=blobPath(ref.BundlePath,b.Content);if e!=nil{return nil,e}
   info,e:=os.Lstat(path);if e!=nil{return nil,e};if !info.Mode().IsRegular()||info.Size()!=int64(b.Bytes){return nil,fmt.Errorf("bundle file closure differs")}
   manifest[b.Content]=b
  }
 }
 for _,op:=range p.Initialization {
  v,exists:=c.values[op.Output]
  // Original encode instructions carry no place field. Their output Value
  // already supplies the required host/rank0/NTT/context descriptor.
  if op.Kind!="encode"||!exists||v.Kind!="plaintext"||seen[op.Output]||ordinals[op.Ordinal]{return nil,fmt.Errorf("invalid original initialization")}
  switch op.Payload.Kind {
  case "bundle":if _,ok:=manifest[op.Payload.Content];!ok{return nil,fmt.Errorf("unbound original plaintext blob")}
  case "inline":if len(op.Payload.Values)!=1||math.IsNaN(op.Payload.Values[0])||math.IsInf(op.Payload.Values[0],0){return nil,fmt.Errorf("unsupported: only finite singleton inline payload")}
  default:return nil,fmt.Errorf("unsupported original plaintext payload")
  }
  c.encodes[op.Output]=op.Payload;seen[op.Output]=true;ordinals[op.Ordinal]=true
 }
 for n,op:=range p.Execution {
  if seen[op.Output]||ordinals[op.Ordinal]{return nil,fmt.Errorf("native SSA/ordinal redefinition")}
  for _,id:=range op.Inputs {if !seen[id]{return nil,fmt.Errorf("native SSA operand has no producer")};c.last[id]=n}
  if err=validateOperation(op,c.values);err!=nil{return nil,err};seen[op.Output]=true;ordinals[op.Ordinal]=true
  if op.Op=="boot"{bootOutputs[op.Output]=true}
 }
 final:=map[string]bool{}
 for _,id:=range p.Outputs {
  v,exists:=c.values[id]
  if !exists||!seen[id]||final[id]||v.Kind!="ciphertext"||v.Components!=2||v.Scale!=43||v.Level<1||v.Level>5{return nil,fmt.Errorf("invalid complete final output metadata")}
  if (ref.Role=="layer"||ref.Role=="head")&&(!sameType(v,boundaryValue(id))||!bootOutputs[id]){return nil,fmt.Errorf("prefill output must retain actual normalized level5 final boot")}
  final[id]=true
 }
 if len(seen)!=len(c.values){return nil,fmt.Errorf("unproduced native values")}
 return c,nil
}

// One cache per invocation prevents equal numeric IDs in different components
// from sharing a plaintext. Only a single encoded plaintext remains resident.
type plaintextCache struct{ shared *SharedContext; component *Component; id string; current *rlwe.Plaintext }
func(c *plaintextCache)get(id string)(*rlwe.Plaintext,error){
 if c.current!=nil&&c.id==id{return c.current,nil}
 c.current=nil;payload,ok:=c.component.encodes[id];if !ok{return nil,fmt.Errorf("missing plaintext payload")}
 var raw []float64;var err error
 if payload.Kind=="inline" {raw=make([]float64,Slots);for i:=range raw{raw[i]=payload.Values[0]}} else {
  path,e:=blobPath(c.component.reference.BundlePath,payload.Content);if e!=nil{return nil,e};raw,err=readVector(path);if err!=nil{return nil,err}
 }
 v:=c.component.values[id];pt:=ckks.NewPlaintext(c.shared.params,v.Level)
 pt.Scale=rlwe.NewScale(math.Exp2(float64(v.Scale)));pt.LogDimensions.Cols=14
 if err=c.shared.encoder.Encode(raw,pt);err!=nil{return nil,err}
 c.id=id;c.current=pt;return pt,nil
}

func(s *SharedContext)nativeBoot(a *rlwe.Ciphertext)(*rlwe.Ciphertext,error){
 if a==nil||a.Level()!=1||s.bootstrapper==nil{return nil,fmt.Errorf("native level1 bootstrap unavailable")}
 beforeLevel,beforeScale:=a.Level(),rlwe.NewScale(a.Scale)
 input:=a.CopyNew() // Original r31 copy protects all later original SSA uses.
 out,err:=s.bootstrapper.Bootstrap(input);if err!=nil{return nil,err}
 if a.Level()!=beforeLevel||a.Scale.Cmp(beforeScale)!=0{return nil,fmt.Errorf("native bootstrap mutated original input metadata")}
 if out.Level()<5{return nil,fmt.Errorf("actual native bootstrap output below5")}
 s.evaluator.DropLevel(out,out.Level()-5)
 return out,nil
}

func(s *SharedContext)RunComponent(c *Component,inputs []Handle)([]Handle,error){
 if c==nil||len(inputs)!=len(c.plan.Inputs){return nil,fmt.Errorf("component input arity differs")}
 pool:=map[string]*rlwe.Ciphertext{}
 for i,id:=range c.plan.Inputs{if err:=s.require(inputs[i],c.values[id]);err!=nil{return nil,err};pool[id]=inputs[i].value}
 cache:=plaintextCache{shared:s,component:c};final:=map[string]bool{};for _,id:=range c.plan.Outputs{final[id]=true}
 for n,op:=range c.plan.Execution {
  a:=pool[op.Inputs[0]];if a==nil{return nil,fmt.Errorf("missing live ciphertext")}
  if err:=validateCipher(a,c.values[op.Inputs[0]],a.Scale);err!=nil{return nil,err}
  var b *rlwe.Ciphertext;var pt *rlwe.Plaintext;var err error
  if len(op.Inputs)==2 {
   id:=op.Inputs[1]
   if c.values[id].Kind=="plaintext" {pt,err=cache.get(id);if err!=nil{return nil,err}} else {
    b=pool[id];if b==nil{return nil,fmt.Errorf("missing second ciphertext")};if err=validateCipher(b,c.values[id],b.Scale);err!=nil{return nil,err}
   }
  }
  v:=c.values[op.Output];var out *rlwe.Ciphertext;predicted:=a.Scale
  aBefore:=snapshotInput(a);var bBefore inputSnapshot;if b!=nil{bBefore=snapshotInput(b)}
  switch op.Op {
  case "mod_switch":out=a.CopyNew();s.evaluator.DropLevel(out,a.Level()-op.Attrs.Level)
  case "rotate":
   r:=(op.Attrs.Steps%Slots+Slots)%Slots
   if r==0{out=a.CopyNew()}else{out=ckks.NewCiphertext(s.params,a.Degree(),a.Level());err=s.evaluator.Rotate(a,r,out)}
  case "mul_cp":out=ckks.NewCiphertext(s.params,a.Degree(),a.Level());predicted=a.Scale.Mul(pt.Scale);err=s.evaluator.Mul(a,pt,out)
  case "mul_cc":out=ckks.NewCiphertext(s.params,a.Degree()+b.Degree(),a.Level());predicted=a.Scale.Mul(b.Scale);err=s.evaluator.Mul(a,b,out)
  case "rescale":out=ckks.NewCiphertext(s.params,a.Degree(),op.Attrs.Level);predicted=a.Scale.Div(rlwe.NewScale(s.params.Q()[a.Level()]));err=s.evaluator.Rescale(a,out)
  case "add_cc":out=ckks.NewCiphertext(s.params,max(a.Degree(),b.Degree()),a.Level());if b.Scale.Cmp(predicted)>0{predicted=b.Scale};err=s.evaluator.Add(a,b,out)
  case "add_cp":out=ckks.NewCiphertext(s.params,a.Degree(),a.Level());if pt.Scale.Cmp(predicted)>0{predicted=pt.Scale};err=s.evaluator.Add(a,pt,out)
  case "relinearize":out=ckks.NewCiphertext(s.params,1,a.Level());err=s.evaluator.Relinearize(a,out)
  case "negate":out=ckks.NewCiphertext(s.params,a.Degree(),a.Level());err=s.evaluator.Mul(a,-1,out)
  case "boot":out,err=s.nativeBoot(a);predicted=s.params.DefaultScale()
  default:return nil,fmt.Errorf("unsupported native op")
  }
  if err!=nil{return nil,err}
  if !inputUnchanged(a,aBefore)||(b!=nil&&!inputUnchanged(b,bBefore)){return nil,fmt.Errorf("native operation mutated original input metadata")}
  if err=validateCipher(out,v,predicted);err!=nil{return nil,err};pool[op.Output]=out
  // Retain EVERY final output. Caller-owned input Handles retain their own
  // pointers even after this invocation releases a local pool map entry.
  for _,id:=range op.Inputs{if c.last[id]==n&&!final[id]{delete(pool,id)}}
  if _,used:=c.last[op.Output];!used&&!final[op.Output]{delete(pool,op.Output)}
 }
 outputs:=make([]Handle,len(c.plan.Outputs))
 for i,id:=range c.plan.Outputs {
  ct:=pool[id];if ct==nil{return nil,fmt.Errorf("missing final ciphertext")};outputs[i]=s.issue(ct,c.values[id])
 }
 cache.current=nil;return outputs,nil
}

// BootstrapBoundary implements the three real original mixed/mask connectors.
// Dropping a COPY to1 preserves the represented value; there is no divisor,
// scale relabel, plaintext simulation or decrypt/reencrypt in this path.
func(s *SharedContext)BootstrapBoundary(input Handle)(Handle,error){
 if input.owner!=s||input.serial==0||input.descriptor.Kind!="ciphertext"||input.descriptor.Context!=ContextID||input.descriptor.Scale!=43||input.descriptor.Level<1||input.descriptor.Level>5{return Handle{},fmt.Errorf("unsupported boundary input")}
 if err:=validateCipher(input.value,input.descriptor,input.value.Scale);err!=nil{return Handle{},err}
 copied:=input.value.CopyNew();s.evaluator.DropLevel(copied,copied.Level()-1)
 output,err:=s.nativeBoot(copied);if err!=nil{return Handle{},err}
 v:=boundaryValue("caller-boundary")
 if err=validateCipher(output,v,s.params.DefaultScale());err!=nil{return Handle{},err}
 return s.issue(output,v),nil
}

func(p *Program)Validate()error{
 if p==nil||p.SourceCase==""||p.PrefixLength<1||p.PrefixLength>30{return fmt.Errorf("saved right-padded source prefix1..30 required")}
 refs:=[]ComponentRef{}
 for i,ref:=range p.Layers{if ref.Role!="layer"||ref.Index!=i{return fmt.Errorf("original24 layer order differs")};refs=append(refs,ref)}
 if p.Head.Role!="head"||p.First.Role!="first"||p.Terminal.Role!="terminal"{return fmt.Errorf("complete head/first/terminal roles required")}
 refs=append(refs,p.Head,p.First,p.Terminal)
 paths:=map[string]bool{}
 for _,ref:=range refs {
  if ref.SourceCase!=p.SourceCase||ref.PlanPath==""||paths[ref.PlanPath]{return fmt.Errorf("distinct same-source original27 components required")}
  paths[ref.PlanPath]=true
 }
 return nil
}
func(s *SharedContext)forward(p *Program,inputs [34]Handle)([]Handle,error){
 current:=inputs
 for _,ref:=range p.Layers {
  c,err:=LoadComponent(ref);if err!=nil{return nil,err};out,err:=s.RunComponent(c,current[:]);if err!=nil{return nil,err}
  if len(out)!=32{return nil,fmt.Errorf("layer output arity differs")}
  var next [34]Handle;copy(next[:32],out)
  next[32],next[33]=current[32],current[33] // Exact original mask handle aliases.
  current=next
 }
 c,err:=LoadComponent(p.Head);if err!=nil{return nil,err};return s.RunComponent(c,current[:])
}

// RunTwoTokens contains52 actual component invocations:25 reused twice, first
// once, terminal once. Building this library calls none of these functions.
// C2/full27-artifact admission is the future experiment caller's obligation;
// Program.Validate proves structural identity only and never marks it linked.
func RunTwoTokens(s *SharedContext,p *Program,initial [34]Handle)(result [38]Handle,err error){
 if s==nil{return result,fmt.Errorf("borrowed shared context required")};if err=p.Validate();err!=nil{return}
 for i,h:=range initial {if err=s.require(h,boundaryValue(fmt.Sprintf("original:%d",i)));err!=nil{return}}
 original:=initial
 logits0,err:=s.forward(p,original);if err!=nil{return result,err}
 first,err:=LoadComponent(p.First);if err!=nil{return result,err}
 firstInputs:=make([]Handle,20);copy(firstInputs[:19],logits0);firstInputs[19]=original[0]
 chosen0,err:=s.RunComponent(first,firstInputs);if err!=nil{return result,err}
 if len(chosen0)!=22{return result,fmt.Errorf("first output arity differs")}
 copy(result[:19],chosen0[:19]) // Preserve first onehots through their last use.
 mixed,err:=s.BootstrapBoundary(chosen0[19]);if err!=nil{return result,err}
 attention,err:=s.BootstrapBoundary(chosen0[20]);if err!=nil{return result,err}
 selector,err:=s.BootstrapBoundary(chosen0[21]);if err!=nil{return result,err}
 second:=original // Retain original prefix and padded tail, not forward0 hidden.
 second[p.PrefixLength],second[32],second[33]=mixed,attention,selector
 logits1,err:=s.forward(p,second);if err!=nil{return result,err}
 terminal,err:=LoadComponent(p.Terminal);if err!=nil{return result,err}
 chosen1,err:=s.RunComponent(terminal,logits1);if err!=nil{return result,err}
 if len(chosen1)!=19{return result,fmt.Errorf("terminal output arity differs")};copy(result[19:],chosen1)
 return result,nil
}
