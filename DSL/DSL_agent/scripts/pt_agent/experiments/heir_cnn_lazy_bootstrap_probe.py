"""Isolated Lattigo 6.1 key/DFT lifetime adaptation for the native HEIR probe.

The original module is read-only. This entry prepares and builds by default;
--regression also checks small plaintext DFT matrices. Only --execute enables
one scheduled real native bootstrap primitive.
"""
import difflib
import json
from pathlib import Path
import shutil
import sys

from .heir_cnn_bootstrap_probe import bootstrap_source, main as probe_main

MODULE = 'github.com/tuneinsight/lattigo/v6'
PATCHES = (
    ('core/rlwe/keys.go', 'type MemEvaluationKeySet struct {\n',
     'type MemEvaluationKeySet struct {\n'
     '\t// Isolated HEIR lifetime adapter; nil preserves the native map.\n'
     '\tGetGaloisKeyFunc func(uint64) (*GaloisKey, error)\n'
     '\tGetGaloisKeysListFunc func() []uint64\n'),
    ('core/rlwe/keys.go', 'func (evk MemEvaluationKeySet) GetGaloisKey(galEl uint64) (gk *GaloisKey, err error) {\n',
     'func (evk MemEvaluationKeySet) GetGaloisKey(galEl uint64) (gk *GaloisKey, err error) {\n'
     '\tif evk.GetGaloisKeyFunc != nil { return evk.GetGaloisKeyFunc(galEl) }\n'),
    ('core/rlwe/keys.go', 'func (evk MemEvaluationKeySet) GetGaloisKeysList() (galEls []uint64) {\n',
     'func (evk MemEvaluationKeySet) GetGaloisKeysList() (galEls []uint64) {\n'
     '\tif evk.GetGaloisKeysListFunc != nil { return evk.GetGaloisKeysListFunc() }\n'),
    ('circuits/ckks/dft/dft.go', '\t"math/big"\n', '\t"math/big"\n\t"runtime"\n'),
    ('circuits/ckks/dft/dft.go', '// NewMatrixFromLiteral generates the factorized DFT/IDFT matrices for the homomorphic encoding/decoding.\n',
     '// HEIRMatrixObserver records construction boundaries, with no effect when nil.\n'
     'var HEIRMatrixObserver func(MatrixLiteral, bool)\n\n'
     '// NewMatrixFromLiteral generates the factorized DFT/IDFT matrices for the homomorphic encoding/decoding.\n'),
    ('circuits/ckks/dft/dft.go', 'func NewMatrixFromLiteral(params ckks.Parameters, d MatrixLiteral, encoder *ckks.Encoder) (Matrix, error) {\n',
     'func NewMatrixFromLiteral(params ckks.Parameters, d MatrixLiteral, encoder *ckks.Encoder) (Matrix, error) {\n'
     '\tif observer := HEIRMatrixObserver; observer != nil { observer(d, true); defer observer(d, false) }\n'),
    ('circuits/ckks/dft/dft.go', '\tpVecDFT := d.GenMatrices(params.LogN(), params.EncodingPrecision())\n',
     '\tpVecDFT := d.GenMatrices(params.LogN(), params.EncodingPrecision())\n'
     '\t// FFT scratch has no further use after the factors are returned.\n'
     '\truntime.GC()\n'),
    ('circuits/ckks/dft/dft.go', '\t\t\tmatrices = append(matrices, mat)\n\t\t\tidx++\n',
     '\t\t\tmatrices = append(matrices, mat)\n'
     '\t\t\t// Encode copied this factor into QP coefficients.\n'
     '\t\t\tpVecDFT[idx] = nil\n\t\t\truntime.GC()\n\t\t\tidx++\n'),
)

GO_HELPERS = r'''
type heirLazyGalois struct {
 mu sync.Mutex
 params ckks.Parameters
 sk *rlwe.SecretKey
 generator *rlwe.KeyGenerator
 allowed map[uint64]bool
 list []uint64
 current *rlwe.GaloisKey
 record map[string]any
 save func()
 gets,generations,hits,evictions int
 generationSeconds float64
}
func (l *heirLazyGalois) snapshot() map[string]any {
 resident:=0;if l.current!=nil{resident=1}
 return map[string]any{"capacity":1,"resident_keys":resident,"allowed_unique":len(l.list),
 "get_calls":l.gets,"generations":l.generations,"hits":l.hits,"evictions":l.evictions,
 "generation_seconds":l.generationSeconds,"serialized_keys":0}
}
func (l *heirLazyGalois) get(g uint64)(*rlwe.GaloisKey,error){
 l.mu.Lock();defer l.mu.Unlock();l.gets++
 if !l.allowed[g]{return nil,fmt.Errorf("unlisted native Galois element %d",g)}
 if l.current!=nil&&l.current.GaloisElement==g{
  l.hits++;l.record["lazy_galois"]=l.snapshot();l.save();return l.current,nil
 }
 if l.current!=nil{l.current=nil;l.evictions++;runtime.GC()}
 l.record["active_key_phase"]="galois_generation"
 l.record["galois_generation_attempts"]=l.generations+1;l.record["lazy_galois"]=l.snapshot();l.save()
 start:=time.Now();key:=l.generator.GenGaloisKeyNew(g,l.sk)
 l.generationSeconds+=time.Since(start).Seconds();l.current=key;l.generations++
 l.record["galois_generation_completed"]=l.generations
 l.record["actual_galois_key_levels"]=map[string]any{"levelQ":key.LevelQ(),"levelP":key.LevelP(),"scope":"public levels of the most recently generated native key; no coefficients exported"}
 l.record["lazy_galois"]=l.snapshot();l.save();return key,nil
}
func (l *heirLazyGalois) keys()[]uint64{
 l.mu.Lock();defer l.mu.Unlock();return append([]uint64{},l.list...)
}
func heirLifecycleKeys(b bootstrapping.Parameters,sk *rlwe.SecretKey,result map[string]any,save func())(*bootstrapping.EvaluationKeys,*heirLazyGalois,error){
 p:=b.BootstrappingParameters
 if b.ResidualParameters.N()!=p.N()||p.RingType()!=ring.Standard||b.EphemeralSecretWeight!=32{
  return nil,nil,fmt.Errorf("lifetime adapter requires the recorded same-N standard-ring and Eph32 native path")
 }
 kg:=rlwe.NewKeyGenerator(p)
 calls:=map[string]map[string]any{};result["key_api_counts"]=calls
 phase:=func(name string,fn func()){
  result["active_key_phase"]=name;calls[name]=map[string]any{"attempted":1,"completed":0};save()
  start:=time.Now();fn();calls[name]["completed"]=1;calls[name]["seconds"]=time.Since(start).Seconds();save()
 }
 extended:=rlwe.NewSecretKey(p)
 phase("extend_same_dense_secret",func(){
  q,aux:=p.RingQ(),p.RingP();buffer:=q.NewPoly()
  rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(q,q,sk.Value.Q,buffer,extended.Value.Q)
  rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(q,aux,sk.Value.Q,buffer,extended.Value.P)
 })
 sparse,e:=rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{LogN:p.LogN(),Q:p.Q()[:1],P:p.P()[:1]});if e!=nil{return nil,nil,e}
 smallKG:=rlwe.NewKeyGenerator(sparse);var sparseSK *rlwe.SecretKey
 phase("ephemeral_H32_secret",func(){sparseSK=smallKG.GenSecretKeyWithHammingWeightNew(b.EphemeralSecretWeight)})
 var denseToSparse,sparseToDense *rlwe.EvaluationKey;var rlk *rlwe.RelinearizationKey
 // Both calls deliberately use the original 6.1.0 dense key generator.
 phase("dense_to_sparse",func(){denseToSparse=kg.GenEvaluationKeyNew(extended,sparseSK)})
 phase("sparse_to_dense",func(){sparseToDense=kg.GenEvaluationKeyNew(sparseSK,extended)})
 phase("relinearization",func(){rlk=kg.GenRelinearizationKeyNew(extended)})
 result["actual_key_levels"]=map[string]any{
  "dense_to_sparse":map[string]int{"levelQ":denseToSparse.LevelQ(),"levelP":denseToSparse.LevelP()},
  "sparse_to_dense":map[string]int{"levelQ":sparseToDense.LevelQ(),"levelP":sparseToDense.LevelP()},
  "relinearization":map[string]int{"levelQ":rlk.LevelQ(),"levelP":rlk.LevelP()},
  "ephemeral_secret_storage":map[string]int{"levelQ":sparseSK.LevelQ(),"levelP":sparseSK.LevelP()}}
 result["secret_sampler_source"]=map[string]any{
  "main":"residual dense ternary P=2/3 via the original GenKeyPairNew; same secret extended to bootstrap Q/P",
  "ephemeral":"GenSecretKeyWithHammingWeightNew(32)","configured_bootstrap_Xs_used_for_main_sampling":false,
  "realized_hamming_weights_read":false,"coefficients_exported":false}
 allowed:=map[uint64]bool{}
 for _,g:=range append(b.GaloisElements(p),p.GaloisElementForComplexConjugation()){allowed[g]=true}
 list:=make([]uint64,0,len(allowed));for g:=range allowed{list=append(list,g)};sort.Slice(list,func(i,j int)bool{return list[i]<list[j]})
 lazy:=&heirLazyGalois{params:p,sk:extended,generator:kg,allowed:allowed,list:list,record:result,save:save}
 mem:=rlwe.NewMemEvaluationKeySet(rlk);mem.GetGaloisKeyFunc=lazy.get;mem.GetGaloisKeysListFunc=lazy.keys
 result["bootstrap_key_setup_kind"]="same native RLK/encapsulation arithmetic, capacity-one on-demand Galois storage"
 result["native_GenEvaluationKeys_called"]=false;result["lazy_galois"]=lazy.snapshot();save()
 return &bootstrapping.EvaluationKeys{EvkDenseToSparse:denseToSparse,EvkSparseToDense:sparseToDense,MemEvaluationKeySet:mem},lazy,nil
}
func heirDFTRegression(path string)(err error){
 result:=map[string]any{"status":"running","keygen_runs":0,"encryptions":0,"bootstraps":0,"decryptions":0,"model_forwards":0,"original_constructors":0,"lifetime_constructors":0}
 defer func(){
  if value:=recover();value!=nil{err=fmt.Errorf("panic: %v",value)}
  if err!=nil{result["status"]="error";result["error"]=err.Error()}
  var usage syscall.Rusage;syscall.Getrusage(syscall.RUSAGE_SELF,&usage);result["ru_maxrss_kib"]=usage.Maxrss
  f,e:=os.Create(path);if e!=nil{panic(e)};defer f.Close();if e=json.NewEncoder(f).Encode(result);e!=nil{panic(e)}
 }()
 p,e:=ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN:10,LogQ:[]int{55,45,45},LogP:[]int{60},LogDefaultScale:45});if e!=nil{return e}
 // Exercise optional lookup dispatch with metadata-only placeholders, no keys.
 mem:=rlwe.NewMemEvaluationKeySet(nil);placeholder:=&rlwe.GaloisKey{GaloisElement:5};mem.GaloisKeys[5]=placeholder
 got,e:=mem.GetGaloisKey(5);if e!=nil||got!=placeholder{return fmt.Errorf("native map lookup changed")}
 calls:=0;mem.GetGaloisKeyFunc=func(g uint64)(*rlwe.GaloisKey,error){calls++;if g!=7{return nil,fmt.Errorf("wrong callback element")};return placeholder,nil}
 mem.GetGaloisKeysListFunc=func()[]uint64{return []uint64{7}}
 got,e=mem.GetGaloisKey(7);list:=mem.GetGaloisKeysList();if e!=nil||got!=placeholder||calls!=1||len(list)!=1||list[0]!=7{return fmt.Errorf("optional lookup callback failed")}
 mem.GetGaloisKeyFunc=nil;mem.GetGaloisKeysListFunc=nil
 got,e=mem.GetGaloisKey(5);if e!=nil||got!=placeholder{return fmt.Errorf("nil callback did not preserve map behavior")}
 result["native_nil_callback_map_passed"]=true;result["optional_callback_dispatch_passed"]=true
 maximum,ratio:=0.0,0.0;elements:=0
 for _,kind:=range []dft.Type{dft.HomomorphicEncode,dft.HomomorphicDecode}{
  literal:=dft.MatrixLiteral{Type:kind,LogSlots:p.LogMaxSlots(),LevelQ:p.MaxLevel(),LevelP:p.PCount()-1,Levels:[]int{1,1,1},Format:dft.RepackImagAsReal,LogBSGSRatio:1}
  original,e:=dft.OriginalMatrixForLifetimeRegression(p,literal,ckks.NewEncoder(p));if e!=nil{return e};result["original_constructors"]=result["original_constructors"].(int)+1
  actual,e:=dft.NewMatrixFromLiteral(p,literal,ckks.NewEncoder(p));if e!=nil{return e};result["lifetime_constructors"]=result["lifetime_constructors"].(int)+1
  runtime.GC()
  if len(original.Matrices)!=len(actual.Matrices){return fmt.Errorf("DFT factor count differs")}
  for i,a:=range actual.Matrices{
   b:=original.Matrices[i]
   if len(a.Vec)!=len(b.Vec)||a.LevelQ!=b.LevelQ||a.LevelP!=b.LevelP||a.N1!=b.N1{return fmt.Errorf("DFT layout metadata differs")}
   for diagonal,poly:=range a.Vec{
    want,ok:=b.Vec[diagonal];if !ok{return fmt.Errorf("DFT diagonal missing")}
    decode:=func(value ring.Poly,scale rlwe.Scale)([]complex128,error){
     pt:=ckks.NewPlaintext(p,a.LevelQ);pt.Scale=scale;p.RingQ().AtLevel(a.LevelQ).IMForm(value,pt.Value)
     values:=make([]complex128,p.MaxSlots());e:=ckks.NewEncoder(p).Decode(pt,values);return values,e
    }
    x,e:=decode(poly.Q,a.Scale);if e!=nil{return e};y,e:=decode(want.Q,b.Scale);if e!=nil{return e}
    for j:=range x{for _,pair:=range [][2]float64{{real(x[j]),real(y[j])},{imag(x[j]),imag(y[j])}}{
     if math.IsNaN(pair[0])||math.IsInf(pair[0],0)||math.IsNaN(pair[1])||math.IsInf(pair[1],0){return fmt.Errorf("nonfinite DFT value")}
     delta:=math.Abs(pair[0]-pair[1]);bound:=1e-12+1e-12*math.Abs(pair[1]);elements++
     maximum=math.Max(maximum,delta);ratio=math.Max(ratio,delta/bound)
     if delta>bound{return fmt.Errorf("unchanged DFT mathematical tolerance failed")}
    }}
   }
  }
 }
 result["status"]="ok";result["elements_checked"]=elements;result["max_absolute_error"]=maximum;result["max_tolerance_ratio"]=ratio
 result["criterion"]="1e-12 + 1e-12*abs(reference)";result["scope"]="small plaintext DFT constructor regression after forced GC; no keys or encrypted execution"
 return nil
}
'''


def lifecycle_source(residual, bootstrap):
    source = bootstrap_source(residual, bootstrap)
    source = source.replace(' "encoding/json"', ' "encoding/json"\n "sort"\n "sync"\n "github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"')
    marker = 'func main(){'
    if source.count(marker) != 1:
        raise ValueError('The trusted Go main boundary changed')
    source = source.replace(marker, GO_HELPERS + '\n' + marker + r'''
 if len(os.Args)>1&&os.Args[1]=="--dft-regression"{
  runtime.GOMAXPROCS(1);debug.SetMemoryLimit(1<<30)
  if err:=heirDFTRegression(os.Args[2]);err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(1)};return
 }
''')
    marker = 'keys,_,e:=b.GenEvaluationKeys(sk)'
    if source.count(marker) != 1:
        raise ValueError('The trusted native key boundary changed')
    source = source.replace(marker, 'keys,lazy,e:=heirLifecycleKeys(b,sk,result,write)')
    marker = 'evaluator,e:=bootstrapping.NewEvaluator(b,keys)'
    if source.count(marker) != 1:
        raise ValueError('The trusted evaluator boundary changed')
    source = source.replace(marker, r'''
 dftRows:=[]map[string]any{};var dftStart time.Time
 result["DFT_constructor_boundaries"]=dftRows;result["DFT_matrices_returned"]=0
 dft.HEIRMatrixObserver=func(literal dft.MatrixLiteral,begin bool){
  if begin{dftStart=time.Now();dftRows=append(dftRows,map[string]any{"type":literal.Type,"log_slots":literal.LogSlots,"levelQ":literal.LevelQ,"levelP":literal.LevelP,"levels":literal.Levels,"completed":false})
  }else{dftRows[len(dftRows)-1]["completed"]=true;dftRows[len(dftRows)-1]["seconds"]=time.Since(dftStart).Seconds();result["DFT_matrices_returned"]=result["DFT_matrices_returned"].(int)+1}
  result["DFT_constructor_boundaries"]=dftRows;write()
 }
 defer func(){dft.HEIRMatrixObserver=nil;result["lazy_galois"]=lazy.snapshot()}()
 evaluator,e:=bootstrapping.NewEvaluator(b,keys)''')
    return source


def prepare_module(out):
    original = out / 'module-cache/github.com/tuneinsight/lattigo/v6@v6.1.0'
    isolated = out / 'lattigo-lifetime'
    if not (original / 'go.mod').is_file():
        raise ValueError('The original cached Lattigo 6.1.0 module is required')
    shutil.copytree(original, isolated)
    # Cached modules are read-only; writable permissions apply only to this copy.
    for p in isolated.rglob('*'):
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o200)
    dft_file = original / 'circuits/ckks/dft/dft.go'
    dft_text = dft_file.read_text()
    begin = dft_text.index('func NewMatrixFromLiteral(')
    end = dft_text.index('// CoeffsToSlotsNew', begin)
    reference = dft_text[begin:end].replace('func NewMatrixFromLiteral(', 'func OriginalMatrixForLifetimeRegression(', 1)
    before = {name: (isolated / name).read_text() for name in {x[0] for x in PATCHES}}
    for name, old, new in PATCHES:
        p = isolated / name
        text = p.read_text()
        if text.count(old) != 1:
            raise ValueError('Expected one source edit location in ' + name)
        p.write_text(text.replace(old, new, 1))
    p = isolated / 'circuits/ckks/dft/dft.go'
    p.write_text(p.read_text() + '\n// Unchanged native body retained only for the small plaintext regression.\n' + reference)
    diff = []
    for name, text in before.items():
        diff.extend(difflib.unified_diff(text.splitlines(True), (isolated / name).read_text().splitlines(True), fromfile='original/' + name, tofile='isolated/' + name))
    (out / 'lifetime-adaptation.patch').write_text(''.join(diff))
    mod = out / 'module/go.mod'
    mod.write_text(mod.read_text() + '\nreplace ' + MODULE + ' => ' + str(isolated) + '\n')
    result = dict(original_module=str(original), isolated_module=str(isolated), version='6.1.0',
                  original_module_modified=False, isolated_files_changed=sorted(before),
                  patch=str(out / 'lifetime-adaptation.patch'),
                  changes=['optional Galois lookup callbacks; native map preserved when callbacks are nil',
                           'capacity-one generated Galois key storage, same secret and native generator',
                           'release completed DFT input factors after native Encode and collect dead FFT scratch',
                           'retain unchanged native DFT constructor under an explicit regression-only name'],
                  key_arithmetic='HEIR cached 6.1.0 same-N dense extension, both encapsulation directions use native dense generator',
                  security_validated=False, extra_integrity_checks=0)
    (out / 'lifetime-adaptation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    execute = '--execute' in argv
    regression = '--regression' in argv
    argv = [x for x in argv if x not in ('--execute', '--regression')]
    if execute and regression:
        raise ValueError('The small regression and scheduled real primitive use separate runs')
    if not execute and '--prepare-only' not in argv:
        argv.append('--prepare-only')
    if execute and '--prepare-only' in argv:
        raise ValueError('Choose explicit execution or preparation')
    return probe_main(argv, source_builder=lifecycle_source, module_adapter=prepare_module,
                      backend_metadata=dict(round='HEIR-CNN-LIFETIME', mode='isolated-native-bootstrap-memory-lifetime',
                                            original_libraries_modified=False, isolated_lifetime_adapter=True,
                                            native_eager_result_preserved=True),
                      lifetime_regression=regression)


if __name__ == '__main__':
    sys.exit(main())
