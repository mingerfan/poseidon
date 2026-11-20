"""Independent clear arithmetic replay of saved trusted native HEIR artifacts.

This diagnostic preserves the native arithmetic graph and saved training probes.
It constructs no keys, RNS plaintexts, ciphertexts or encrypted runtime objects.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

from .. import paths
from .heir_cnn_preflight import finalize_size, FORMAL
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json


STEM_RUNTIME=r'''
func referenceConv(input,weight []float64,cin,cout,height,width int)[]float64{if len(input)!=cin*height*width||len(weight)!=cout*cin*9{panic("reference conv shapes")};out:=make([]float64,cout*height*width);for oc:=0;oc<cout;oc++{for y:=0;y<height;y++{for x:=0;x<width;x++{sum:=0.0;for ic:=0;ic<cin;ic++{for ky:=0;ky<3;ky++{for kx:=0;kx<3;kx++{iy,ix:=y+ky-1,x+kx-1;if iy>=0&&iy<height&&ix>=0&&ix<width{term:=float64(weight[((oc*cin+ic)*3+ky)*3+kx]*input[(ic*height+iy)*width+ix]);sum=sum+term}}}};out[(oc*height+y)*width+x]=sum}}};return out}
func checkPacking(got,input []float64,period int)bool{if period<len(input)||len(got)%period!=0{return false};for i,x:=range got{index:=i%period;expected:=0.0;if index<len(input){expected=input[index]};if x!=expected{return false}};return true}
func referenceConvStride(input,weight []float64,cin,cout,height,width,stride int)[]float64{if stride<=0||height%stride!=0||width%stride!=0||len(input)!=cin*height*width||len(weight)!=cout*cin*9{panic("reference stride convolution shapes")};oh,ow:=height/stride,width/stride;out:=make([]float64,cout*oh*ow);for oc:=0;oc<cout;oc++{for y:=0;y<oh;y++{for x:=0;x<ow;x++{sum:=0.0;for ic:=0;ic<cin;ic++{for ky:=0;ky<3;ky++{for kx:=0;kx<3;kx++{iy,ix:=stride*y+ky-1,stride*x+kx-1;if iy>=0&&iy<height&&ix>=0&&ix<width{term:=float64(weight[((oc*cin+ic)*3+ky)*3+kx]*input[(ic*height+iy)*width+ix]);sum=sum+term}}}};out[(oc*oh+y)*ow+x]=sum}}};return out}
func packFirstTransition(input []float64)[]float64{if len(input)!=32*16*16{panic("first transition logical shape")};out:=make([]float64,16384);seen:=make([]bool,8192);for c:=0;c<32;c++{for h:=0;h<16;h++{for w:=0;w<16;w++{slot:=(c/4)*1024+(2*h+(c%4)/2)*32+2*w+c%2;if seen[slot]{panic("first transition packing collision")};seen[slot]=true;out[slot]=input[(c*16+h)*16+w];out[slot+8192]=out[slot]}}};for _,v:=range seen{if !v{panic("first transition packing coverage")}};return out}
func padFirstTransition(input []float64)[]float64{if len(input)!=32*16*16{panic("first transition pad logical shape")};out:=make([]float64,16384);for c:=0;c<32;c++{for h:=0;h<16;h++{for w:=0;w<16;w++{out[c*324+(h+1)*18+w+1]=input[(c*16+h)*16+w]}}};return out}
func repeatLogical(input []float64)[]float64{if len(input)==0||16384%len(input)!=0{panic("repeated logical slot shape")};out:=make([]float64,16384);for i:=range out{out[i]=input[i%len(input)]};return out}
func firstOptionA(input []float64)[]float64{if len(input)!=16*32*32{panic("first OptionA source shape")};out:=make([]float64,32*16*16);for c:=0;c<16;c++{for h:=0;h<16;h++{for w:=0;w<16;w++{out[((c+8)*16+h)*16+w]=input[(c*32+2*h)*32+2*w]}}};return out}
'''


def uniform_plaintext_values(program):
    """Follow exact uniform native constants through preprocessing copy/views."""
    from .heir_native_clear_ir import refs
    tensors={};plains={};constants={};indices={}
    for node in program.functions['resnet20_silu__preprocessing'].body:
        used=refs(node.text);op=node.kind
        if op=='arith.constant':
            if node.text.endswith((' : index',' : i32',' : i64')):constants[node.defs[0]]=int(node.text.split()[1])
        elif op=='memref.get_global':tensors[node.defs[0]]={'fill':program.globals[re.search(r'@(\w+)',node.text)[1]].get('fill')}
        elif op=='memref.alloc':tensors[node.defs[0]]={'fill':None}
        elif op=='memref.copy':
            if used[1] in tensors:tensors[used[1]]['fill']=tensors.get(used[0],{}).get('fill')
        elif op in ('memref.subview','memref.cast'):tensors[node.defs[0]]=tensors.get(used[0],{'fill':None})
        elif op in ('call','func.call'):
            for value in node.defs:tensors[value]={'fill':None}
            for value in used:
                if value in tensors:tensors[value]['fill']=None
        elif op=='preprocessing.load_resource':
            if used[0] in tensors:tensors[used[0]]['fill']=None
        elif op=='lattigo.ckks.encode':plains[node.defs[0]]=tensors.get(used[1],{}).get('fill')
        elif op=='memref.store':
            if used[0] in plains:indices[constants[used[2]]]=plains[used[0]]
            elif used[1] in tensors:tensors[used[1]]['fill']=None
        elif op in ('for','if'):raise ValueError('Uniform plaintext analysis needs explicit loop-write effects')
    return indices


def verify_first_transition_packing(program):
    """Exhaust the small channel/spatial map directly from native integer SSA."""
    from .heir_native_clear_ir import constant,transfer_region
    function=program.functions['_assign_layout_17023056966022539499']
    constants={n.defs[0]:('c',int(n.text.split(' ')[1])) for n in function.body
               if n.kind=='arith.constant' and n.text.endswith((' : index',' : i32',' : i64'))}
    regions=[transfer_region(n,constants) for n in function.body if n.kind=='for']
    candidates=[r for r in regions if r['source_shape']==[32,16,16,16,1024]
                and r['destination_shape']==[8,16,32,32,16384]]
    if len(candidates)!=1:raise ValueError('First transition native channel/spatial map differs')
    region=candidates[0]
    if region['guards'] or region['destination_indices'][1]!=region['source_indices'][1] or region['destination_indices'][4]!=region['source_indices'][4]:
        raise ValueError('First transition source channel/element projection differs')
    def evaluate(expr,values):
        kind=expr[0]
        if kind=='c':return expr[1]
        if kind=='v':return values[expr[1]]
        a,b=evaluate(expr[1],values),evaluate(expr[2],values)
        if kind=='addi':return a+b
        if kind=='subi':return a-b
        if kind=='muli':return a*b
        if kind=='floordivsi' and b>0:return a//b
        raise ValueError('Unsupported first transition integer map operation '+kind)
    seen=set()
    for c in range(32):
        for h in range(16):
            for w in range(16):
                values={expr[1]:value for expr,value in zip(region['source_indices'],(c,0,h,w,0))}
                actual=[evaluate(expr,values) for expr in region['destination_indices']]
                expected=[c//4,0,2*h+(c%4)//2,2*w+c%2,0]
                if actual!=expected:raise ValueError('First transition native coordinate map differs')
                seen.add(actual[0]*1024+actual[2]*32+actual[3])
    if seen!=set(range(8192)):raise ValueError('First transition native coordinate map is not bijective')
    return dict(passed=True,native_function=function.name,native_transfer_line=region['line'],
                channel_spatial_coordinates_checked=8192,all_source_input_channels_elements_preserved=True,
                slot='floor(c/4)*1024+(2*h+floor((c%4)/2))*32+2*w+c%2',
                second_half='Original rotate(+8192) and add combines the two channel-partial halves into repeated8192 outputs')


def verify_ninth_output_packing(program):
    from .heir_native_clear_ir import transfer_region
    function=program.functions['_assign_layout_15401164577013790799']
    constants={n.defs[0]:('c',int(n.text.split(' ')[1])) for n in function.body
               if n.kind=='arith.constant' and n.text.endswith((' : index',' : i32',' : i64'))}
    regions=[transfer_region(n,constants) for n in function.body if n.kind=='for']
    candidates=[r for r in regions if r['source_shape']==[32,32,16,16,324]
                and r['destination_shape']==[8192,32,10368]]
    if len(candidates)!=1:raise ValueError('Ninth convolution output coordinate map differs')
    region=candidates[0]
    if region['guards'] or region['destination_indices'][1]!=region['source_indices'][1] or region['destination_indices'][2]!=region['source_indices'][4]:
        raise ValueError('Ninth convolution input channel/element coordinate preservation differs')
    def evaluate(expr,values):
        if expr[0]=='c':return expr[1]
        if expr[0]=='v':return values[expr[1]]
        a,b=evaluate(expr[1],values),evaluate(expr[2],values)
        if expr[0]=='addi':return a+b
        if expr[0]=='muli':return a*b
        raise ValueError('Unsupported ninth convolution output coordinate expression')
    seen=set()
    for c in range(32):
        for h in range(16):
            for w in range(16):
                values={expr[1]:value for expr,value in zip(region['source_indices'],(c,0,h,w,0))}
                actual=[evaluate(expr,values) for expr in region['destination_indices']]
                if actual!=[c*256+h*16+w,0,0]:raise ValueError('Ninth convolution output coordinate map differs')
                seen.add(actual[0])
    if seen!=set(range(8192)):raise ValueError('Ninth convolution output map is not bijective')
    return dict(passed=True,native_function=function.name,native_transfer_line=region['line'],
                channel_spatial_coordinates_checked=8192,all_source_input_channels_elements_preserved=True,
                slot='c*256+h*16+w',second_half='Original rotate(+8192) and add combines input-channel partial sums and repeats the output')


def optiona_source(program,rows):
    """Extract the exact saved affine OptionA region, including buffer aliases."""
    from .heir_native_clear_ir import Function,Node,refs
    original=program.functions['resnet20_silu__preprocessed'];definitions={v:n for n in original.body for v in n.defs}
    start,end=definitions['%pt_3717'],definitions['%ct_4882']
    if start.kind!='memref.load' or end.kind!='lattigo.ckks.add':raise ValueError('Saved native OptionA region identity differs')
    body=[n for n in original.body if start.line<=n.line<=end.line]
    inputs={'%ct_2808','%ct_3716'};defined={v for n in body for v in n.defs};used={v for n in body for v in refs(n.text)}
    external_pt=sorted(v for v in used-defined if v.startswith('%pt'))
    for value in external_pt:
        node=definitions[value]
        if node.kind!='memref.load' or refs(node.text)[0]!='%arg1':raise ValueError('External OptionA plaintext must have an original constant load')
        body.insert(0,node)
    defined={v for n in body for v in n.defs};used={v for n in body for v in refs(n.text)}
    external_ct=sorted(v for v in used-defined if v.startswith('%ct'))
    aliases={};targets={};data_external=set();ops=[];taints={}
    for node in original.body:
        operands=refs(node.text);target=None
        if node.kind in ('lattigo.ckks.add','lattigo.ckks.sub','lattigo.ckks.mul'):target=operands[3]
        elif node.kind in ('lattigo.ckks.rotate','lattigo.ckks.rescale','lattigo.ckks.relinearize','lattigo.rlwe.drop_level'):target=operands[2]
        if node.defs and node.defs[0].startswith('%ct'):
            aliases[node.defs[0]]=aliases.get(target,target) if target else node.defs[0]
        if target:targets[node.line]=target
    for value in external_ct:taints[aliases[value]]=value in inputs
    for node in body:
        operands=refs(node.text);data=[]
        if node.kind in ('lattigo.ckks.add','lattigo.ckks.sub','lattigo.ckks.mul','lattigo.ckks.mul_new'):data=operands[1:3]
        elif node.kind.startswith(('lattigo.ckks.','lattigo.rlwe.')):data=operands[1:2]
        for value in data:
            if value.startswith('%ct') and value not in defined:data_external.add(value)
        dependent=[taints.get(aliases.get(value,value),False) for value in data]
        if node.kind in ('lattigo.ckks.mul','lattigo.ckks.mul_new') and sum(dependent)>1:
            raise ValueError('OptionA source region has nonlinear data-dependent multiplication')
        if node.defs and node.defs[0].startswith('%ct'):taints[aliases[node.defs[0]]]=any(dependent)
        ops.append(node.kind)
    if data_external!=inputs or aliases['%ct_2808']==aliases['%ct_3716']:
        raise ValueError('OptionA source affine input identity differs')
    used={v for n in body for v in refs(n.text)}
    constants=[n for n in original.body if n.kind=='arith.constant' and n.defs[0] in used]
    constant_values={n.defs[0]:int(n.text.split()[1]) for n in constants}
    indices=sorted({constant_values[refs(n.text)[1]] for n in body if n.kind=='memref.load' and refs(n.text)[0]=='%arg1'})
    name='original_optiona_region'
    function=Function(name,start.line,original.args+[(v,'!ct') for v in external_ct],['!ct'],constants+body+[Node('return',end.line+1,'return %ct_4882 : !ct')])
    source,analysis=stem_source(program,rows,pt_indices=indices,lt_indices=(),evaluate=function)
    analysis.update(scope='Exact original affine native OptionA region; deterministic coordinate tags minus zero input, main convolution contribution0',
                    source_cut=dict(native_start=start.line,native_end=end.line,native_output='%ct_4882'),
                    data_inputs=sorted(inputs),external_ct_aliases={v:aliases[v] for v in external_ct},
                    nonlinear_data_products=0,all_original_operations_preserved=True,native_plaintext_indices=indices)
    names=['p'+str(i) for i in indices]
    initialization=','.join(names)+':=f_stem_prefix_preprocessing(nil,nil);pts:=objectArray(1973);'+''.join('pts.data['+str(i)+']=p'+str(i)+';' for i in indices)
    groups=sorted({aliases[v] for v in external_ct}-{aliases[v] for v in inputs});buffer_names={g:'b'+str(i) for i,g in enumerate(groups)}
    buffer_names.update({aliases['%ct_2808']:'branch',aliases['%ct_3716']:'main'})
    arguments=['nil','nil','nil','nil','nil','pts','nil']+[buffer_names[aliases[v]] for v in external_ct]
    wrapper='func callOptionA(branch,main *V,pts *A)*V{branch=clone(branch);main=clone(main);'+''.join(buffer_names[g]+':=&V{data:make([]float64,16384)};' for g in groups)+'return f_'+name+'('+','.join(arguments)+')}'
    analysis['input_ownership']='Each synthetic native call receives owned vector copies; the independent expected coordinate vector is computed before the original in-place region runs'
    main=r'''
func main(){runtime.GOMAXPROCS(1);debug.SetMemoryLimit(256<<20);_=syscall.Getrusage;_=time.Now;go physicalGuard()
 result:=map[string]any{"status":"running","scope":"synthetic exact coordinate tags through original affine OptionA region; no model sample or CKKS","actual_crypto_operations":0,"model_forwards":0,"coordinate_tag_runs":0,"zero_input_runs":0};defer func(){if x:=recover();x!=nil{result["status"]="error";result["error"]=fmt.Sprint(x)};f,e:=os.Create(os.Args[2]);if e!=nil{panic(e)};defer f.Close();json.NewEncoder(f).Encode(result)}()
 initGlobals();INITIALIZE
 labels:=make([]float64,16384);for i:=range labels{labels[i]=float64(i+1)};reference:=repeatLogical(firstOptionA(labels))
 baseline:=clone(callOptionA(&V{data:make([]float64,16384)},&V{data:make([]float64,16384)},pts));result["zero_input_runs"]=1
 observed:=clone(callOptionA(&V{data:labels},&V{data:make([]float64,16384)},pts));result["coordinate_tag_runs"]=1
 for i,v:=range labels{if v!=float64(i+1){panic("synthetic input ownership changed")}};maximum,ratio:=0.0,0.0;bad:=[]any{};badCount:=0
 for i,want:=range reference{got:=observed.data[i]-baseline.data[i];delta:=math.Abs(got-want);r:=delta/(1e-12+1e-12*math.Abs(want));maximum=math.Max(maximum,delta);ratio=math.Max(ratio,r);if r>1{badCount++;if len(bad)<32{row:=map[string]any{"output_slot":i,"expected_coordinate_label":want,"observed_coordinate_label":got,"max_tolerance_ratio":r};if got==math.Trunc(got)&&got>=1&&got<=16384{index:=int(got)-1;row["observed_source_coordinate"]=[]int{index/1024,(index%1024)/32,index%32}};if want>=1{index:=int(want)-1;row["expected_source_coordinate"]=[]int{index/1024,(index%1024)/32,index%32}};bad=append(bad,row)}}}
 result["coordinate_pairs"]=len(reference);result["bad_coordinate_pairs"]=badCount;result["first_bad_coordinates"]=bad;result["max_absolute_error"]=maximum;result["max_tolerance_ratio"]=ratio;result["counts"]=count
 if badCount==0{result["status"]="ok"}else{result["status"]="math_failed"}
}
'''.replace('INITIALIZE',initialization)
    return source+'\n'+wrapper,analysis,main


def stem_source(program,transform_rows,*,pt_indices=(0,1,2),lt_indices=(0,),evaluate=None,taps=None):
    """Slice pure native preprocessing while retaining all memref mutations."""
    from .heir_native_clear_ir import Program,Function,Node,refs,walk
    from .heir_native_clear_codegen import Emitter,RUNTIME,go_fn,vector
    original=program.functions['resnet20_silu__preprocessing']
    constants={n.defs[0]:int(n.text.split(' ')[1]) for n in original.body
               if n.kind=='arith.constant' and n.text.endswith((' : index',' : i32',' : i64'))}
    stores={}
    for node in original.body:
        if node.kind=='memref.store' and 'memref<1973x!pt>' in node.text:
            used=refs(node.text);index=constants.get(used[2])
            if index in pt_indices:
                if index in stores:raise ValueError('Native plaintext index has multiple writes')
                stores[index]=used[0]
    transforms=[n for n in original.body if n.kind=='lattigo.ckks.prepare_linear_transform']
    if len(transforms)!=20 or set(stores)!=set(pt_indices):raise ValueError('Original native stem interface differs')
    if len(set(lt_indices))!=len(lt_indices) or any(not 0<=i<len(transforms) for i in lt_indices):
        raise ValueError('Native prefix transform selection differs')
    chosen=[transforms[i] for i in lt_indices];first=chosen[0] if chosen else original.body[-1]
    outputs=[stores[i] for i in pt_indices]+[n.defs[0] for n in chosen]
    needed=set(outputs);selected=[]
    for node in reversed(original.body):
        used=refs(node.text);written=None
        if node.kind in ('memref.copy','memref.store'):written=used[1]
        elif node.kind=='preprocessing.load_resource':written=used[0]
        if needed.intersection(node.defs) or written in needed:
            if node.kind in ('for','if'):raise ValueError('Native stem slice requires an explicit effect description for regions')
            selected.append(node);needed.difference_update(node.defs);needed.update(used)
    types=['!pt']*len(pt_indices)+['!linear_transformation']*len(chosen)
    selected.reverse();selected.append(Node('return',first.line,'return '+','.join(outputs)+' : '+','.join(types)))
    prefix=Function('stem_prefix_preprocessing',original.line,original.args,types,selected)
    functions={prefix.name:prefix}
    pending=[prefix]
    while pending:
        function=pending.pop()
        for node in walk(function.body):
            if node.kind not in ('call','func.call'):continue
            name=re.search(r'@(\w+)',node.text)[1]
            if name not in functions:functions[name]=program.functions[name];pending.append(functions[name])
    pack=program.functions['resnet20_silu__encrypt__arg0'];functions[pack.name]=pack
    if evaluate is not None:functions[evaluate.name]=evaluate
    global_names={re.search(r'@(\w+)',n.text)[1] for f in functions.values() for n in walk(f.body) if n.kind=='memref.get_global'}
    subset=Program({n:program.globals[n] for n in global_names},functions,{})
    emitter=Emitter(subset,[r for r in transform_rows if r['line'] in {n.line for n in chosen}],taps=taps)
    pieces=[RUNTIME]
    for function in functions.values():pieces.append(emitter.function(function))
    pieces.append('func initGlobals(){')
    for name,g in subset.globals.items():
        value='rawTensor('+vector(g['shape'])+','+json.dumps(g['raw_hex'])+')' if 'raw_hex' in g else 'splat('+vector(g['shape'])+','+repr(g['fill'])+')'
        pieces.append('globals['+json.dumps(name)+']='+value)
    pieces.append('}')
    return '\n'.join(pieces),dict(prefix_native_lines=[n.line for n in selected[:-1]],functions=list(functions),
                                  address_checks=emitter.proofs,projection=emitter.projection_evidence,
                                  source_cut=dict(conv_native_line=15169,conv_ssa='%ct_2',input_pack_period=4096,
                                                  output_slots='oc*1024+h*32+w; derived from original first layout helper'))


def activation_source(program,rows,frontend,weights,*,convolutions=1):
    from .heir_native_clear_ir import Function,Node,refs
    original=program.functions['resnet20_silu__preprocessed'];definitions={v:n for n in original.body for v in n.defs}
    transforms=[n for n in original.body if n.kind=='lattigo.ckks.apply_linear_transform']
    if not 1<=convolutions<=9:raise ValueError('Only the original first nine convolutions are source-verified')
    value=refs(transforms[convolutions].text)[1];end=definitions[value]
    identities={'lattigo.ckks.bootstrap','lattigo.ckks.rescale','lattigo.ckks.relinearize','lattigo.rlwe.drop_level','lattigo.rlwe.drop_level_new'}
    while end.kind in identities:value=refs(end.text)[1];end=definitions[value]
    if end.kind!='lattigo.ckks.add':raise ValueError('Native first-activation boundary is unverified')
    kept=[n for n in original.body if n.line<=end.line and n.kind!='arith.constant']
    used={v for n in kept for v in refs(n.text)}
    constants=[n for n in original.body if n.kind=='arith.constant' and n.defs[0] in used]
    index_values={n.defs[0]:int(n.text.split(' ')[1]) for n in constants}
    indices=sorted({index_values[refs(n.text)[1]] for n in kept if n.kind=='memref.load' and refs(n.text)[0]=='%arg1'})
    first_value=refs(transforms[1].text)[1];first_end=definitions[first_value]
    while first_end.kind in identities:first_value=refs(first_end.text)[1];first_end=definitions[first_value]
    name='stem_prefix_evaluate';cuts={'conv':15169,'bn':15175,'normalized':15180,'first_activation':first_end.line}
    if convolutions>=2:
        middle=[n for n in kept if transforms[1].line<=n.line<transforms[2].line]
        additions=[n for n in middle if n.kind=='lattigo.ckks.add']
        if len(additions)<2:raise ValueError('Second convolution BN/normalization boundaries are absent')
        second_value=refs(transforms[2].text)[1];second_end=definitions[second_value]
        while second_end.kind in identities:second_value=refs(second_end.text)[1];second_end=definitions[second_value]
        cuts.update(second_conv=transforms[1].line,second_bn=additions[0].line,
                    second_normalized=additions[1].line,second_activation=second_end.line)
    if convolutions>=3:
        third_value=refs(transforms[3].text)[1];third_end=definitions[third_value]
        while third_end.kind in identities:third_value=refs(third_end.text)[1];third_end=definitions[third_value]
        additions=[n for n in kept if transforms[2].line<=n.line<=third_end.line and n.kind=='lattigo.ckks.add']
        if len(additions)<3:raise ValueError('First residual BN/add/normalization boundaries are absent')
        cuts.update(third_conv=transforms[2].line,third_bn=additions[0].line,
                    first_residual=additions[1].line,third_normalized=additions[2].line,third_activation=third_end.line)
    for index in range(3,min(convolutions,7)):
        stage_value=refs(transforms[index+1].text)[1];stage_end=definitions[stage_value]
        while stage_end.kind in identities:stage_value=refs(stage_end.text)[1];stage_end=definitions[stage_value]
        additions=[n for n in kept if transforms[index].line<=n.line<=stage_end.line and n.kind=='lattigo.ckks.add']
        residual=index%2==0;prefix='layer'+str(index+1)
        if len(additions)<2+int(residual):raise ValueError('Native later 16-channel prefix boundary differs')
        cuts[prefix+'_conv']=transforms[index].line;cuts[prefix+'_bn']=additions[0].line
        if residual:cuts[prefix+'_residual']=additions[1].line
        cuts[prefix+'_normalized']=additions[1+int(residual)].line;cuts[prefix+'_activation']=stage_end.line
    if convolutions>=8:
        transition_value=refs(transforms[8].text)[1];transition_end=definitions[transition_value]
        while transition_end.kind in identities:transition_value=refs(transition_end.text)[1];transition_end=definitions[transition_value]
        additions=[n for n in kept if transforms[7].line<=n.line<=transition_end.line and n.kind=='lattigo.ckks.add']
        if len(additions)<3:raise ValueError('First transition partial merge/BN/normalization boundaries differ')
        rotations=[n for n in kept if additions[2].line<n.line<=transition_end.line and n.kind=='lattigo.ckks.rotate']
        if not rotations:raise ValueError('First transition explicit padding conversion is absent')
        first_mask_value=refs(rotations[0].text)[1];first_mask=definitions[first_mask_value]
        while first_mask.kind in identities:first_mask_value=refs(first_mask.text)[1];first_mask=definitions[first_mask_value]
        if first_mask.kind!='lattigo.ckks.mul':raise ValueError('First transition conversion must start with a plaintext mask')
        polynomial_value=refs(first_mask.text)[1];polynomial=definitions[polynomial_value]
        if polynomial.kind!='lattigo.ckks.add':raise ValueError('First transition polynomial boundary differs')
        mask_writes=[n for n in kept if polynomial.line<n.line<rotations[0].line and n.kind=='lattigo.ckks.mul']
        if len(mask_writes)!=2 or any(refs(n.text)[1]!=polynomial_value for n in mask_writes):
            raise ValueError('First transition two native conversion branches must share the polynomial result')
        cuts.update(transition_conv=additions[0].line,transition_bn=additions[1].line,
                    transition_normalized=additions[2].line,transition_activation=polynomial.line,transition_padded_input=transition_end.line)
    if convolutions==9:
        scalar_blocks=list(re.finditer(r'\^bb0\(%s0: f64, %acc: f64\):\s*(.*?)linalg.yield (%s\d+) : f64',frontend,re.S))
        if len(scalar_blocks)!=19:raise ValueError('Original scalar activation bodies differ')
        coefficients=re.findall(r'arith.constant ([\deE+.-]+) : f64',scalar_blocks[8][1])[:2]
        uniform=uniform_plaintext_values(program);normal_indices=[]
        for coefficient in coefficients:
            matches=[i for i,fill in uniform.items() if fill==float(coefficient)]
            if len(matches)!=1:raise ValueError('Ninth normalization must identify one exact uniform native plaintext')
            normal_indices.append(matches[0])
        all_indices={n.defs[0]:int(n.text.split()[1]) for n in original.body if n.kind=='arith.constant'}
        loads=[n for n in kept if n.kind=='memref.load' and refs(n.text)[0]=='%arg1' and all_indices[refs(n.text)[1]] in normal_indices]
        if len(loads)!=2:raise ValueError('Ninth normalization native load identity differs')
        norm_mul=[n for n in kept if n.kind=='lattigo.ckks.mul' and refs(n.text)[2]==loads[0].defs[0]]
        norm_add=[n for n in kept if n.kind=='lattigo.ckks.add' and refs(n.text)[2]==loads[1].defs[0]]
        if len(norm_mul)!=1 or len(norm_add)!=1:raise ValueError('Ninth normalization native arithmetic identity differs')
        residual_value=refs(norm_mul[0].text)[1];residual_node=definitions[residual_value]
        while residual_node.kind in identities:residual_value=refs(residual_node.text)[1];residual_node=definitions[residual_value]
        additions=[n for n in kept if transforms[8].line<=n.line<loads[0].line and n.kind=='lattigo.ckks.add']
        scaled=[n for n in kept if additions[0].line<n.line<additions[1].line and n.kind=='lattigo.ckks.rescale']
        if not scaled:raise ValueError('Ninth native BN scaling boundary differs')
        cuts.update(layer9_conv=additions[0].line,layer9_bn_scaled=scaled[0].line,
                    layer9_residual=residual_node.line,layer9_normalized=norm_add[0].line,layer9_activation=end.line)
    for label,line in cuts.items():
        candidates=[n for n in kept if n.line==line]
        if len(candidates)!=1 or len(candidates[0].defs)!=1:raise ValueError('Native cut is not a unique scalar-vector definition')
    evaluate=Function(name,original.line,original.args,['!ct'],constants+kept+[Node('return',end.line+1,'return '+value+' : !ct')])
    source,analysis=stem_source(program,rows,pt_indices=indices,lt_indices=tuple(range(convolutions)),evaluate=evaluate,taps={(name,line):label for label,line in cuts.items()})
    blocks=list(re.finditer(r'\^bb0\(%s0: f64, %acc: f64\):\s*(.*?)linalg.yield (%s\d+) : f64',frontend,re.S))
    if len(blocks)!=19:raise ValueError('Original scalar activation bodies differ')
    def scalar_source(block,function):
        statements=[];ops=[]
        for line in block[1].strip().splitlines():
            match=re.fullmatch(r'\s*(%s\d+) = arith\.(constant|addf|subf|mulf) (.*?) : f64',line)
            if not match:raise ValueError('Unsupported original scalar activation operation: '+line)
            target=match[1][1:];kind=match[2]
            if kind=='constant':expression=repr(float(match[3]))
            else:
                operands=refs(match[3])
                if len(operands)!=2:raise ValueError('Original scalar activation arity differs')
                expression=operands[0][1:]+{'addf':'+','subf':'-','mulf':'*'}[kind]+operands[1][1:]
                if kind=='mulf':expression='float64('+expression+')'
            statements+=[''+target+':='+expression,'_='+target];ops.append(kind)
        if ops[:4]!=['constant','mulf','constant','addf']:raise ValueError('Original normalization prefix differs')
        return 'func '+function+'(s0 float64)(float64,float64){'+';'.join(statements)+';return s4,'+block[2][1:]+'}',len(ops)
    block=blocks[0];scalar,scalar_ops=scalar_source(block,'originalScalar')
    bn_literals=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16xf64>',frontend[:block.start()])
    if len(bn_literals)!=2:raise ValueError('Original first BN constants are ambiguous')
    scale,shift=(ast.literal_eval(s) for s in bn_literals)
    if len(scale)!=16 or len(shift)!=16:raise ValueError('Original BN constants shape differs')
    analysis.update(native_cuts=cuts,native_first_activation_ssa=value,native_required_plaintext_indices=indices,
                    scalar_reference='Original frontend scalar SSA body, including literal coefficients and order; no coefficient fitting',
                    basis='Chebyshev T_j with full c0 (original kernels/poly.py:cheb_eval base); no monomial reinterpretation',
                    coefficient_refit_runs=0,scalar_operations=scalar_ops,convolutions=convolutions)
    prefix_names=['p'+str(i) for i in indices]+['lt'+str(i) for i in range(convolutions)]
    initialize=','.join(prefix_names)+':=f_stem_prefix_preprocessing(nil,nil);pts:=objectArray(1973);lts:=objectArray(20);'+''.join('lts.data['+str(i)+']=lt'+str(i)+';' for i in range(convolutions))+''.join('pts.data['+str(i)+']=p'+str(i)+';' for i in indices)
    extra_reference='';extra_pairs=''
    if convolutions>=2:
        scalar2,ops2=scalar_source(blocks[1],'originalScalar2');scalar+='\n'+scalar2
        section=frontend[blocks[0].end():blocks[1].start()]
        weights2=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16x16x3x3xf64>',section)
        bn2=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16xf64>',section)
        if len(weights2)!=1 or len(bn2)!=2:raise ValueError('Original second convolution/BN constants are ambiguous')
        def flatten2(value):return sum((flatten2(x) for x in value),[]) if isinstance(value,list) else [float(value)]
        w2=flatten2(ast.literal_eval(weights2[0]));scale2,shift2=(ast.literal_eval(s) for s in bn2)
        if len(w2)!=16*16*9 or len(scale2)!=16 or len(shift2)!=16 or not all(math.isfinite(x) for x in w2+scale2+shift2):
            raise ValueError('Original second convolution/BN constant shape differs')
        extra_reference='weight2:=[]float64{'+','.join(repr(x) for x in w2)+'};scale2:=[]float64{'+','.join(repr(x) for x in scale2)+'};shift2:=[]float64{'+','.join(repr(x) for x in shift2)+'};conv2:=referenceConv(act,weight2,16,16,32,32);bn2:=make([]float64,len(conv2));norm2:=make([]float64,len(conv2));act2:=make([]float64,len(conv2));for i,v:=range conv2{oc:=i/1024;bn2[i]=float64(v*scale2[oc])+shift2[oc];norm2[i],act2[i]=originalScalar2(bn2[i])};'
        extra_pairs=',{"second_conv",conv2},{"second_bn",bn2},{"second_normalized",norm2},{"second_activation",act2}'
        analysis.update(second_scalar_operations=ops2,second_conv_reference_terms=16*9,second_conv_weights=len(w2))
    if convolutions>=3:
        scalar3,ops3=scalar_source(blocks[2],'originalScalar3');scalar+='\n'+scalar3
        section=frontend[blocks[1].end():blocks[2].start()]
        weights3=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16x16x3x3xf64>',section)
        bn3=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16xf64>',section)
        residual=re.findall(r'ins\((%\w+), (%\w+) : tensor<1x16x32x32xf64>, tensor<1x16x32x32xf64>\)',section)
        if len(weights3)!=1 or len(bn3)!=2 or residual!=[('%t42','%t14')]:
            raise ValueError('Original first residual operands or third convolution/BN constants differ')
        w3=flatten2(ast.literal_eval(weights3[0]));scale3,shift3=(ast.literal_eval(s) for s in bn3)
        if len(w3)!=16*16*9 or len(scale3)!=16 or len(shift3)!=16 or not all(math.isfinite(x) for x in w3+scale3+shift3):
            raise ValueError('Original third convolution/BN constant shape differs')
        extra_reference+='weight3:=[]float64{'+','.join(repr(x) for x in w3)+'};scale3:=[]float64{'+','.join(repr(x) for x in scale3)+'};shift3:=[]float64{'+','.join(repr(x) for x in shift3)+'};conv3:=referenceConv(act2,weight3,16,16,32,32);bn3:=make([]float64,len(conv3));residual3:=make([]float64,len(conv3));norm3:=make([]float64,len(conv3));act3:=make([]float64,len(conv3));for i,v:=range conv3{oc:=i/1024;bn3[i]=float64(v*scale3[oc])+shift3[oc];residual3[i]=bn3[i]+act[i];norm3[i],act3[i]=originalScalar3(residual3[i])};'
        extra_pairs+=',{"third_conv",conv3},{"third_bn",bn3},{"first_residual",residual3},{"third_normalized",norm3},{"third_activation",act3}'
        analysis.update(third_scalar_operations=ops3,third_conv_reference_terms=16*9,third_conv_weights=len(w3),
                        residual_reference='Original frontend ins(%t42, %t14): third BN + first activation')
    later_rows=[]
    for index in range(3,min(convolutions,7)):
        ordinal=index+1;prefix='layer'+str(ordinal);suffix=str(ordinal)
        function='originalScalar'+suffix;scalar_n,ops_n=scalar_source(blocks[index],function);scalar+='\n'+scalar_n
        section=frontend[blocks[index-1].end():blocks[index].start()]
        weight_literals=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16x16x3x3xf64>',section)
        bn_literals=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16xf64>',section)
        residuals=re.findall(r'ins\((%\w+), (%\w+) : tensor<1x16x32x32xf64>, tensor<1x16x32x32xf64>\)',section)
        has_residual=index%2==0
        if len(weight_literals)!=1 or len(bn_literals)!=2 or len(residuals)!=int(has_residual):
            raise ValueError('Original later 16-channel convolution/BN/add structure differs')
        if has_residual:
            skip_result=re.findall(r'(%\w+) = linalg.generic',frontend[:blocks[index-2].start()])[-1]
            current_results=re.findall(r'(%\w+) = linalg.generic',section)
            if len(current_results)!=4 or residuals!=[(current_results[1],skip_result)]:
                raise ValueError('Original later residual operands differ')
        weight_n=flatten2(ast.literal_eval(weight_literals[0]));scale_n,shift_n=(ast.literal_eval(s) for s in bn_literals)
        if len(weight_n)!=16*16*9 or len(scale_n)!=16 or len(shift_n)!=16 or not all(math.isfinite(x) for x in weight_n+scale_n+shift_n):
            raise ValueError('Original later 16-channel constant shape differs')
        literal=lambda values:'[]float64{'+','.join(repr(x) for x in values)+'}'
        previous='act'+str(index);conv='conv'+suffix;bn='bn'+suffix;norm='norm'+suffix;act='act'+suffix
        extra_reference+='weight'+suffix+':='+literal(weight_n)+';scale'+suffix+':='+literal(scale_n)+';shift'+suffix+':='+literal(shift_n)+';'+conv+':=referenceConv('+previous+',weight'+suffix+',16,16,32,32);'+bn+':=make([]float64,len('+conv+'));'+norm+':=make([]float64,len('+conv+'));'+act+':=make([]float64,len('+conv+'));'
        residual='residual'+suffix
        if has_residual:extra_reference+=residual+':=make([]float64,len('+conv+'));'
        extra_reference+='for i,v:=range '+conv+'{oc:=i/1024;'+bn+'[i]=float64(v*scale'+suffix+'[oc])+shift'+suffix+'[oc];'
        argument=bn+'[i]'
        if has_residual:
            skip='act'+str(index-1);extra_reference+=residual+'[i]='+bn+'[i]+'+skip+'[i];';argument=residual+'[i]'
        extra_reference+=norm+'[i],'+act+'[i]='+function+'('+argument+')};'
        extra_pairs+=',{"'+prefix+'_conv",'+conv+'},{"'+prefix+'_bn",'+bn+'}'
        if has_residual:extra_pairs+=',{"'+prefix+'_residual",'+residual+'}'
        extra_pairs+=',{"'+prefix+'_normalized",'+norm+'},{"'+prefix+'_activation",'+act+'}'
        later_rows.append(dict(layer=ordinal,scalar_operations=ops_n,weights=len(weight_n),residual=has_residual))
    analysis['later_16_channel_references']=later_rows
    if convolutions>=8:
        analysis['first_transition_packing']=verify_first_transition_packing(program)
        scalar8,ops8=scalar_source(blocks[7],'originalScalar8');scalar+='\n'+scalar8
        section=frontend[blocks[6].end():blocks[7].start()]
        weight_literals=re.findall(r'arith.constant dense<(\[.*\])> : tensor<32x16x3x3xf64>',section)
        bn_literals=re.findall(r'arith.constant dense<(\[.*\])> : tensor<32xf64>',section)
        if len(weight_literals)!=1 or len(bn_literals)!=2 or 'strides = dense<[2, 2]>' not in section:
            raise ValueError('Original first32-channel transition convolution/BN/stride differs')
        weight8=flatten2(ast.literal_eval(weight_literals[0]));scale8,shift8=(ast.literal_eval(s) for s in bn_literals)
        if len(weight8)!=32*16*9 or len(scale8)!=32 or len(shift8)!=32 or not all(math.isfinite(x) for x in weight8+scale8+shift8):
            raise ValueError('Original first32-channel constant shape differs')
        padding_section=frontend[blocks[7].end():blocks[8].start()]
        padding=re.search(r'(%\w+) = tensor.pad (%\w+) low\[0, 0, 1, 1\] high\[0, 0, 1, 1\] \{\s*\^bb0\([^\n]+\):\s*tensor.yield (%\w+) : f64\s*\} : tensor<1x32x16x16xf64> to tensor<1x32x18x18xf64>',padding_section)
        zero=re.search(re.escape(padding[3])+r' = arith.constant ([\deE+.-]+) : f64',padding_section) if padding else None
        if padding is None or zero is None or float(zero[1])!=0:raise ValueError('Original next-convolution zero-padding operation differs')
        extra_reference+='weight8:='+literal(weight8)+';scale8:='+literal(scale8)+';shift8:='+literal(shift8)+';conv8:=referenceConvStride(act7,weight8,16,32,32,32,2);bn8:=make([]float64,len(conv8));norm8:=make([]float64,len(conv8));act8:=make([]float64,len(conv8));for i,v:=range conv8{oc:=i/256;bn8[i]=float64(v*scale8[oc])+shift8[oc];norm8[i],act8[i]=originalScalar8(bn8[i])};packedConv8,packedBn8,packedNorm8,packedAct8:=packFirstTransition(conv8),packFirstTransition(bn8),packFirstTransition(norm8),packFirstTransition(act8);paddedAct8:=padFirstTransition(act8);'
        extra_pairs+=',{"transition_conv",packedConv8},{"transition_bn",packedBn8},{"transition_normalized",packedNorm8},{"transition_activation",packedAct8},{"transition_padded_input",paddedAct8}'
        analysis['first_transition_reference']=dict(scalar_operations=ops8,weights=len(weight8),stride=2,input_shape=[1,16,32,32],output_shape=[1,32,16,16],kernel_terms=16*9)
        analysis['next_convolution_padding']=dict(source=padding[2],result=padding[1],zero=float(zero[1]),shape=[1,32,18,18],
                                                 slot='c*324+(h+1)*18+w+1; original NCHW tensor.pad, zero tail to16384',
                                                 earlier_boundary_error='A previous new cut compared this post-conversion vector against the pre-conversion quad-spatial layout; original result is retained')
    if convolutions==9:
        scalar9,ops9=scalar_source(blocks[8],'originalScalar9');scalar+='\n'+scalar9
        section=frontend[blocks[7].end():blocks[8].start()]
        weight_literals=re.findall(r'arith.constant dense<(\[.*\])> : tensor<32x32x3x3xf64>',section)
        bn_literals=re.findall(r'arith.constant dense<(\[.*\])> : tensor<32xf64>',section)
        expected_slice=['[0, 0, 0, 0] [1, 16, 16, 32] [1, 1, 2, 1]', '[0, 0, 0, 0] [1, 16, 16, 16] [1, 1, 1, 2]']
        if len(weight_literals)!=1 or len(bn_literals)!=2 or any(x not in section for x in expected_slice) or 'low[0, 8, 0, 0] high[0, 8, 0, 0]' not in section:
            raise ValueError('Original first OptionA convolution/BN/stride/channel-pad differs')
        skip_source=re.findall(r'(%\w+) = linalg.generic',frontend[:blocks[6].start()])[-1]
        first_slice=re.search(r'(%\w+) = tensor.extract_slice '+re.escape(skip_source)+re.escape(expected_slice[0])+r' :',section)
        if first_slice is None:
            first_slice=re.search(r'(%\w+) = tensor.extract_slice '+re.escape(skip_source)+r'\s*'+re.escape(expected_slice[0])+r' :',section)
        second_slice=re.search(r'(%\w+) = tensor.extract_slice '+re.escape(first_slice[1])+r'\s*'+re.escape(expected_slice[1])+r' :',section) if first_slice else None
        skip_pad=re.search(r'(%\w+) = tensor.pad '+re.escape(second_slice[1])+r' low\[0, 8, 0, 0\] high\[0, 8, 0, 0\] \{\s*\^bb0\([^\n]+\):\s*tensor.yield (%\w+) : f64',section) if second_slice else None
        skip_zero=re.search(re.escape(skip_pad[2])+r' = arith.constant ([\deE+.-]+) : f64',section) if skip_pad else None
        if skip_pad is None or skip_zero is None or float(skip_zero[1])!=0:raise ValueError('Original first OptionA exact slice-source/pad-zero chain differs')
        weight9=flatten2(ast.literal_eval(weight_literals[0]));scale9,shift9=(ast.literal_eval(s) for s in bn_literals)
        if len(weight9)!=32*32*9 or len(scale9)!=32 or len(shift9)!=32 or not all(math.isfinite(x) for x in weight9+scale9+shift9):
            raise ValueError('Original first32-channel residual constant shape differs')
        extra_reference+='weight9:='+literal(weight9)+';scale9:='+literal(scale9)+';shift9:='+literal(shift9)+';conv9:=referenceConv(act8,weight9,32,32,16,16);scaled9:=make([]float64,len(conv9));residual9:=make([]float64,len(conv9));norm9:=make([]float64,len(conv9));act9:=make([]float64,len(conv9));skip9:=firstOptionA(act7);for i,v:=range conv9{oc:=i/256;scaled9[i]=float64(v*scale9[oc]);residual9[i]=(scaled9[i]+shift9[oc])+skip9[i];norm9[i],act9[i]=originalScalar9(residual9[i])};packedConv9,packedScaled9,packedResidual9,packedNorm9,packedAct9:=repeatLogical(conv9),repeatLogical(scaled9),repeatLogical(residual9),repeatLogical(norm9),repeatLogical(act9);'
        extra_pairs+=',{"layer9_conv",packedConv9},{"layer9_bn_scaled",packedScaled9},{"layer9_residual",packedResidual9},{"layer9_normalized",packedNorm9},{"layer9_activation",packedAct9}'
        analysis['first32_residual_reference']=dict(scalar_operations=ops9,weights=len(weight9),normalization_uniform_plaintext_indices=normal_indices,
                                                   skip='Original tensor.extract_slice stride2 in H/W followed by channel zero-pad8 on both sides',
                                                   logical_slot='c*256+h*16+w, repeated8192')
        analysis['ninth_output_packing']=verify_ninth_output_packing(program)
    main=r'''
func mathRow(name string,observed,reference []float64)map[string]any{if len(observed)!=len(reference){panic("cut shape")};maximum,ratio:=0.0,0.0;bad:=-1;for i,want:=range reference{got:=observed[i];if math.IsNaN(got)||math.IsInf(got,0){panic("nonfinite cut")};delta:=math.Abs(got-want);r:=delta/(1e-12+1e-12*math.Abs(want));maximum=math.Max(maximum,delta);ratio=math.Max(ratio,r);if r>1&&bad<0{bad=i}};row:=map[string]any{"name":name,"pairs":len(reference),"passed":bad<0,"max_absolute_error":maximum,"max_tolerance_ratio":ratio,"first_bad_slot":bad};if bad>=0{row["first_bad_reference"]=reference[bad];row["first_bad_observed"]=observed[bad]};return row}
func main(){runtime.GOMAXPROCS(1);debug.SetMemoryLimit(256<<20);_=syscall.Getrusage;_=time.Now;go physicalGuard()
 result:=map[string]any{"status":"running","scope":"train:0 native prefix through first SiLU; original frontend literal scalar reference; crypto0","actual_crypto_operations":0,"coefficient_refit_runs":0};defer func(){if x:=recover();x!=nil{result["status"]="error";result["error"]=fmt.Sprint(x)};f,e:=os.Create(os.Args[2]);if e!=nil{panic(e)};defer f.Close();json.NewEncoder(f).Encode(result)}()
 raw,e:=os.ReadFile(os.Args[1]);if e!=nil{panic(e)};var sample struct{Input []float64 `json:"input"`};if e=json.Unmarshal(raw,&sample);e!=nil{panic(e)}
 initGlobals();INITIALIZE
 input:=tensor([]int64{1,3,32,32});input.s.dense=sample.Input;input.initialized=true;packed:=f_resnet20_silu__encrypt__arg0(nil,nil,nil,nil,input)
 if !checkPacking(packed.data[0].(*V).data,sample.Input,4096){panic("original packing differs")}
 _=f_stem_prefix_evaluate(nil,nil,nil,nil,packed,pts,lts)
 weight:=[]float64{WEIGHTS};scale:=[]float64{SCALES};shift:=[]float64{SHIFTS}
 conv:=referenceConv(sample.Input,weight,3,16,32,32);bn:=make([]float64,len(conv));norm:=make([]float64,len(conv));act:=make([]float64,len(conv))
 for i,v:=range conv{oc:=i/1024;bn[i]=float64(v*scale[oc])+shift[oc];norm[i],act[i]=originalScalar(bn[i])}
 EXTRA_REFERENCE
 checks:=[]any{};for _,pair:=range []struct{name string;reference []float64}{{"conv",conv},{"bn",bn},{"normalized",norm},{"first_activation",act}EXTRA_PAIRS}{observed:=captured[pair.name];if observed==nil{panic("missing native cut capture")};row:=mathRow(pair.name,observed.data,pair.reference);checks=append(checks,row);result["checks"]=checks;if row["passed"]!=true{result["status"]="math_failed";result["first_difference"]=pair.name;return}}
 result["status"]="ok";result["counts"]=count;result["scalar_literal_ref_only"]=true
}
'''.replace('INITIALIZE',initialize).replace('WEIGHTS',','.join(repr(float(x)) for x in weights)).replace('SCALES',','.join(repr(float(x)) for x in scale)).replace('SHIFTS',','.join(repr(float(x)) for x in shift)).replace('EXTRA_REFERENCE',extra_reference).replace('EXTRA_PAIRS',extra_pairs).replace('native prefix through first SiLU','native prefix through '+str(convolutions)+' original convolutions/activations')
    return source+'\n'+scalar,analysis,main


def optiona_diagnostic(args):
    from .heir_native_clear_ir import parse
    from .heir_native_clear_codegen import native_transform_rows
    out=args.out.absolute();out.mkdir(parents=True,exist_ok=False);recorder=StageRecorder(out/'events.jsonl')
    result=dict(round='HEIR-NATIVE-OPTIONA-COORDINATE-CUT',mode='native-affine-coordinate-regression',model='resnet20_silu',
                status='running',error=None,processes=[],worker=None,artifact_bytes=None,**{n:None for n in FORMAL},
                llm_calls=0,keygen_runs=0,real_ckks_attempts=0,real_ckks_completed=0,model_forwards=0,
                heldout_samples=0,confirm_samples=0,report_samples=0,training_samples=0,quality_selection=False,candidate_frozen=False,
                downloads=0,installations=0,scope='Deterministic exact coordinate tags and zero input through the saved affine native OptionA region; no model samples/quality/CKKS')
    try:
        with recorder.span('native-optiona-coordinate-cut'):
            ir,utils,_,_=checked_sources(args);program=parse(ir.read_text());rows=native_transform_rows(program,utils.read_text())
            source,analysis,main=optiona_source(program,rows);write_json(out/'analysis.json',analysis)
            (out/'main.go').write_text(source+'\n'+STEM_RUNTIME+'\n'+main);(out/'go.mod').write_text('module heir-optiona-coordinate-cut\n\ngo 1.26\n')
            env=environment(out,256);env['HEIR_NATIVE_CLEAR_RSS_MIB']='512';binary=out/'optiona-cut'
            for stage,command,seconds in [('build',[str(args.go),'build','-p=1','-buildvcs=false','-gcflags=-l','-o',str(binary),'.'],180),('cut',[str(binary),'synthetic-coordinate-tags',str(out/'worker-result.json')],60)]:
                with recorder.span(stage,argv=command):
                    obs=run_process(command,cwd=out,log=out/(stage+'.log'),resources=out/(stage+'-resources.jsonl'),timeout=seconds,mem_gib=2,env=env)
                result['processes'].append(dict(stage=stage,argv=command,observation=obs))
                if obs['exit_code']!=0 or obs['timed_out'] or obs['remaining_pids']:raise RuntimeError(stage+' failed')
            result['worker']=json.loads((out/'worker-result.json').read_text());result['status']=result['worker']['status']
    except BaseException as exc:result['status']='error';result['error']=type(exc).__name__+': '+str(exc)
    finally:
        if (out/'worker-result.json').is_file():result['worker']=json.loads((out/'worker-result.json').read_text())
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')));result['ts']=datetime.now(timezone.utc).isoformat();finalize_size(out,result);register(out,result)
    print(json.dumps({k:result[k] for k in ('status','error','worker')}));return 0 if result['status']=='ok' else 1


def _numpy_affine_step(np,operation,operands,*,rotation=0):
    if operation in ('mul','mul_new'):return np.multiply(operands[0],operands[1])
    if operation=='add':return np.add(operands[0],operands[1])
    if operation=='sub':return np.subtract(operands[0],operands[1])
    if operation=='rotate':return np.roll(operands[0],-rotation)
    if operation in ('bootstrap','rescale','relinearize','drop_level','drop_level_new'):return operands[0].copy()
    raise ValueError('Independent native OptionA operation unsupported: '+operation)


def independent_optiona_numpy(args):
    """Independent fresh-array SSA evaluation; no clear Go runtime is imported."""
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    import collections
    import resource
    import numpy as np
    from .heir_native_clear_ir import parse,refs,shape_type
    ir=args.native_result.parent/'model-lattigo.mlir';program=parse(ir.read_text());function=program.functions['resnet20_silu__preprocessed']
    definitions={v:n for n in function.body for v in n.defs};start,end=definitions['%pt_3717'].line,definitions['%ct_4882'].line
    region=[n for n in function.body if start<=n.line<=end]
    constants={n.defs[0]:int(n.text.split()[1]) for n in function.body if n.kind=='arith.constant'}
    plain_loads={n.defs[0]:constants[refs(n.text)[1]] for n in function.body if n.kind=='memref.load' and refs(n.text)[0]=='%arg1'}
    methods={'mul':'Mul','mul_new':'MulNew','add':'Add','sub':'Sub','rotate':'Rotate','rescale':'Rescale','relinearize':'Relinearize','bootstrap':'Bootstrap','drop_level':'DropLevel','drop_level_new':'DropLevel','apply_linear_transform':'EvaluateNew'}
    go=args.go_result.parent/'module/heircnn.go';go_lines=go.read_text().splitlines();begin=next(i for i,s in enumerate(go_lines) if s.startswith('func Resnet20_silu__preprocessed('));finish=next(i for i in range(begin+1,len(go_lines)) if go_lines[i].startswith('func '))
    go_constants={};go_ops=[]
    for number in range(begin,finish):
        line=go_lines[number];constant=re.search(r'(\w+) := int(?:32|64)\((-?\d+)\)',line)
        if constant:go_constants[constant[1]]=int(constant[2])
        call=re.search(r'\b(evaluator|bootstrapping_evaluator|\w+_lteval)\.([A-Z]\w*)\((.*)\)',line)
        if call and call[2] in set(methods.values()):go_ops.append((number,call[2],[s.strip() for s in call[3].split(',')]))
    native_ops=[n for n in function.body if n.kind.startswith('lattigo.')]
    if len(native_ops)!=len(go_ops):raise ValueError('Original native IR / official Go operator sequence length differs')
    selected=[];rotations=0
    for node,(number,method,operands) in zip(native_ops,go_ops):
        if methods.get(node.kind.rsplit('.',1)[1])!=method:raise ValueError('Original native IR / official Go operator kind differs')
        if method=='Rotate':
            value=re.fullmatch(r'int\((\w+)\)',operands[1])
            if value is None or go_constants[value[1]]!=constants[refs(node.text)[3]]:raise ValueError('Original native IR / official Go rotation differs')
            rotations+=1
        if start<=node.line<=end:selected.append(number)
    (args.worker_out.parent/'official-go-optiona-excerpt.txt').write_text('\n'.join(go_lines[min(selected):max(selected)+5])+'\n')
    tensors={};encoded={};descriptors={};pre_constants={}
    for node in program.functions['resnet20_silu__preprocessing'].body:
        used=refs(node.text);op=node.kind
        if op=='arith.constant':pre_constants[node.defs[0]]=int(node.text.split()[1])
        elif op=='memref.get_global':tensors[node.defs[0]]=dict(program.globals[re.search(r'@(\w+)',node.text)[1]])
        elif op=='memref.alloc':tensors[node.defs[0]]={'unsupported':'uninitialized tensor','shape':shape_type(node.text)[0]}
        elif op=='memref.copy':
            if used[1] in tensors:
                snapshot=dict(tensors.get(used[0],{'unsupported':'unknown copy source'}));tensors[used[1]].clear();tensors[used[1]].update(snapshot)
        elif op=='preprocessing.load_resource':
            destination=tensors[used[0]];shape=destination['shape'];destination.clear();destination.update(shape=shape,resource=re.search(r'"([^"]+)"',node.text)[1])
        elif op=='memref.cast':tensors[node.defs[0]]=tensors.get(used[0],{'unsupported':'unknown cast'})
        elif op=='memref.subview':
            view=re.search(r'\[([^]]*)\] \[([^]]*)\] \[([^]]*)\]',node.text)
            if view is None:raise ValueError('Independent native plaintext view syntax differs')
            offsets,sizes,strides=([int(s.strip()) for s in view[i].split(',')] for i in (1,2,3))
            parent=tensors.get(used[0],{'unsupported':'unknown view'});view_ok=not any(offsets) and all(s==1 for s in strides) and math.prod(sizes)==16384 and math.prod(parent.get('shape',[0]))==16384
            tensors[node.defs[0]]=parent if view_ok else {'unsupported':'nontrivial view not independently implemented'}
        elif op in ('call','func.call'):
            for value in node.defs:tensors[value]={'unsupported':'computed tensor not independently implemented'}
        elif op=='lattigo.ckks.encode':encoded[node.defs[0]]=dict(tensors.get(used[1],{'unsupported':'unknown encoded tensor'}))
        elif op=='memref.store' and used[0] in encoded:descriptors[pre_constants[used[2]]]=encoded[used[0]]
        elif op in ('for','if'):raise ValueError('Independent plaintext descriptor analysis requires explicit loop handling')
    required=set()
    for node in region:
        for value in refs(node.text):
            if value.startswith('%pt'):required.add(plain_loads[value])
    arrays={};resource_cache={}
    for index in sorted(required):
        descriptor=descriptors[index]
        if 'unsupported' in descriptor or math.prod(descriptor.get('shape',[0]))!=16384:raise ValueError('Independent native plaintext source unsupported: '+str(index))
        if 'resource' in descriptor:
            path=Path(descriptor['resource'])
            if ir.parent.resolve() not in path.resolve().parents or path.stat().st_size!=16384*8:raise ValueError('Independent native resource path/size differs')
            if str(path) not in resource_cache:resource_cache[str(path)]=np.fromfile(path,dtype='<f8')
            array=resource_cache[str(path)]
        elif 'fill' in descriptor:array=np.full(16384,descriptor['fill'],dtype=np.float64)
        elif 'raw_hex' in descriptor:array=np.frombuffer(bytes.fromhex(descriptor['raw_hex']),dtype='<f8')
        else:raise ValueError('Independent native plaintext representation unsupported')
        if array.dtype!=np.float64 or array.shape!=(16384,) or not np.isfinite(array).all():raise ValueError('Independent native plaintext dtype/shape/finite differs')
        array.setflags(write=False);arrays[index]=array
    def data_refs(node):
        operands=refs(node.text)
        if node.kind in ('lattigo.ckks.mul','lattigo.ckks.mul_new','lattigo.ckks.add','lattigo.ckks.sub'):return operands[1:3]
        if node.kind.startswith('lattigo.'):return operands[1:2]
        return []
    def run(branch):
        values={'%ct_2808':branch.copy(),'%ct_3716':np.zeros(16384,dtype=np.float64)};remaining=collections.Counter(v for n in region for v in data_refs(n) if v.startswith('%ct'));counts=collections.Counter()
        for node in region:
            if node.kind=='memref.load':continue
            operands=refs(node.text);data=data_refs(node)
            def value(name):return arrays[plain_loads[name]] if name.startswith('%pt') else values[name]
            op=node.kind.rsplit('.',1)[1]
            output=_numpy_affine_step(np,op,[value(name) for name in data],rotation=constants[operands[3]] if op=='rotate' else 0)
            if output.dtype!=np.float64 or output.shape!=(16384,) or not np.isfinite(output).all():raise ValueError('Independent native OptionA output dtype/shape/finite differs')
            values[node.defs[0]]=output;counts[op]+=1
            for name in data:
                if name.startswith('%ct'):
                    remaining[name]-=1
                    if remaining[name]==0 and name!='%ct_4882':values.pop(name,None)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>512*1024:raise RuntimeError('Independent NumPy RSS exceeds512MiB')
        return values['%ct_4882'].copy(),dict(counts)
    labels=np.arange(1,16385,dtype=np.float64);reference=np.zeros((32,16,16),dtype=np.float64);reference[8:24,:,:]=labels.reshape(16,32,32)[:,::2,::2];reference=np.tile(reference.reshape(-1),2)
    zero,counts_zero=run(np.zeros(16384,dtype=np.float64));tag,counts_tag=run(labels);observed=tag-zero
    difference=np.abs(observed-reference);ratio=difference/(1e-12+1e-12*np.abs(reference));bad=np.flatnonzero(ratio>1);first=[]
    for slot in bad[:32]:
        got,want=float(observed[slot]),float(reference[slot]);row=dict(output_slot=int(slot),expected_coordinate_label=want,observed_coordinate_label=got)
        for name,label in [('expected_source_coordinate',want),('observed_source_coordinate',got)]:
            if label>=1 and label<=16384 and label==int(label):i=int(label)-1;row[name]=[i//1024,(i%1024)//32,i%32]
        first.append(row)
    result=dict(status='ok' if not len(bad) else 'math_failed',scope='Independent NumPy fresh-array SSA OptionA replay, original binary plaintext masks and official Go operator/rotation source alignment; all level/bootstrap math identities',
                actual_crypto_operations=0,model_forwards=0,zero_input_runs=1,coordinate_tag_runs=1,coordinate_pairs=16384,bad_coordinate_pairs=int(len(bad)),
                first_bad_coordinates=first,max_absolute_error=float(difference.max()),max_tolerance_ratio=float(ratio.max()),
                exact_original_plaintext_indices=sorted(required),binary_mask_resources_read=len(resource_cache),native_region_operations=len(region),
                official_go_operator_sequence_verified=len(native_ops),official_go_rotation_values_verified=rotations,official_go_line_range=[min(selected)+1,max(selected)+1],
                counts_zero=counts_zero,counts_tag=counts_tag,array_ownership='Every arithmetic result owns a new NumPy array; prior SSA values retained until their last data read; target buffers are ignored in this independent mathematical interpretation',
                original_library_changes=0,reference_changes=0,tolerance_changes=0,ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(args.worker_out,result);return 0


def numpy_optiona_diagnostic(args):
    out=args.out.absolute();out.mkdir(parents=True,exist_ok=False);recorder=StageRecorder(out/'events.jsonl')
    result=dict(round='HEIR-NATIVE-OPTIONA-INDEPENDENT-NUMPY',mode='independent-fresh-SSA-coordinate-regression',model='resnet20_silu',
                status='running',error=None,processes=[],worker=None,artifact_bytes=None,**{n:None for n in FORMAL},
                llm_calls=0,keygen_runs=0,real_ckks_attempts=0,real_ckks_completed=0,model_forwards=0,
                heldout_samples=0,confirm_samples=0,report_samples=0,training_samples=0,quality_selection=False,candidate_frozen=False,
                downloads=0,installations=0,scope='Independent NumPy coordinate-map replay of original affine native OptionA region; no model inputs/weights/CKKS')
    try:
        with recorder.span('independent-native-optiona-numpy'):
            checked_sources(args)
            command=[sys.executable,'-m','pt_agent.experiments.heir_native_clear_math']
            for name in ('native_result','go_result','adapter_result','probes','probe_source','go','out'):command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            command+=['--numpy-optiona-worker','--worker-out',str(out/'worker-result.json')]
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            with recorder.span('numpy-cut',argv=command):
                obs=run_process(command,cwd=Path(__file__).resolve().parents[2],log=out/'cut.log',resources=out/'cut-resources.jsonl',timeout=180,mem_gib=2,env=env)
            result['processes'].append(dict(stage='numpy-cut',argv=command,observation=obs))
            if obs['exit_code']!=0 or obs['timed_out'] or obs['remaining_pids']:raise RuntimeError('Independent NumPy cut failed')
            result['worker']=json.loads((out/'worker-result.json').read_text());result['status']=result['worker']['status']
    except BaseException as exc:result['status']='error';result['error']=type(exc).__name__+': '+str(exc)
    finally:
        if (out/'worker-result.json').is_file():result['worker']=json.loads((out/'worker-result.json').read_text())
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')));result['ts']=datetime.now(timezone.utc).isoformat();finalize_size(out,result);register(out,result)
    print(json.dumps({k:result[k] for k in ('status','error','worker')}));return 0 if result['status']=='ok' else 1


def stem_diagnostic(args):
    from .heir_native_clear_ir import parse
    from .heir_native_clear_codegen import native_transform_rows
    out=args.out.absolute();out.mkdir(parents=True,exist_ok=False);recorder=StageRecorder(out/'events.jsonl')
    result=dict(round='HEIR-NATIVE-STEM-CUT',mode='independent-native-first-convolution',model='resnet20_silu',
                status='running',error=None,processes=[],worker=None,artifact_bytes=None,**{n:None for n in FORMAL},
                llm_calls=0,keygen_runs=0,real_ckks_attempts=0,real_ckks_completed=0,
                heldout_samples=0,confirm_samples=0,report_samples=0,quality_selection=False,candidate_frozen=False,
                downloads=0,installations=0,scope='One saved train:0 first-convolution cut; no full model/quality/CKKS result')
    try:
        with recorder.span('first-native-convolution-cut'):
            ir,utils,_,_=checked_sources(args)
            cached=json.loads((args.cached_probes.parent/'result.json').read_text())
            if Path(cached['source']['probes']).resolve()!=args.probes.resolve() or cached.get('completed',{}).get('probe_read')!=1:
                raise ValueError('Cached training input must identify the original saved probes')
            samples=json.loads(args.cached_probes.read_text())
            if len(samples)!=3 or samples[0]['sample_id']!='train:0' or len(samples[0]['input'])!=3072:
                raise ValueError('Only original cached train:0 is used for this cut')
            sample=samples[0]
            if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in sample['input']):raise ValueError('Cached input must remain finite float64')
            program=parse(ir.read_text());rows=native_transform_rows(program,utils.read_text())
            source,analysis=stem_source(program,rows) if not args.activation_cut else ('',{})
            native=json.loads(args.native_result.read_text());frontend=Path(native['source']).read_text()
            weights=re.findall(r'arith.constant dense<(\[.*\])> : tensor<16x3x3x3xf64>',frontend)
            if len(weights)!=1:raise ValueError('Original first-convolution weight literal is ambiguous')
            array=ast.literal_eval(weights[0]);flat=[]
            def flatten(value):
                if isinstance(value,list):
                    for child in value:flatten(child)
                elif isinstance(value,(float,int)) and math.isfinite(value):flat.append(float(value))
                else:raise ValueError('Original float64 source constant differs')
            flatten(array)
            if len(flat)!=432:raise ValueError('Original first-convolution weight shape differs')
            (out/'sample.json').write_text(json.dumps(sample)+'\n');write_json(out/'analysis.json',analysis)
            main=r'''
func main(){runtime.GOMAXPROCS(1);debug.SetMemoryLimit(256<<20);_=syscall.Getrusage;_=time.Now;go physicalGuard()
 result:=map[string]any{"status":"running","scope":"train:0 source-derived first convolution only; crypto0","actual_crypto_operations":0};defer func(){if x:=recover();x!=nil{result["status"]="error";result["error"]=fmt.Sprint(x)};f,e:=os.Create(os.Args[2]);if e!=nil{panic(e)};defer f.Close();json.NewEncoder(f).Encode(result)}()
 raw,e:=os.ReadFile(os.Args[1]);if e!=nil{panic(e)};var sample struct{Input []float64 `json:"input"`};if e=json.Unmarshal(raw,&sample);e!=nil{panic(e)};if len(sample.Input)!=3072{panic("cached input shape")}
 initGlobals();p0,_,_,lt:=f_stem_prefix_preprocessing(nil,nil)
 weight:=[]float64{WEIGHTS};nativeWeight:=globals["__constant_16x3x3x3xf64"];for oc:=int64(0);oc<16;oc++{for ic:=int64(0);ic<3;ic++{for ky:=int64(0);ky<3;ky++{for kx:=int64(0);kx<3;kx++{i:=((oc*3+ic)*3+ky)*3+kx;if weight[i]!=nativeWeight.get(oc,ic,ky,kx){panic("frontend/native weight values differ")}}}}};result["weight_pairs_equal"]=432
 input:=tensor([]int64{1,3,32,32});input.s.dense=sample.Input;input.initialized=true;packed:=f_resnet20_silu__encrypt__arg0(nil,nil,nil,nil,input).data[0].(*V)
 result["packing_exact"]=checkPacking(packed.data,sample.Input,4096);if result["packing_exact"]!=true{panic("native packing differs")}
 masked:=binaryVec(packed,p0,nil,"mul");result["masked_input_exact"]=checkPacking(masked.data,sample.Input,4096);if result["masked_input_exact"]!=true{panic("native first mask differs")}
 observed:=applyTransform(masked,lt).data;reference:=referenceConv(sample.Input,weight,3,16,32,32);maximum,ratio:=0.0,0.0;bad:=-1
 for i,want:=range reference{delta:=math.Abs(observed[i]-want);r:=delta/(1e-12+1e-12*math.Abs(want));maximum=math.Max(maximum,delta);ratio=math.Max(ratio,r);if r>1&&bad<0{bad=i}}
 result["output_pairs"]=len(reference);result["max_absolute_error"]=maximum;result["max_tolerance_ratio"]=ratio;result["first_bad_slot"]=bad;result["passed"]=bad<0;result["native_lt_count"]=count["linear_transforms_applied"];result["counts"]=count;result["frozen_layouts"]=frozenLayouts
 if bad>=0{result["status"]="math_failed";result["first_bad_reference"]=reference[bad];result["first_bad_observed"]=observed[bad]}else{result["status"]="ok"}
}
'''.replace('WEIGHTS',','.join(repr(x) for x in flat))
            if args.activation_cut:
                convolutions=args.convolution_prefix or (3 if args.first_block_cut else 2 if args.second_activation_cut else 1)
                source,analysis,main=activation_source(program,rows,frontend,flat,convolutions=convolutions)
                result.update(round='HEIR-NATIVE-FIRST-ACTIVATION-CUT',mode='independent-native-first-activation',
                              scope='train:0 first conv/BN/normalization/SiLU strict prefix comparisons; no full model/quality/CKKS')
                if args.second_activation_cut:
                    result.update(round='HEIR-NATIVE-SECOND-ACTIVATION-CUT',mode='independent-native-second-activation',
                                  scope='train:0 first two conv/BN/normalization/SiLU strict prefix comparisons; no full model/quality/CKKS')
                if args.first_block_cut:
                    result.update(round='HEIR-NATIVE-FIRST-BLOCK-CUT',mode='independent-native-first-residual-block',
                                  scope='train:0 three conv/BN/SiLU and first residual addition strict prefix comparisons; no full model/quality/CKKS')
                if args.convolution_prefix:
                    result.update(round='HEIR-NATIVE-CONVOLUTION-PREFIX',mode='independent-native-convolution-prefix',
                                  scope='train:0 first '+str(convolutions)+' conv/BN/normalization/SiLU and original residual additions; no full model/quality/CKKS')
                write_json(out/'analysis.json',analysis)
            (out/'main.go').write_text(source+'\n'+STEM_RUNTIME+'\n'+main);(out/'go.mod').write_text('module heir-stem-cut\n\ngo 1.26\n')
            env=environment(out,256);env['HEIR_NATIVE_CLEAR_RSS_MIB']='512';binary=out/'stem-cut'
            for stage,command,seconds in [('build',[str(args.go),'build','-p=1','-buildvcs=false','-gcflags=-l','-o',str(binary),'.'],180),('cut',[str(binary),str(out/'sample.json'),str(out/'worker-result.json')],60)]:
                with recorder.span(stage,argv=command):
                    obs=run_process(command,cwd=out,log=out/(stage+'.log'),resources=out/(stage+'-resources.jsonl'),timeout=seconds,mem_gib=2,env=env)
                result['processes'].append(dict(stage=stage,argv=command,observation=obs))
                if obs['exit_code']!=0 or obs['timed_out'] or obs['remaining_pids']:raise RuntimeError(stage+' failed')
            result['worker']=json.loads((out/'worker-result.json').read_text());result['status']=result['worker']['status']
    except BaseException as exc:result['status']='error';result['error']=type(exc).__name__+': '+str(exc)
    finally:
        if (out/'worker-result.json').is_file():result['worker']=json.loads((out/'worker-result.json').read_text())
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')));result['ts']=datetime.now(timezone.utc).isoformat();finalize_size(out,result);register(out,result)
    print(json.dumps({k:result[k] for k in ('status','error','worker')}));return 0 if result['status']=='ok' else 1


def checked_exact_optiona_source(native,reference_source):
    """Accept the explicit equivalent frontend using retained original references.

    The old mathematical-reference artifact stays untouched. The new export
    separately proves every original tensor constant/activation coefficient and
    identifies the two fixed-public selector replacements.
    """
    record_path=Path(native.get('source_export_result',''))
    if not record_path.is_file():raise ValueError('Exact OptionA export source record is required')
    exported=json.loads(record_path.read_text());metadata=exported.get('export_record') or {};checks=exported.get('source_checks') or {}
    fixed_pool=native.get('exact_spatial_mean_matvec',False)
    if type(fixed_pool) is not bool:raise ValueError('Fixed spatial mean source option must be explicit boolean')
    if (exported.get('status')!='ok' or exported.get('model')!='resnet20_silu'
            or exported.get('exact_optiona_selector') is not True
            or metadata.get('exact_optiona_selector') is not True or metadata.get('graph_nodes')!=74
            or len(metadata.get('graph_operations',[]))!=74 or len(metadata.get('approximations',[]))!=19
            or checks.get('status')!='ok' or checks.get('same_saved_approximations')!=19
            or checks.get('original_constants')!=59 or checks.get('derived_fixed_selector_constants')!=2
            or checks.get('reference_changes')!=0 or checks.get('tolerance_changes')!=0):
        raise ValueError('Exact OptionA source constant/graph/approximation gate is incomplete')
    for key,expected in [('export_mlir',native.get('source','')),('original_frontend',reference_source.get('mlir','')),
                         ('graph_dir',reference_source.get('graph','')),('program',reference_source.get('program',''))]:
        if not expected or Path(exported.get(key,'')).resolve()!=Path(expected).resolve():
            raise ValueError('Exact OptionA original/new frontend source identity differs: '+key)
    selectors=metadata.get('exact_optiona_lowerings') or []
    if ([s.get('original_nodes') for s in selectors]!=[['n29','n30','n31'],['n53','n54','n55']]
            or any(s.get('trainable_weights_changed') is not False for s in selectors)):
        raise ValueError('Exact OptionA original two source triples differ')
    constants=checks.get('constants') or []
    if (len(constants)!=61+int(fixed_pool) or sum(r.get('name','').startswith('derived-exact-optiona:') for r in constants)!=2
            or not all(r.get('passed') is True and r.get('numerically_equal') is True for r in constants)):
        raise ValueError('Exact OptionA original/derived numerical constant inventory differs')
    proofs=exported.get('proofs') or []
    if len(proofs)!=3:raise ValueError('Exact OptionA prerequisite proof sources are incomplete')
    minimal=json.loads(Path(proofs[0]).read_text());tests=json.loads(Path(proofs[1]).read_text())
    small=minimal.get('worker') or {};samples=small.get('samples') or []
    if (minimal.get('status')!='ok' or small.get('status')!='ok' or small.get('coordinate_pairs')!=16384
            or small.get('coordinate_inputs_completed')!=2 or small.get('actual_crypto_operations')!=0
            or len(samples)!=2 or not all(r.get('passed') is True and r.get('input_ownership_preserved') is True and r.get('pairs')==8192 for r in samples)
            or tests.get('status')!='ok' or tests.get('tests',{}).get('count')!=5
            or tests.get('tests',{}).get('skipped') is not False):
        raise ValueError('Exact OptionA native coordinate equivalence or new adapter tests did not pass')
    pool_proof=None
    if fixed_pool:
        mean=metadata.get('exact_spatial_mean_lowerings') or [];pool_constants=[r for r in constants if r.get('name','').startswith('derived-exact-spatial-mean:')]
        if (exported.get('exact_spatial_mean_matvec') is not True or metadata.get('exact_spatial_mean_matvec') is not True
                or checks.get('derived_fixed_pool_constants')!=1 or len(mean)!=1 or len(pool_constants)!=1
                or pool_constants[0].get('name')!='derived-exact-spatial-mean:n72'
                or mean[0].get('original_node')!='n72' or mean[0].get('input_shape')!=[1,64,8,8]
                or mean[0].get('output_shape')!=[1,64] or mean[0].get('axes')!=[2,3]
                or mean[0].get('matrix_shape')!=[64,4096] or mean[0].get('coefficient')!=1/64
                or mean[0].get('nonzeros')!=4096 or mean[0].get('trainable_weights_changed') is not False
                or mean[0].get('bias') is not False):
            raise ValueError('Fixed pool source shape/coefficient inventory is incomplete')
        proof_paths=exported.get('fixed_pool_proofs') or []
        if len(proof_paths)!=2:raise ValueError('Fixed pool source proofs are required')
        tests=json.loads(Path(proof_paths[0]).read_text());minimal=json.loads(Path(proof_paths[1]).read_text());worker=minimal.get('worker') or {};samples=worker.get('samples') or []
        if (tests.get('status')!='ok' or tests.get('tests',{}).get('count')!=5 or tests.get('tests',{}).get('skipped') is not False
                or minimal.get('status')!='ok' or worker.get('status')!='ok' or worker.get('math_pairs')!=30
                or worker.get('actual_crypto_operations')!=0 or worker.get('actual_model_forwards')!=0
                or len(samples)!=3 or not all(r.get('passed') is True and r.get('pairs')==10
                    and r.get('actual_client_pack_verified') is True and r.get('input_ownership_preserved') is True for r in samples)):
            raise ValueError('Fixed pool actual minimal native coordinate proof did not pass')
        pool_proof=dict(original_node='n72',derived_public_constants=1,original_trainable_tensors_unchanged=True,
                        native_coordinate_proof=str(proof_paths[1]),native_pairs=30,reference_changes=0,tolerance_changes=0)
    elif any(r.get('name','').startswith('derived-exact-spatial-mean:') for r in constants):
        raise ValueError('Fixed pool constant requires its explicit source option')
    return dict(source_export_result=str(record_path),original_frontend_preserved=exported['original_frontend'],
                fixed_selector_triples=[s['original_nodes'] for s in selectors],original_constants=59,derived_selector_constants=2,
                saved_approximation_coefficients_preserved=19,minimal_native_equivalence=str(proofs[0]),
                original_reference_reused_by_exact_operator_equivalence=True,reference_changes=0,tolerance_changes=0,
                fixed_spatial_mean=pool_proof)


def checked_native_source(record_path):
    """Resolve a native artifact without changing a rejected original case.

    An independently recorded terminal audit may reuse a compiler child that
    exited normally while its parent rejected an incomplete final /proc sample.
    This special source records no additional compilation and retains the
    original parent result and signal observation.
    """
    record_path=Path(record_path)
    source=json.loads(record_path.read_text())
    if source.get('mode')=='independent-native-terminal-audit':
        if (source.get('status')!='ok' or source.get('native_compile_process_passed') is not True
                or source.get('original_complete_case_passed') is not False
                or source.get('compiler_runs')!=0 or source.get('original_result_preserved') is not True
                or not all(source.get('checks',{}).get(k) is True for k in
                           ('native_ir_complete','same_native_parameters','source_shape_identity','no_child_termination_signal'))):
            raise ValueError('Independent native compiler terminal audit is incomplete')
        original_path=Path(source.get('original_native_result',''))
        native=json.loads(original_path.read_text())
        observations=native.get('processes') or []
        if len(observations)!=1:raise ValueError('Exactly one original native compiler observation is required')
        observation=observations[0].get('observation') or {}
        ir=Path(source.get('compiled_artifact',''))
        if (native.get('status')!='error' or native.get('guard_termination',{}).get('reason')!='rss_metadata_missing'
                or native.get('attempts',{}).get('compile')!=1 or native.get('completed',{}).get('compile')!=0
                or observation.get('exit_code')!=0 or observation.get('timed_out') is not False
                or observation.get('signals_sent')!=[] or observation.get('remaining_pids')!=[]
                or ir.resolve()!=(original_path.parent/'model-lattigo.mlir').resolve()
                or source.get('model')!=native.get('model') or native.get('backend_executed') is not False):
            raise ValueError('Original compiler process completion evidence differs from the independent audit')
    else:
        native=source;ir=record_path.parent/'model-lattigo.mlir'
        if (native.get('status')!='ok' or native.get('model_compiled') is not True
                or native.get('completed',{}).get('compile')!=1 or native.get('backend_executed') is not False):
            raise ValueError('A completed saved native compiler artifact is required')
    if not ir.is_file() or not native.get('model'):raise ValueError('Native artifact or explicit model identity is absent')
    return native,ir


def checked_sources(args):
    native,ir=checked_native_source(args.native_result)
    emitted=json.loads(args.go_result.read_text())
    adapter=json.loads(args.adapter_result.read_text())
    worker=adapter.get('worker') or {}
    utils=args.go_result.parent/'module/heircnn_utils/heircnn_utils.go'
    if getattr(args,'common_model',None):
        return checked_common_sources(args,native,ir,emitted,adapter,utils)
    if native.get('model')!='resnet20_silu':raise ValueError('The saved native ResNet20 compiler artifact is required')
    if (emitted.get('status')!='ok' or emitted.get('model')!='resnet20_silu'
            or emitted.get('library_build_passed') is not True
            or Path(emitted.get('input_mlir','')).resolve()!=ir.resolve()
            or Path(emitted.get('source_compiler_result','')).resolve()!=args.native_result.resolve()):
        raise ValueError('Original generated Go source must identify this native artifact')
    if (adapter.get('status')!='ok' or worker.get('status')!='ok'
            or worker.get('activation_pairs')!=57 or len(worker.get('constants',[]))!=59
            or worker.get('reference_forward_completed')!=3 or worker.get('adapter_forward_completed')!=3
            or worker.get('quality_selection') is not False):
        raise ValueError('The saved three-probe adapter arithmetic regression is required')
    source=worker.get('source') or {}
    if (Path(source.get('probes','')).resolve()!=args.probes.resolve()
            or Path(source.get('probe_source','')).resolve()!=args.probe_source.resolve()):
        raise ValueError('Saved training probes and original trusted frontend source must match')
    if native.get('exact_optiona_selector') is True:checked_exact_optiona_source(native,source)
    elif Path(source.get('mlir','')).resolve()!=Path(native.get('source','')).resolve():
        raise ValueError('Original trusted frontend source must match')
    references=worker.get('probes') or []
    if [row.get('sample_id') for row in references]!=['train:0','train:1','train:2']:
        raise ValueError('Only saved train:0,1,2 mathematical probes are in scope')
    for row in references:
        ref=row.get('reference')
        if (not isinstance(ref,list) or len(ref)!=1 or len(ref[0])!=10
                or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in ref[0])
                or row.get('logit_check',{}).get('passed') is not True
                or len(row.get('activation_checks',[]))!=19
                or not all(check.get('passed') is True for check in row['activation_checks'])):
            raise ValueError('Saved strict reference or activation regression is incomplete')
    probe_source=json.loads(args.probe_source.read_text())
    if (probe_source.get('split')!='train' or probe_source.get('indices')!=[0,1,2]
            or probe_source.get('sample_ids')!=['train:0','train:1','train:2']):
        raise ValueError('Only original saved training probes may be replayed')
    for name in (ir,utils,args.probes,args.go):
        if not name.is_file():raise ValueError('Required existing source/tool is absent: '+str(name))
    return ir,utils,references,probe_source


def checked_common_sources(args,native,ir,emitted,adapter,utils):
    from .heir_common_models import read_graph_contract,checked_export_contract,checked_math_record,checked_saved_frontend_identity
    source=(adapter.get('worker') or {}).get('source') or {}
    contract=read_graph_contract(source.get('graph',''),args.common_model)
    metadata_path=Path(native.get('source','')).parent/'export.json'
    metadata=json.loads(metadata_path.read_text())
    checked_export_contract(metadata,contract)
    if (native.get('model')!=args.common_model or native.get('common_model')!=args.common_model
            or native.get('common_model_contract')!=contract
            or emitted.get('status')!='ok' or emitted.get('model')!=args.common_model
            or emitted.get('library_build_passed') is not True
            or Path(emitted.get('input_mlir','')).resolve()!=ir.resolve()
            or Path(emitted.get('source_compiler_result','')).resolve()!=args.native_result.resolve()
            or Path(source.get('mlir','')).resolve()!=Path(native.get('source','')).resolve()
            or Path(metadata.get('sources',{}).get('graph','')).resolve()!=Path(source.get('graph','')).resolve()
            or Path(metadata.get('sources',{}).get('program','')).resolve()!=Path(source.get('program','')).resolve()
            or Path(source.get('probes','')).resolve()!=args.probes.resolve()
            or Path(source.get('probe_source','')).resolve()!=args.probe_source.resolve()):
        raise ValueError('Common-model native, export, adapter and original source identities differ')
    references=checked_math_record(adapter,contract,metadata)
    original_program=Path(source['program'])
    checked_saved_frontend_identity(json.loads((original_program.parent.parent/'result.json').read_text()),original_program,args.common_model)
    from .heir_cnn_export import saved_activations
    _,configs=saved_activations(Path(source['program']).read_text())
    if (len(configs)!=contract['activation_count'] or any(
            any(json.loads(json.dumps(row.get(k)))!=json.loads(json.dumps(v)) for k,v in config.items())
            for config,row in zip(configs,metadata['approximations']))):
        raise ValueError('Common-model saved activation source differs')
    if native.get('exact_optiona_selector') is True:checked_exact_optiona_source(native,source)
    probe_source=json.loads(args.probe_source.read_text())
    if (probe_source.get('split')!='train' or probe_source.get('indices')!=[0,1,2]
            or probe_source.get('sample_ids')!=['train:0','train:1','train:2']):
        raise ValueError('Only original saved common-model training probes may be replayed')
    for name in (ir,utils,args.probes,args.go):
        if not name.is_file():raise ValueError('Required existing common-model source/tool is absent: '+str(name))
    return ir,utils,references,probe_source


def saved_samples(probes,references,*,input_shape=None):
    import numpy as np
    with np.load(probes,allow_pickle=False) as archive:
        images=archive['input'];indices=archive['indices']
    expected=(3,*input_shape[1:]) if input_shape is not None else (3,3,32,32)
    if (images.dtype!=np.float64 or images.shape!=expected
            or indices.tolist()!=[0,1,2] or not np.isfinite(images).all()):
        raise ValueError('Saved training probe dtype, shape, indices or finiteness differs')
    return [dict(sample_id=references[i]['sample_id'],input=images[i].reshape(-1).tolist(),
                 reference=references[i]['reference'][0]) for i in range(3)]


def environment(out,gc_mib):
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')
         and not k.startswith(('LD_','NIX_','__NIX_'))}
    for name in ('telemetry','build-cache','tmp'):(out/name).mkdir(exist_ok=False)
    (out/'telemetry/mode').write_text('off\n')
    env.update(GOTOOLCHAIN='local',GOPROXY='off',GOSUMDB='off',GOTELEMETRY='off',
               TEST_TELEMETRY_DIR=str(out/'telemetry'),GOCACHE=str(out/'build-cache'),
               TMPDIR=str(out/'tmp'),CGO_ENABLED='0',GOMAXPROCS='1',GOMEMLIMIT=str(gc_mib)+'MiB')
    return env


def register(out,result):
    if paths.RESULTS_DIR.resolve() not in out.resolve().parents:return
    entry=dict(run_id=out.name,round=result['round'],model=result['model'],mode=result['mode'],
               git_rev='working-tree',metrics={k:result[k] for k in ('status',*FORMAL)},
               artifacts_dir=str(out),ts=result['ts'])
    registry=paths.RESULTS_DIR/'registry.jsonl'
    with registry.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):
            raise ValueError('Result identity already registered')
        stream.seek(0,2);stream.write(json.dumps(entry)+'\n');stream.flush()
        fcntl.flock(stream,fcntl.LOCK_UN)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-result','go-result','adapter-result','probes','probe-source','go','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Read three saved probes and execute the independent clear replay')
    from .heir_common_models import COMMON_MODELS
    parser.add_argument('--common-model',choices=COMMON_MODELS)
    parser.add_argument('--stem-cut',action='store_true',help='Source-derived lightweight first-convolution diagnosis only')
    parser.add_argument('--activation-cut',action='store_true',help='Include first BN/normalization/SiLU source cuts')
    parser.add_argument('--second-activation-cut',action='store_true',help='Include the second original convolution/BN/normalization/SiLU source cuts')
    parser.add_argument('--first-block-cut',action='store_true',help='Include the third original convolution/BN and first residual addition/SiLU cuts')
    parser.add_argument('--convolution-prefix',type=int,choices=range(1,10),help='Original first nine convolution prefixes only')
    parser.add_argument('--optiona-coordinate-cut',action='store_true',help='Exact synthetic coordinate-tag regression of the original affine OptionA subgraph')
    parser.add_argument('--numpy-optiona-cut',action='store_true',help='Independent fresh-array NumPy SSA replay of the same affine coordinate subgraph')
    parser.add_argument('--numpy-optiona-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--worker-out',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--cached-probes',type=Path)
    parser.add_argument('--memory-gib',type=float,default=4)
    parser.add_argument('--rss-limit-gib',type=int,default=4)
    parser.add_argument('--compiler-gc-mib',type=int,default=512)
    parser.add_argument('--execute-seconds',type=int,default=1800,help='Finite mathematical replay allowance; at most1800seconds')
    args=parser.parse_args(argv)
    if args.common_model and any((args.stem_cut,args.activation_cut,args.second_activation_cut,args.first_block_cut,
                                  args.convolution_prefix,args.optiona_coordinate_cut,args.numpy_optiona_cut,args.numpy_optiona_worker)):
        parser.error('common-model selects the complete source-derived entry; legacy ResNet cuts have separate provenance')
    if args.numpy_optiona_worker:
        if args.worker_out is None:parser.error('Independent worker result path is required')
        return independent_optiona_numpy(args)
    if args.numpy_optiona_cut:return numpy_optiona_diagnostic(args)
    if args.optiona_coordinate_cut:return optiona_diagnostic(args)
    if args.second_activation_cut or args.first_block_cut or args.convolution_prefix:args.activation_cut=True
    if args.stem_cut:
        if args.cached_probes is None:parser.error('stem-cut requires the existing cached-training-probes JSON path')
        return stem_diagnostic(args)
    if not 0<args.memory_gib<=12 or not 1<=args.rss_limit_gib<=8 or args.rss_limit_gib>args.memory_gib:
        parser.error('This clear diagnostic supports AS at most12GiB, RSS at most8GiB, and host available at least1GiB')
    if not 0<args.execute_seconds<=1800:parser.error('Mathematical replay seconds must be1 through1800')
    args.out=args.out.absolute()
    for source in (args.native_result.parent,args.go_result.parent,args.adapter_result.parent,args.probes.parent):
        if args.out.resolve()==source.resolve() or source.resolve() in args.out.resolve().parents:
            parser.error('New diagnostic output must preserve every original source artifact directory')
    args.out.mkdir(parents=True,exist_ok=False)
    out=args.out;recorder=StageRecorder(out/'events.jsonl')
    result=dict(round='HEIR-NATIVE-CLEAR',model=args.common_model or 'resnet20_silu',mode='independent-native-arithmetic-regression',
                status='running',error=None,source={n:str(getattr(args,n).absolute()) for n in ('native_result','go_result','adapter_result','probes','probe_source')},
                attempts=dict(source=0,build=0,probe_read=0,execute=0),completed=dict(source=0,build=0,probe_read=0,execute=0),
                processes=[],worker=None,artifact_bytes=None,quality_selection=False,candidate_frozen=False,
                heldout_samples=0,confirm_samples=0,report_samples=0,training_samples_for_selection=0,
                keygen_runs=0,encryptions=0,bootstraps=0,decryptions=0,real_ckks_attempts=0,real_ckks_completed=0,
                llm_calls=0,compiler_runs=0,calibration_attempts=0,plan_attempts=0,count_attempts=0,emit_attempts=0,
                downloads=0,installations=0,original_libraries_modified=False,security_validated=False,
                backend_executed=False,full_model_ckks_passed=False,**{name:None for name in FORMAL},
                resource_policy=dict(address_space_gib=args.memory_gib,rss_limit_gib=args.rss_limit_gib,minimum_host_available_gib=1,
                                     go_heap_soft_limit_gib=max(1,args.rss_limit_gib-2),build_address_space_gib=min(args.memory_gib,4)),
                scope='Saved native arithmetic with mathematical identities for native level/scale/bootstrap operations; three saved training probes only')
    env=None
    def save():write_json(out/'result.json',result)
    def run(stage,command,seconds):
        if shutil.disk_usage(out).free<4*1024**3:raise RuntimeError('Preserve 4GiB disk reserve')
        result['attempts'][stage]+=1;save()
        with recorder.span(stage,argv=command):
            observation=run_process(command,cwd=out,log=out/(stage+'.log'),resources=out/(stage+'-resources.jsonl'),
                                    timeout=seconds,mem_gib=min(args.memory_gib,4) if stage=='build' else args.memory_gib,env=env)
        result['processes'].append(dict(stage=stage,argv=command,observation=observation));save()
        if observation['exit_code']!=0 or observation['timed_out'] or observation['signal'] is not None or observation['remaining_pids']:
            raise RuntimeError(stage+' failed; original native/model records remain unchanged')
        result['completed'][stage]+=1;save()
    try:
        with recorder.span('native-clear-math'):
            result['attempts']['source']=1;save()
            with recorder.span('source-translation'):
                ir,utils,references,probe_source=checked_sources(args)
                if args.common_model:
                    result['common_model']=args.common_model
                    result['common_model_contract']=(json.loads(args.adapter_result.read_text())['worker']['common_model_contract'])
                source_native,_=checked_native_source(args.native_result)
                if source_native.get('exact_optiona_selector') is True:
                    adapter_source=(json.loads(args.adapter_result.read_text()).get('worker') or {}).get('source') or {}
                    result['exact_optiona_source_equivalence']=checked_exact_optiona_source(source_native,adapter_source)
                from .heir_native_clear_ir import parse
                from .heir_native_clear_codegen import Emitter,native_transform_rows
                text=ir.read_text();program=parse(text);rows=native_transform_rows(program,utils.read_text())
                resources=set(re.findall(r'preprocessing.load_resource "([^\"]+)"',text))
                for source in resources:
                    path=Path(source)
                    if ir.parent.resolve() not in path.resolve().parents or not path.is_file():
                        raise ValueError('Native external constant must exist inside its original saved compile artifact')
                if args.common_model:
                    contract=result['common_model_contract']
                    source,analysis=Emitter(program,rows).emit(args.common_model,input_shape=contract['input_shape'],output_shape=contract['output_shape'])
                else:source,analysis=Emitter(program,rows).emit('resnet20_silu')
                write_json(out/'analysis.json',analysis);write_json(out/'transform-rows.json',rows)
                (out/'main.go').write_text(source);(out/'go.mod').write_text('module heir-native-clear\n\ngo 1.26\n')
                result['native_source']=dict(ir=str(ir),utils=str(utils),resources=len(resources),
                                             resource_bytes=sum(Path(p).stat().st_size for p in resources),
                                             operations=program.operation_counts,index_checks=len(analysis['index_relation_checks']),
                                             SSA_pointer_releases=analysis['SSA_pointer_releases'],
                                             transform_order='Native FindBestBSGSRatio and sorted giant/baby groups; source rows taken from generated Go; f64 multiplication and separate additions',
                                             template='bounded SSA blocks with exact native last-use deletions')
                result['completed']['source']=1;save()
            env=environment(out,args.compiler_gc_mib)
            env.update(HEIR_NATIVE_CLEAR_RSS_GIB=str(args.rss_limit_gib),HEIR_NATIVE_CLEAR_HEAP_GIB=str(max(1,args.rss_limit_gib-2)))
            binary=out/'native-clear'
            run('build',[str(args.go.absolute()),'build','-p=1','-mod=readonly','-buildvcs=false','-gcflags=-l','-o',str(binary),'.'],300)
            if args.execute:
                result['attempts']['probe_read']=1;save()
                with recorder.span('saved-training-probe-read'):
                    samples=saved_samples(args.probes,references,input_shape=result['common_model_contract']['input_shape'] if args.common_model else None);write_json(out/'saved-training-probes.json',samples)
                result['completed']['probe_read']=1;save()
                run('execute',[str(binary),str(out/'saved-training-probes.json'),str(out/'worker-result.json')],args.execute_seconds)
                result['worker']=json.loads((out/'worker-result.json').read_text())
                worker=result['worker']
                if (worker.get('status')!='ok' or worker.get('native_math_attempts')!=3
                        or worker.get('native_math_completed')!=3 or worker.get('math_probes_completed')!=3
                        or any(worker.get(k)!=0 for k in ('keygen_runs','encryptions','bootstraps','decryptions'))
                        or len(worker.get('samples',[]))!=3 or not all(r.get('passed') is True for r in worker['samples'])):
                    raise RuntimeError('Independent native mathematical replay did not pass the original strict criterion')
            result['status']='ok' if args.execute else 'prepared'
    except BaseException as error:
        result['status']='error';result['error']=type(error).__name__+': '+str(error)
    finally:
        if (out/'worker-result.json').is_file():result['worker']=json.loads((out/'worker-result.json').read_text())
        if (out/'worker-result.json.resource-guard.json').is_file():result['resource_guard']=json.loads((out/'worker-result.json.resource-guard.json').read_text())
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')))
        result['ts']=datetime.now(timezone.utc).isoformat();finalize_size(out,result);register(out,result)
    print(json.dumps({k:result[k] for k in ('status','error','attempts','completed','artifact_bytes')}))
    return 0 if result['status'] in ('ok','prepared') else 1


if __name__=='__main__':raise SystemExit(main())
