"""Synthetic source and small-vector checks for independent native clear replay."""
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from pt_agent.experiments.heir_native_clear_ir import Function,Node,Unsupported,parse,transfer_region,check_functional_destination
from pt_agent.experiments.heir_native_clear_codegen import Emitter,RUNTIME,native_transform_rows


def transfer_fixture(target='%i'):
    return '''module {
  memref.global "private" constant @zero : memref<4xf64> = dense<0.0>
  func.func private @_assign_layout_small(%arg0: memref<4xf64>) -> memref<4xf64> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %zero = memref.get_global @zero : memref<4xf64>
    %out = memref.alloc() : memref<4xf64>
    memref.copy %zero, %out : memref<4xf64> to memref<4xf64>
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %value = memref.load %arg0[%i] : memref<4xf64>
      memref.store %value, %out[TARGET] : memref<4xf64>
    }
    return %out : memref<4xf64>
  }
}'''.replace('TARGET',target)


def region(program):
    fn=next(iter(program.functions.values()))
    loop=next(n for n in fn.body if n.kind=='for')
    return transfer_region(loop,{'%c0':('c',0),'%c1':('c',1),'%c4':('c',4)})


def projection_fixture(extra_consumer=False):
    text='''module {
  func.func private @_assign_layout_projected(%arg0: memref<2x4xf64>) -> memref<2x4xf64> {
    %out = memref.alloc() : memref<2x4xf64>
    memref.copy %arg0, %out : memref<2x4xf64> to memref<2x4xf64>
    return %out : memref<2x4xf64>
  }
  func.func @owner(%param: !param, %enc: !encoder, %input: memref<2x4xf64>) -> !linear_transformation {
    %layout = call @_assign_layout_projected(%input) : (memref<2x4xf64>) -> memref<2x4xf64>
    EXTRA
    %lt = lattigo.ckks.prepare_linear_transform %param, %enc, %layout <diagonal_indices = [3], logSlots = 2 : i64> : (!param,!encoder,memref<2x4xf64>) -> !linear_transformation
    return %lt : !linear_transformation
  }
}'''.replace('EXTRA','%other = memref.cast %layout : memref<2x4xf64> to memref<2x4xf64>' if extra_consumer else '')
    program=parse(text);node=next(n for n in program.functions['owner'].body if n.kind=='lattigo.ckks.prepare_linear_transform')
    return program,[dict(line=node.line,diagonals=[3],source_rows=[1],logSlots=2)]


class NativeSourceChecks(unittest.TestCase):
    def test_float64_raw_global_is_not_interpreted_as_a_splat(self):
        raw=struct.pack('<2d',1.25,-2.5).hex()
        program=parse('module {\n memref.global "private" constant @raw : memref<2xf64> = dense<"0x'+raw+'">\n func.func @f() {\n return\n }\n}')
        self.assertEqual(program.globals['raw']['raw_hex'],raw)
        self.assertNotIn('fill',program.globals['raw'])

    def test_grouped_ssa_result_names_preserved(self):
        program=parse('module {\n func.func @f() {\n %a:2 = call @g() : () -> (i64,i64)\n return %a#0, %a#1 : i64,i64\n }\n}')
        self.assertEqual(program.functions['f'].body[0].defs,['%a#0','%a#1'])

    def test_unknown_arithmetic_fails_explicitly(self):
        f=Function('unknown',1,[],['i64'],[Node('arith.divsi',2,'arith.divsi %a, %b : i64',['%x'])])
        with self.assertRaisesRegex(Unsupported,'Integer operation'):
            Emitter(None,[]).function(f)

    def test_original_chebyshev_basis_uses_full_c0(self):
        import numpy as np
        from pt_agent.kernels.poly import cheb_eval
        u=np.array([-1.,-.25,0.,.5,1.],dtype=np.float64)
        expected=3.+2.*u+4.*(2.*u*u-1.)
        np.testing.assert_array_equal(cheb_eval(None,u,[3.,2.,4.]),expected)

    def test_uniform_plaintext_source_tracks_view_alias_writes(self):
        from types import SimpleNamespace
        from pt_agent.experiments.heir_native_clear_math import uniform_plaintext_values
        body=[Node('arith.constant',1,'arith.constant 0 : index',['%c0']),
              Node('memref.get_global',2,'memref.get_global @uniform : memref<4xf64>',['%global']),
              Node('memref.alloc',3,'memref.alloc() : memref<4xf64>',['%copy']),
              Node('memref.copy',4,'memref.copy %global, %copy : memref<4xf64> to memref<4xf64>'),
              Node('memref.subview',5,'memref.subview %copy[0] [4] [1] : memref<4xf64> to memref<4xf64>',['%view']),
              Node('lattigo.ckks.encode',6,'lattigo.ckks.encode %encoder, %view, %empty : (!encoder,memref<4xf64>,!pt) -> !pt',['%encoded']),
              Node('memref.store',7,'memref.store %encoded, %plains[%c0] : memref<1x!pt>')]
        function=Function('resnet20_silu__preprocessing',1,[],[],body)
        program=SimpleNamespace(globals={'uniform':{'fill':.25}},functions={function.name:function})
        self.assertEqual(uniform_plaintext_values(program),{0:.25})
        body.insert(5,Node('memref.store',6,'memref.store %unknown, %copy[%c0] : memref<4xf64>'))
        self.assertEqual(uniform_plaintext_values(program),{0:None})

    def test_optiona_affine_cut_rejects_data_dependent_products(self):
        from types import SimpleNamespace
        from pt_agent.experiments.heir_native_clear_math import optiona_source
        body=[Node('lattigo.ckks.bootstrap',1,'lattigo.ckks.bootstrap %boot, %prior : (!boot,!ct) -> !ct',['%ct_2808']),
              Node('lattigo.ckks.bootstrap',2,'lattigo.ckks.bootstrap %boot, %prior : (!boot,!ct) -> !ct',['%ct_3716']),
              Node('lattigo.ckks.bootstrap',3,'lattigo.ckks.bootstrap %boot, %prior : (!boot,!ct) -> !ct',['%ct_3']),
              Node('memref.load',4,'memref.load %arg1[%c0] : memref<1x!pt>',['%pt_3717']),
              Node('lattigo.ckks.mul',5,'lattigo.ckks.mul %eval, %ct_2808, %ct_3716, %ct_3 : (!eval,!ct,!ct,!ct) -> !ct',['%ct_3718']),
              Node('lattigo.ckks.add',6,'lattigo.ckks.add %eval, %ct_3718, %ct_3716, %ct_3 : (!eval,!ct,!ct,!ct) -> !ct',['%ct_4882'])]
        function=Function('resnet20_silu__preprocessed',1,[],['!ct'],body)
        with self.assertRaisesRegex(ValueError,'nonlinear'):
            optiona_source(SimpleNamespace(functions={function.name:function}),[])

    def test_independent_numpy_steps_preserve_input_values_and_ownership(self):
        import numpy as np
        from pt_agent.experiments.heir_native_clear_math import _numpy_affine_step
        original=np.array([1.,2.,3.,4.],dtype=np.float64);original.setflags(write=False)
        rotated=_numpy_affine_step(np,'rotate',[original],rotation=1)
        np.testing.assert_array_equal(rotated,np.array([2.,3.,4.,1.]))
        np.testing.assert_array_equal(_numpy_affine_step(np,'rotate',[original],rotation=-1),np.array([4.,1.,2.,3.]))
        squared=_numpy_affine_step(np,'mul',[original,original]);summed=_numpy_affine_step(np,'add',[squared,rotated])
        np.testing.assert_array_equal(summed,np.array([3.,7.,13.,17.]))
        copied=_numpy_affine_step(np,'bootstrap',[original]);copied[0]=99
        np.testing.assert_array_equal(original,np.array([1.,2.,3.,4.]))
        self.assertFalse(np.shares_memory(squared,original));self.assertFalse(np.shares_memory(rotated,original))
        with self.assertRaisesRegex(ValueError,'unsupported'):_numpy_affine_step(np,'unknown',[original])

    def test_source_coordinate_inversion_is_explicit(self):
        r=region(parse(transfer_fixture()))
        self.assertEqual(r['known'],{'%i':0})
        self.assertEqual(r['source_shape'],[4])
        self.assertEqual(r['destination_shape'],[4])
        self.assertEqual(r['bounds'],{'%i':(0,3)})

    def test_sparse_region_rejects_changed_loaded_value(self):
        text=transfer_fixture().replace('memref.store %value','memref.store %c1')
        with self.assertRaisesRegex(Unsupported,'unchanged loaded f64'):
            region(parse(text))

    def test_source_row_mapping_is_not_guessed_from_diagonals(self):
        p=parse('module {\n func.func @f() {\n %lt = lattigo.ckks.prepare_linear_transform %p, %e, %t <diagonal_indices = [3, 1], logSlots = 2 : i64> : (!param,!encoder,memref<2x4xf64>) -> !linear_transformation\n return\n }\n}')
        rows=native_transform_rows(p,'a_diags_idx := []int{3,1}\na_source_rows := []int{0,1}')
        self.assertEqual(rows[0]['diagonals'],[3,1]);self.assertEqual(rows[0]['source_rows'],[0,1])
        with self.assertRaisesRegex(Unsupported,'unverified'):
            native_transform_rows(p,'a_diags_idx := []int{3,1}')

    def test_large_function_uses_bounded_ssa_blocks(self):
        body=[Node('arith.constant',i+2,'arith.constant '+str(i)+' : i64',['%c'+str(i)]) for i in range(401)]
        body.append(Node('return',404,'return %c400 : i64'))
        f=Function('large',1,[],['i64'],body)
        source=Emitter(None,[]).function(f)
        self.assertEqual(source.count('(env map[string]any)'),3)
        self.assertIn('env["%c400"]=int64(400)',source)
        self.assertIn('return env["return:0"].(int64)',source)

    def test_native_row_projection_requires_the_complete_unique_access_set(self):
        program,rows=projection_fixture();emitter=Emitter(program,rows,prove=False)
        self.assertEqual(emitter.projection_evidence[0]['required_source_rows'],1)
        self.assertIn('.project(projection)',emitter.function(program.functions['_assign_layout_projected']))
        self.assertIn('[]int{1}',emitter.function(program.functions['owner']))
        program,rows=projection_fixture(extra_consumer=True)
        with self.assertRaisesRegex(Unsupported,'exactly one transform consumer'):Emitter(program,rows)

    def test_projected_helper_internal_alias_or_read_is_rejected(self):
        program,rows=projection_fixture();helper=program.functions['_assign_layout_projected']
        helper.body.insert(-1,Node('memref.cast',9,'memref.cast %out : memref<2x4xf64> to memref<2x4xf64>',['%alias']))
        emitter=Emitter(program,rows,prove=False)
        with self.assertRaisesRegex(Unsupported,'internal read or alias'):emitter.function(helper)


class NativeIntegerRelationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pt_agent.tests._artifacts import require
        try:
            import gurobipy
        except ImportError:raise unittest.SkipTest('Uses existing approved Orbit Gurobi environment')
        require(Path(gurobipy.__file__).parent/'.libs/gurobi.lic')

    def test_injective_sparse_copy_relation_passes(self):
        check=check_functional_destination(region(parse(transfer_fixture())))
        self.assertTrue(check['passed']);self.assertTrue(check['source_coordinate_functional'])

    def test_overwriting_different_source_coordinates_is_rejected(self):
        with self.assertRaisesRegex(Unsupported,'unproved'):
            check_functional_destination(region(parse(transfer_fixture('%c0'))))

    def test_invalid_zero_store_address_is_rejected_before_sparse_skip(self):
        with self.assertRaisesRegex(Unsupported,'constant address outside'):
            check_functional_destination(region(parse(transfer_fixture('%c4'))))


class NativeSmallGoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pt_agent import paths
        from pt_agent.tests._artifacts import require
        cls.go=Path(os.environ.get('HEIR_NATIVE_CLEAR_GO',str(paths.WORKSPACE/'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/go')))
        require(cls.go)

    def test_small_native_arithmetic_sparse_shapes_and_chunk_execution(self):
        from pt_agent.experiments.r10_paper_metrics import run_process
        from pt_agent.experiments.heir_native_clear_math import environment
        from pt_agent.experiments.heir_native_clear_math import STEM_RUNTIME
        body=[Node('arith.constant',i+2,'arith.constant '+str(i)+' : i64',['%c'+str(i)]) for i in range(401)]
        body.append(Node('return',404,'return %c400 : i64'))
        source=Emitter(None,[]).function(Function('large',1,[],['i64'],body))
        program=parse(transfer_fixture());emitter=Emitter(program,[],prove=False)
        source+='\n'+emitter.function(program.functions['_assign_layout_small'])
        program,rows=projection_fixture();emitter=Emitter(program,rows,prove=False)
        source+='\n'+emitter.function(program.functions['_assign_layout_projected'])
        source+='\n'+emitter.function(program.functions['owner'])
        f=Function('tap_alias',1,[('%e','!evaluator'),('%a','!ct')],['!ct'],[
            Node('lattigo.ckks.add',2,'lattigo.ckks.add %e, %a, %a, %a : (!evaluator,!ct,!ct,!ct) -> !ct',['%b']),
            Node('lattigo.ckks.add',3,'lattigo.ckks.add %e, %b, %b, %b : (!evaluator,!ct,!ct,!ct) -> !ct',['%c']),
            Node('return',4,'return %c : !ct')])
        source+='\n'+Emitter(None,[],taps={('tap_alias',2):'before_second_mutation'}).function(f)
        main=r'''
func main(){
 runtime.GOMAXPROCS(1);debug.SetMemoryLimit(256<<20);_ =syscall.Getrusage;_ =time.Now
 globals["zero"]=splat([]int64{4},0)
 a:=tensor([]int64{4});a.set(-1,0);a.set(3,2)
 copied:=f__assign_layout_small(a)
 huge:=tensor([]int64{4294967296});huge.initialized=true;huge.set(7,4294967295);seen:=0;huge.nonzero(func(ix []int64,v float64){if ix[0]!=4294967295||v!=7{panic("sparse huge shape")};seen++})
 matrix:=tensor([]int64{4,4});matrix.initialized=true;matrix.set(2,1,0);matrix.set(-4,3,2)
 tr:=prepareTransform(matrix,[]int{1,3},[]int{1,3},4)
 x:=&V{data:[]float64{1,3,5,7},level:3};y:=applyTransform(x,tr)
 r:=rotate(x,nil,1);negative:=rotate(x,nil,-1)
 same:=identity(x,nil,"bootstrap");x.data[0]=9
 z:=binaryVec(&V{data:[]float64{1,2,3,4}},&V{data:[]float64{5,6,7,8}},nil,"mul")
 projectedInput:=tensor([]int64{2,4});projectedInput.initialized=true;for i:=int64(0);i<4;i++{projectedInput.set(float64(i+1),1,i)}
 projectedTransform:=f_owner(nil,nil,projectedInput);projectedOutput:=applyTransform(&V{data:[]float64{1,3,5,7}},projectedTransform)
 if len(projectedTransform.entries)!=4||projectedOutput.data[0]!=7||projectedOutput.data[1]!=2||projectedOutput.data[2]!=9||projectedOutput.data[3]!=20{panic("native row projection")}
 groupedMatrix:=tensor([]int64{4,4});groupedMatrix.initialized=true;groupedMatrix.set(1e16,0,0);groupedMatrix.set(1,1,0);groupedMatrix.set(-1e16,2,0);groupedMatrix.set(1,3,0)
 grouped:=prepareTransform(groupedMatrix,[]int{0,1,2,3},[]int{0,1,2,3},4)
 groupedOutput:=applyTransform(&V{data:[]float64{1,1,1,1}},grouped)
 if grouped.n1!=2||groupedOutput.data[0]!=0{panic("native BSGS floating accumulation groups")}
 wideCoordinates:=tensor([]int64{1,16384});wideCoordinates.initialized=true;wideCoordinates.set(2,0,16383)
 packedCoordinates:=prepareTransform(wideCoordinates,[]int{16383},[]int{0},16384)
 if len(packedCoordinates.entries)!=1||packedCoordinates.entries[0].diag!=16383||packedCoordinates.entries[0].slot!=16383||packedCoordinates.entries[0].value!=2{panic("lossless native int32 coordinate storage")}
 coordinateOverflow:=false;func(){defer func(){if recover()!=nil{coordinateOverflow=true}}();prepareTransform(tensor([]int64{1,2147483648}),[]int{0},[]int{0},2147483648)}();if !coordinateOverflow{panic("native compact coordinate bound")}
 frozen:=tensor([]int64{1,4});frozen.project([]int{0});frozen.initialized=true;frozen.set(3,0,2);frozen.set(3,0,2);frozen.compact();if len(frozen.s.packed)!=1||frozen.get(0,2)!=3||frozen.get(0,1)!=0{panic("compact native duplicate value")}
 exactCapacity:=prepareTransform(frozen,[]int{3},[]int{0},4);if len(exactCapacity.entries)!=1||cap(exactCapacity.entries)!=1||exactCapacity.entries[0].value!=3{panic("frozen transform exact capacity preserves entries")}
 readonlyFailure:=false;func(){defer func(){if recover()!=nil{readonlyFailure=true}}();frozen.set(4,0,2)}();if !readonlyFailure{panic("compact readonly guard")}
 bad:=tensor([]int64{1,4});bad.project([]int{0});bad.initialized=true;bad.set(1,0,0);bad.set(2,0,0)
 mismatchFailure:=false;func(){defer func(){if recover()!=nil{mismatchFailure=true}}();bad.compact()}();if !mismatchFailure{panic("different duplicate native values must be rejected")}
 convSmall:=referenceConv([]float64{1,2,3,4},[]float64{1,0,0,0,2,0,0,0,3},1,1,2,2)
 if convSmall[0]!=14||convSmall[1]!=4||convSmall[2]!=6||convSmall[3]!=9{panic("independent padded convolution boundary/orientation")}
 strideSmall:=referenceConvStride([]float64{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},[]float64{0,0,0,0,1,0,0,0,0},1,1,4,4,2)
 if len(strideSmall)!=4||strideSmall[0]!=1||strideSmall[1]!=3||strideSmall[2]!=9||strideSmall[3]!=11{panic("independent stride2 convolution boundary")}
 transitionInput:=make([]float64,8192);for i:=range transitionInput{transitionInput[i]=float64(i+1)};transitionPacked:=packFirstTransition(transitionInput)
 for c:=0;c<32;c++{for h:=0;h<16;h++{for w:=0;w<16;w++{q,r:=(c+1)/2,(c+2)/4;group:=c-q-r;row:=-c+2*h+q+2*r;col:=-(c+1-2*q)+2*w+1;slot:=group*1024+row*32+col;want:=transitionInput[(c*16+h)*16+w];if transitionPacked[slot]!=want||transitionPacked[slot+8192]!=want{panic("native first transition integer map and two-half period")}}}}
 transitionPadded:=padFirstTransition(transitionInput);for c:=0;c<32;c++{for h:=0;h<18;h++{for w:=0;w<18;w++{want:=0.0;if h>0&&h<17&&w>0&&w<17{want=transitionInput[(c*16+h-1)*16+w-1]};if transitionPadded[c*324+h*18+w]!=want{panic("original NCHW zero padding at transition")}}}};for _,v:=range transitionPadded[10368:]{if v!=0{panic("original transition padding tail")}}
 optionInput:=make([]float64,16384);for i:=range optionInput{optionInput[i]=float64(i+1)};optionOutput:=firstOptionA(optionInput);repeatedOption:=repeatLogical(optionOutput)
 for c:=0;c<32;c++{for h:=0;h<16;h++{for w:=0;w<16;w++{want:=0.0;if c>=8&&c<24{want=optionInput[((c-8)*32+2*h)*32+2*w]};slot:=(c*16+h)*16+w;if optionOutput[slot]!=want||repeatedOption[slot]!=want||repeatedOption[slot+8192]!=want{panic("original first OptionA stride and channel zeros")}}}}
 if !checkPacking([]float64{1,2,3,0,1,2,3,0},[]float64{1,2,3},4)||checkPacking([]float64{1,2,3,0,9,2,3,0},[]float64{1,2,3},4){panic("packing period/gap regression")}
 alias:=f_tap_alias(nil,&V{data:[]float64{1}});if alias.data[0]!=4||captured["before_second_mutation"].data[0]!=2{panic("native intermediate cut ownership")}
 ownedReference:=&V{data:[]float64{1}};_=f_tap_alias(nil,clone(ownedReference));if ownedReference.data[0]!=1{panic("native input copy must preserve synthetic expected coordinates")}
 if f_large()!=400||seen!=1||copied.get(0)!=-1||copied.get(1)!=0||copied.get(2)!=3||y.data[0]!=6||y.data[2]!=-12||r.data[0]!=3||negative.data[0]!=7||same.data[0]!=1||z.data[3]!=32{panic("small arithmetic mismatch")}
 failed:=false;func(){defer func(){if recover()!=nil{failed=true}}();copied.expectZero()}();if !failed{panic("destination overwrite guard")}
 json.NewEncoder(os.Stdout).Encode(map[string]any{"passed":true,"chunk":400,"sparse_entries":seen,"transform":y.data,"strict_math_tolerance_unchanged":true,"crypto_operations":0})
}
'''
        with tempfile.TemporaryDirectory(prefix='heir-native-clear-small-') as temporary:
            out=Path(temporary);(out/'main.go').write_text(RUNTIME+'\n'+STEM_RUNTIME+'\n'+source+'\n'+main)
            (out/'go.mod').write_text('module native-clear-small\n\ngo 1.26\n')
            env=environment(out,384)
            build=run_process([str(self.go),'build','-p=1','-buildvcs=false','-gcflags=-l','-o',str(out/'probe'),'.'],cwd=out,log=out/'build.log',resources=out/'build-resources.jsonl',timeout=180,mem_gib=2,env=env)
            self.assertEqual(build['exit_code'],0,(out/'build.log').read_text()[-6000:])
            observed=run_process([str(out/'probe')],cwd=out,log=out/'run.log',resources=out/'run-resources.jsonl',timeout=30,mem_gib=1,env=env)
            self.assertEqual(observed['exit_code'],0,(out/'run.log').read_text())
            row=json.loads((out/'run.log').read_text())
            self.assertEqual(row['transform'],[6,0,-12,0]);self.assertEqual(row['crypto_operations'],0)


if __name__=='__main__':unittest.main()
