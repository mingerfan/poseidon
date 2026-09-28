"""Finite directed public probes: not FHE evidence or formal equivalence."""
import ast
import copy
import unittest
import numpy as np
from benchmark_graph import samples,digest
from benchmark_math import evaluate
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate,validate_request
from unified_graph_lowering import lower
from unified_public_contract import CONTRACT,normalize,event_record
from unified_public_exercises import SPECS,golden_variant,STRUCTURAL
from unified_public_coverage import verify_trace_coverage,fingerprint
from test_unified_public import model
from test_unified_graph import probe

class PublicCoverageTests(unittest.TestCase):
    def request(self,name,g=None):
        return prepare(g or model(),PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),
                       name,construction_profile=CONTRACT)
    def test_all_expression_partitions_independent_math_and_trace_binding(self):
        for name in SPECS:
            with self.subTest(name=name):
                req=self.request(name);source=golden_variant(lower(req),name,req)
                checked=validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)
                witness=checked["construction_exercise"]
                events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
                record=dict(events=events,normalized_sha256=expanded["construction"]["normalized_sha256"],
                            candidate_python_executed=False)
                result=verify_trace_coverage(witness,record)
                structural=SPECS[name][0] in STRUCTURAL
                self.assertEqual(result["finite_influence_checked"],not structural)
                self.assertEqual(result["structural_only"],structural)
                self.assertFalse(witness["plaintext_reference_used"])
                self.assertLessEqual(witness["intervention_attempts"],80)
                for inputs in samples(req["model"],4):
                    expected=np.concatenate([v.reshape(-1) for v in evaluate(req["model"],inputs).values()])
                    np.testing.assert_allclose(probe(dict(req,public_constants=expanded["constants"]),expanded["source"],inputs),
                                               expected,atol=1e-12,rtol=1e-12)
                forged=copy.deepcopy(record);forged["events"]=[]
                with self.assertRaises(ValueError):verify_trace_coverage(witness,forged)
    def test_binding_declarations_without_parameter_influence_rejected(self):
        cases={
            "signature-posonly":["def f(v,/):\n return 0.5\ncoverage_bias=f(0.25)",
                                 "def f(v,/):\n v=0.5\n return v\ncoverage_bias=f(0.25)"],
            "signature-kwonly":["def f(*,v):\n return 0.5\ncoverage_bias=f(v=0.25)",
                                "def f(*,v):\n v=0.5\n return v\ncoverage_bias=f(v=0.25)"],
            "signature-vararg":["f=lambda *values:0.5\ncoverage_bias=f(0.25)","def f(*values):\n return 0.5\ncoverage_bias=f(0.25)",
                                "def f(*values):\n values=(0.5,)\n return values[0]\ncoverage_bias=f(0.25)",
                                "def f(*values):\n return len(values)*0.25\ncoverage_bias=f(0.1,0.2)"],
            "signature-kwarg":["f=lambda **values:0.5\ncoverage_bias=f(v=0.25)","def f(**values):\n return 0.5\ncoverage_bias=f(v=0.25)",
                               "def f(**values):\n values={'v':0.5}\n return values['v']\ncoverage_bias=f(v=0.25)",
                               "def f(**values):\n return len(values)*0.5\ncoverage_bias=f(v=0.25)"],
        }
        for feature,bodies in cases.items():
            req=self.request("unified-public-"+feature)
            for body in bodies:
                source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+line+'\n' for line in (body+'\nreturn [x+coverage_bias,y]').splitlines())
                with self.subTest(feature=feature,body=body),self.assertRaisesRegex(ValueError,"Missing contributing"):
                    validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)

    def test_expansion_ignored_empty_and_side_effect_only_rejected(self):
        cases={
            "expansion-star":[
                "def f(*values):\n return 0.5\ncoverage_bias=f(*[0.25])",
                "def f(v=0.5):\n return v\ncoverage_bias=f(*[])",
                "def f(v,*values):\n return v\ncoverage_bias=f(0.5,*[0.25])",
                "value=x\ndef produce():\n nonlocal value\n value=value+0.5\n return [0.25]\ndef f(*values):\n return 0\nunused=f(*produce())\nreturn [value,y]"],
            "expansion-kwstar":[
                "def f(**values):\n return 0.5\ncoverage_bias=f(**{'v':0.25})",
                "def f(v=0.5):\n return v\ncoverage_bias=f(**{})",
                "def f(v,**values):\n return v\ncoverage_bias=f(0.5,**{'unused':0.25})",
                "value=x\ndef produce():\n nonlocal value\n value=value+0.5\n return {'v':0.25}\ndef f(**values):\n return 0\nunused=f(**produce())\nreturn [value,y]"],
        }
        for feature,bodies in cases.items():
            req=self.request("unified-public-"+feature)
            for body in bodies:
                if 'return [value,y]' not in body:body+='\nreturn [x+coverage_bias,y]'
                source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+line+'\n' for line in body.splitlines())
                with self.subTest(feature=feature,body=body),self.assertRaisesRegex(ValueError,"Missing contributing"):
                    validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)

    def test_expanded_values_evaluated_once_and_cipher_binding(self):
        cases={
            "expansion-star":"calls=0\ndef produce():\n nonlocal calls\n calls+=1\n return [x]\ndef f(*values):\n return values[0]\nvalue=f(*produce())\nif calls!=1:\n return [zero_ct,y]\nreturn [value,y]",
            "expansion-kwstar":"calls=0\ndef produce():\n nonlocal calls\n calls+=1\n return {'v':x}\ndef f(**values):\n return values['v']\nvalue=f(**produce())\nif calls!=1:\n return [zero_ct,y]\nreturn [value,y]",
            "signature-vararg":"def f(*values):\n return values[0]\nreturn [f(x),y]",
            "signature-kwarg":"def f(**values):\n return values['v']\nreturn [f(v=x),y]",
        }
        for feature,body in cases.items():
            req=self.request("unified-public-"+feature)
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+line+'\n' for line in body.splitlines())
            with self.subTest(feature=feature):
                checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)
                self.assertTrue(checked['construction_exercise']['witnesses'])

    def test_dead_unused_cancelled_and_unobserved_rejected(self):
        req=self.request("unified-public-call-strip")
        for body in (
            'unused = float(" 0.5 ".strip())\nreturn [x,y]',
            'if False:\n    bias = float(" 0.5 ".strip())\nreturn [x,y]',
            'bias = float(" 0.5 ".strip())\nreturn [x+bias-bias,y]',
            # This changes only the padded tail, beyond both named selectors.
            'bias = float(" 0.5 ".strip())\nreturn [x+bias*np.array([0,0,0,0,0,0,1,1]),y]',
        ):
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join('    '+s+'\n' for s in body.splitlines())
            with self.subTest(body=body),self.assertRaisesRegex(ValueError,"Missing contributing"):
                validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)
    def test_argument_side_effect_does_not_credit_unused_call_result(self):
        req=self.request("unified-public-call-float")
        source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n value=x\n def produce():\n  nonlocal value\n  value=value+1\n  return 0.5\n unused=float(produce())\n return [value,y]\n'
        with self.assertRaisesRegex(ValueError,"Missing contributing"):
            validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)

    def test_typed_partitions_reject_same_spelling_wrong_storage(self):
        cases={
            "cast.float.numeric.zero":"float(np.array([0.5]))",
            "cast.float.numeric.legacy":"float(np.array(0.5))",
            "cast.float.object.zero":"float(np.array(0.5))",
            "cast.float.object.legacy":"float(np.array([0.5]))",
            "cast.float.bool":"float(1)",
            "cast.int.base":"int('2')",
            "cast.int.numeric.negative_fraction":"int(-1.75)",
            "item.numeric.flat_negative":"np.array([0.25,0.5]).item(1)",
            "item.numeric.tuple":"np.array([0.5]).item((0,))",
            "item.object.cipher":"np.array([0.5],dtype=object).item()",
            "object.named_numeric_constructor":"float(np.array(np.array([0.5]),dtype=object))",
        }
        for feature,expression in cases.items():
            req=self.request("unified-public-"+feature.replace(".","-"))
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n bias='+expression+'\n return [x+bias,y]\n'
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,"Missing contributing"):
                validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)
    def test_mutating_call_argument_and_receiver_effects_do_not_credit_result(self):
        cases={
            "call.pop":'box={"v":0.5}\nunused=box.pop("v")\nreturn [x+len(box),y]',
            "call.setdefault":'box={}\nunused=box.setdefault("v",0.5)\nreturn [x+len(box),y]',
            "call.next":'it=iter([0.5,0.25])\nunused=next(it)\nreturn [x+list(it)[0],y]',
        }
        for feature,body in cases.items():
            req=self.request("unified-public-"+feature.replace(".","-"))
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join('    '+line+'\n' for line in body.splitlines())
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,"Missing contributing"):
                validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)
    def test_cipher_scalar_conversion_is_never_allowed(self):
        req=self.request("unified-public-cast-float-object-zero")
        for expression in ("float(x)","float(np.array(x,dtype=object))","int(np.array([x],dtype=object))"):
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n value='+expression+'\n return [x+value,y]\n'
            with self.subTest(expression=expression),self.assertRaises(ValueError):normalize(source,req)
    def test_zero_cipher_item_still_has_a_valid_influence_probe(self):
        b=Builder([(2,)]);out=b.node("subtract",["input0","input0"]);g=b.finish(out)
        name="unified-public-item-object-cipher";req=self.request(name,g)
        source=golden_variant(lower(req),name,req)
        checked=validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)
        self.assertEqual(checked["construction_exercise"]["numeric_features"],["item.object.cipher"])

    def test_periods_and_multiple_named_outputs(self):
        for p in (4,8,16,32,64,128,256):
            b=Builder([(p,)]);out=b.node("slice",["input0"],axis=0,start=0,stop=1,step=1);g=b.finish(out)
            req=self.request("unified-public-cast-float-numeric-zero",g)
            # No rule lowering required, including P=256 where a scalar-only registry suffices below.
            source='@hc.func("c,c")\ndef golden(x,zero_ct):\n b=float(np.array(0.5))\n return [x+b-0.5]\n'
            events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
            self.assertEqual(len(fingerprint(expanded,req)),3)
            checked=validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)
            self.assertEqual(checked["construction_exercise"]["slot_period"],p)
            for inputs in samples(g,4):
                expected=np.concatenate([v.reshape(-1) for v in evaluate(g,inputs).values()])
                np.testing.assert_allclose(probe(dict(req,public_constants=expanded["constants"]),expanded["source"],inputs),expected,
                                           atol=1e-12,rtol=1e-12)
    def test_state_control_and_accounting_false_credit_rejected(self):
        cases={
            "subscript.write":'a=[0]\nvalue=x\ndef produce():\n nonlocal value\n value=value+1\n return 0.5\na[0]=produce()\nreturn [value,y]',
            "slice.write":'a=[0]\na[:]=[0.5]\nreturn [x,y]',
            "call.update":'a={}\nvalue=x\ndef produce():\n nonlocal value\n value=value+1\n return {}\na.update(produce())\nreturn [value,y]',
            "call.clear":'a={}\na.clear()\nreturn [x+len(a),y]',
            "node.While":'i=0\nwhile i<2:\n i+=1\nreturn [x,y]',
            "node.Break":'for i in range(3):\n break\nreturn [x,y]',
            "node.Continue":'for i in range(3):\n continue\nreturn [x,y]',
            "event.for_else":'for i in range(1):\n pass\nelse:\n unused=0.5\nreturn [x,y]',
            "counter.default_evaluations":'def f(v=0.5):\n return v\nvalue=f(0.25)\nreturn [x+value,y]',
            "counter.nonlocal_writes":'unused=0\ndef f():\n nonlocal unused\n unused=0.5\nf()\nreturn [x,y]',
        }
        for feature,body in cases.items():
            req=self.request("unified-public-"+feature.replace(".","-"))
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join('    '+line+'\n' for line in body.splitlines())
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,"Missing contributing"):
                validate_candidate(dict(schema=1,request_id=req["request_id"],hecate_source=source),req)

    def test_counter_events_are_direct_scoped_and_optional(self):
        req=self.request("unified-public-counter-keyword_arguments")
        source=golden_variant(lower(req),req["construction_exercise"]["id"],req)
        events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
        counters=[e for e in events if "event.resource_counter" in e["features"]]
        self.assertTrue(counters)
        for e in counters:
            self.assertLess(e["facts"]["before"],e["facts"]["after"])
            self.assertGreater(e["span"][0],0)
        from function_construction import normalize as legacy
        from hecate_contract import UNIFIED_FLAT_CONTRACT
        raw=[]
        without=legacy(source,req["public_constants"],req["layout"]["output_ciphertexts"],
            input_names=("x","y","zero_ct"),object_unary=True,flat_contract=UNIFIED_FLAT_CONTRACT,
            slot_period=req["layout"]["input_slot_period"],observe=raw.append)
        self.assertFalse(any(type(x) is tuple and x[0]=="resource_counter" for x in raw))
        self.assertEqual(expanded["source"],without["source"])
        self.assertEqual(expanded["construction"],without["construction"])

    def test_request_and_profile_identity(self):
        name="unified-public-call-strip";req=self.request(name)
        validate_request(req)
        bad=copy.deepcopy(req);bad["construction_exercise"]["instruction"]="anything"
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(bad)
        with self.assertRaises(ValueError):prepare(model(),PROFILE_SHA256,construction=name)
        with self.assertRaises(ValueError):self.request("unified-view")

if __name__=="__main__":unittest.main()
