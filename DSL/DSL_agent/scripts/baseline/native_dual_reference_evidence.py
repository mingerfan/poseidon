"""Keep native-core and semantic references separate and verify both.

No reference, output, source or tolerance is rewritten. Binding errors reject
the evidence; numerical differences are returned as failed numerical checks.
This profile reads the existing recorded real-slot ABI only.
"""
from pathlib import Path
from delivery_component import bound,canonical,require,sha
from first_output_backend import _json
from first_output_numerical_evidence import verify_backend_witness
from first_output_assessment import compare,MATH,CKKS

VERSION='poseidon-real-native-dual-reference-v1'


def verify_native_rows(witness,request,source):
    require(MATH==dict(atol=1e-12,rtol=1e-12)and CKKS==dict(atol=1e-5,rtol=1e-4),'Original numerical bounds required')
    require(type(source)is bytes and witness.get('source_sha256')==witness.get('compiler_source_sha256')==sha(source),'Original source required')
    proof=verify_backend_witness(witness,request,source)
    require(proof['real_compilation_observed']and proof['real_encrypted_execution_observed']and witness.get('test_only')is False,
        'Actual compiler and native output evidence required')
    root=Path(proof['evidence']);report=_json(root/'report.json');execution=_json(root/'attempt-00/output/execution.json')
    inputs=_json(root/'attempt-00/output/inputs.json');comparison=report['attempts'][0].get('comparison',{})
    outputs=execution.get('outputs');references=comparison.get('reference');actual=comparison.get('actual')
    require(all(type(v)is list for v in [inputs,outputs,references,actual])and bool(inputs)
        and len(inputs)==len(outputs)==len(references)==len(actual),'Native input/output/reference cells missing')
    selectors=request.get('layout',{}).get('output_selectors')
    require(type(selectors)is list and bool(selectors)and all(type(s)is list and len(s)==2
        and all(type(i)is int and i>=0 for i in s)for s in selectors),'Explicit audited real-output selectors required')
    native={}
    for sample,output,expected,observed in zip(inputs,outputs,references,actual):
        key=sha(canonical(sample));require(key not in native,'Duplicate original native input')
        require(type(output)is list and all(c<len(output)and type(output[c])is list and s<len(output[c])for c,s in selectors),'Native output selector range')
        decoded=[output[c][s]for c,s in selectors]
        require(decoded==observed,'Recorded actual comparison differs from native decode')
        native[key]=dict(actual=decoded,reference=expected)
    rows=witness.get('inputs');require(type(rows)is list and bool(rows),'Numerical input rows required');seen=set();checks=[]
    for row in rows:
        key=row['input_sha256'];require(key in native and key not in seen,'Reported input must be a unique original native input');seen.add(key)
        require(row.get('source_sha256')==sha(source),'Input row used another source')
        require(row['CKKS_real']==native[key]['actual'],'Reported CKKS cells differ from original decoded output')
        require(row.get('native_core_reference')==native[key]['reference'],'Native core reference changed')
        require(row.get('CKKS_imag')is None,'This recorded real-slot ABI cannot claim imaginary cells')
        alignment=compare(row['declared_reference'],row['native_core_reference'],**MATH)
        core=compare(row['native_core_reference'],row['CKKS_real'],**CKKS)
        semantic=compare(row['declared_reference'],row['CKKS_real'],**CKKS)
        checks.append(dict(input_sha256=key,native_core_reference_sha256=sha(canonical(row['native_core_reference'])),
            declared_reference_sha256=sha(canonical(row['declared_reference'])),reference_alignment=alignment,CKKS_against_core=core,
            CKKS_against_declared=semantic,passed=all(v['failed_values']==0 for v in [alignment,core,semantic])))
    require(seen==set(native),'All original native input groups must be reported')
    require(type(witness.get('intermediate_decryptions'))is int and witness['intermediate_decryptions']==0
        and witness.get('reference_or_tolerance_changed')is False,'Original execution and reference boundary')
    return bound(dict(format=VERSION,request_id=request['request_id'],source_sha256=sha(source),backend_binding_sha256=sha(canonical(proof)),
        input_groups=len(checks),compared_values=sum(r['CKKS_against_core']['compared_values']for r in checks),inputs=checks,
        independent_math=dict(MATH),CKKS=dict(CKKS),passed=all(r['passed']for r in checks),
        native_and_declared_references_preserved=True,independent_DSL_mathematics_verified_here=False,source_or_reference_rewritten=False))
