"""Read-only deterministic MHA overlay builder; writes only its own fresh draft."""
import ast, copy, difflib, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
DRAFT=BASE/"benchmarks/tools/static_mha_source_v16_draft_v1"
OLD=BASE/"benchmarks/tools/static_mha_source_draft_v1"
TARGETS=("pytorch_model_intake.py","pytorch_ordered_contract.py",
         "pytorch_ordered_intake.py","pytorch_source_contract.py")
FROZEN={
 "source_plan.py":"4868881693de491e9ceab0400fde80076fe31dd297552bc5ab6741942e3251e6",
 "output_contract.py":"7193cec06b6da2ce0b697d943cfaeb4173341eb810fed709d5ff67f5bf6f675d",
 "mha_source_plan.py":"f1f845815f671091a71b9b86217c5b68644d704b736ed7a237f638dec6164f49",
 "mha_source_reference.py":"25bdaf114101b57fe3b3f8b78f6fd2eb4e77e97f0e357851aef4e86b849817c5"}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def replace(text,old,new,count=1):
 require(text.count(old)==count,"Patch context must match exact expected count")
 return text.replace(old,new)
def source(path):
 raw=path.read_bytes();require(not path.is_symlink(),"Regular fixed source")
 return raw.decode(),sha(raw)
def build_sources():
 originals={};outputs={};frozen={}
 for name,expected in FROZEN.items():
  path=(OLD if name.startswith("mha_source_") else DRAFT)/name
  text,actual=source(path);require(actual==expected,"Frozen math/planning identity")
  frozen[str(path.relative_to(ROOT))]=actual
 for name in TARGETS:
  originals[name],_=source(BASE/name)
 outputs.update(originals)
 newmath,_=source(OLD/"mha_source_plan.py")
 newmath=replace(newmath,"base=Path(__file__).parents[3]","base=Path(__file__).parent")
 outputs["pytorch_mha_math.py"]=newmath
 outputs["pytorch_mha_reference_math.py"]=source(OLD/"mha_source_reference.py")[0]
 outputs["pytorch_mha_outputs.py"]=source(DRAFT/"output_contract.py")[0]
 p=source(DRAFT/"source_plan.py")[0]
 begin=p.index("HERE = ");end=p.index("NAME = ",begin)
 p=p[:begin]+'''HERE = Path(__file__).resolve().parent
BASE = HERE
ROOT = HERE
OLD = HERE
FORMAT = "poseidon-static-mha-source-v16-overlay-v1"
REQUEST_FORMAT = "poseidon-static-mha-source-request-v16-overlay-v1"
OWNED = ("pytorch_mha_plan.py", "pytorch_mha_math.py", "pytorch_mha_reference_math.py",
         "pytorch_mha_outputs.py", "pytorch_mha_capture.py")
LEGACY_SHA = '''+repr({"pytorch_mha_math.py":sha(newmath.encode()),
 "pytorch_mha_reference_math.py":sha(outputs["pytorch_mha_reference_math.py"].encode())})+"\n"+p[end:]
 p=replace(p,"from output_contract import ","from pytorch_mha_outputs import ")
 p=replace(p,'result[str(p.relative_to(ROOT))]','result[p.name]')
 p=replace(p,'p.name for p in paths','p.name for p in paths',count=0) if False else p
 p=p.replace('"mha_source_plan.py"','"pytorch_mha_math.py"').replace('"mha_source_reference.py"','"pytorch_mha_reference_math.py"')
 start=p.index("def check_approximation(");end=p.index("\ndef _shape(",start)
 p=p[:start]+'''def check_approximation(value):
    from pytorch_approximation import validate
    validate(value)

'''+p[end+1:]
 p=replace(p,'\"options\", \"mask_sources\"}', '\"options\", \"mask_sources\", \"module_paths\", \"approximation\"}')
 p=replace(p,'    used = set(values); entries=[]; call_ids=set()',
 '''    from pytorch_approximation import Resolver
    selection=Resolver(request["approximation"])
    used = set(values); entries=[]; call_ids=set()''')
 p=replace(p,'        mini = legacy.plan(call["module"],call["operands"],call["options"],request["approximation"],',
 '''        chosen={name:selection.select(call["id"],name,name,call["module_paths"])
                for name in ("exp","reciprocal")}
        require(chosen==call["approximation"],"MHA approximation/source selector binding")
        mini = legacy.plan(call["module"],call["operands"],call["options"],chosen,''')
 p=replace(p,'    result_value = _decode(request["output_values"], descriptors)',
             '    selection.finish()\n    result_value = _decode(request["output_values"], descriptors)')
 p=replace(p,'    legacy = _load_legacy("pytorch_mha_math.py")\n    source = copy.deepcopy(module)',
 '''    legacy = _load_legacy("pytorch_mha_math.py")
    from pytorch_approximation import Resolver,module_paths
    selection=Resolver(approximation)
    source = copy.deepcopy(module)''')
 p=replace(p,'                      operands=operands,options=options,mask_sources=mask_sources)',
 '''                      operands=operands,options=options,mask_sources=mask_sources,
                      module_paths=module_paths(node),
                      approximation={name:selection.select(node.name,name,name,module_paths(node))
                                     for name in ("exp","reciprocal")})''')
 p=replace(p,'            legacy.plan(call["module"],operands,options,approximation)',
             '            legacy.plan(call["module"],operands,options,call["approximation"])')
 p=replace(p,'    plan=build_request(request)','    selection.finish()\n    plan=build_request(request)')
 p=replace(p,'approximation=request["approximation"],resource_profile=request["resource_profile"])',
             'approximation=call["approximation"],resource_profile=request["resource_profile"])')
 outputs["pytorch_mha_plan.py"]=p
 outputs["pytorch_mha_capture.py"]=source(HERE/"pytorch_mha_capture.py")[0]
 p=outputs["pytorch_model_intake.py"]
 p=replace(p,'GELU_FORMAT="poseidon-trusted-pytorch-v15"',
 '''MHA_FORMAT="poseidon-trusted-pytorch-v16"
MHA_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v16"
MHA_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v16"
GELU_FORMAT="poseidon-trusted-pytorch-v15"''')
 p=p.replace('ORDERED_FUSED_FORMAT,GELU_FORMAT)','ORDERED_FUSED_FORMAT,GELU_FORMAT,MHA_FORMAT)')
 p=p.replace('GELU_FORMAT:GELU_PACKAGE_FORMAT}','GELU_FORMAT:GELU_PACKAGE_FORMAT,MHA_FORMAT:MHA_PACKAGE_FORMAT}')
 p=p.replace('GELU_FORMAT:GELU_REFERENCE_FORMAT}','GELU_FORMAT:GELU_REFERENCE_FORMAT,MHA_FORMAT:MHA_REFERENCE_FORMAT}')
 p=p.replace('ORDERED_FUSED_PACKAGE_FORMAT,GELU_PACKAGE_FORMAT):','ORDERED_FUSED_PACKAGE_FORMAT,GELU_PACKAGE_FORMAT,MHA_PACKAGE_FORMAT):')
 outputs["pytorch_model_intake.py"]=p
 p=outputs["pytorch_ordered_contract.py"]
 p=replace(p,'SOURCE_FORMATS = (FORMAT, COMMON_SOURCE_FORMAT, FUSED_SOURCE_FORMAT, GELU_SOURCE_FORMAT)',
 '''MHA_SOURCE_FORMAT = "poseidon-trusted-pytorch-v16"
MHA_PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v16"
MHA_REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v16"
SOURCE_FORMATS = (FORMAT, COMMON_SOURCE_FORMAT, FUSED_SOURCE_FORMAT, GELU_SOURCE_FORMAT, MHA_SOURCE_FORMAT)''')
 p=p.replace('GELU_SOURCE_FORMAT:GELU_PACKAGE_FORMAT}[spec["format"]]','GELU_SOURCE_FORMAT:GELU_PACKAGE_FORMAT,MHA_SOURCE_FORMAT:MHA_PACKAGE_FORMAT}[spec["format"]]')
 p=p.replace('GELU_SOURCE_FORMAT:GELU_REFERENCE_FORMAT}[spec["format"]]','GELU_SOURCE_FORMAT:GELU_REFERENCE_FORMAT,MHA_SOURCE_FORMAT:MHA_REFERENCE_FORMAT}[spec["format"]]')
 p=replace(p,'    "strict_json_files.py",',
 '''    "strict_json_files.py", "pytorch_mha_plan.py", "pytorch_mha_math.py",
    "pytorch_mha_reference_math.py", "pytorch_mha_outputs.py", "pytorch_mha_capture.py",''')
 p=replace(p,'                           expected_weight_binding):\n    import numpy as np',
 '''                           expected_weight_binding):
    if spec.get("format")==MHA_SOURCE_FORMAT:
        from pytorch_mha_capture import check_capture_metadata as selected
        return selected(metadata,original,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding)
    import numpy as np''')
 p=replace(p,'def check_reference(report, original, lowered, spec, *, resource_profile, expected_weight_binding):\n',
 '''def check_reference(report, original, lowered, spec, *, resource_profile, expected_weight_binding):
    if spec.get("format")==MHA_SOURCE_FORMAT:
        from pytorch_mha_capture import check_reference as selected
        return selected(report,original,lowered,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding)
''')
 outputs["pytorch_ordered_contract.py"]=p
 p=outputs["pytorch_ordered_intake.py"]
 p=replace(p,'GELU_SOURCE_FORMAT,\n','GELU_SOURCE_FORMAT, MHA_SOURCE_FORMAT,\n')
 start=p.index('    prepared = prepare_module(');end=p.index('    high, lowered, captured, context = prepared',start)
 old=p[start:end]
 p=p[:start]+'''    if spec["format"]==MHA_SOURCE_FORMAT:
        # Temporary MHA-only capture prototype; final integration must use
        # registered per-node expansion inside the unified ordered graph.
        from pytorch_mha_capture import prepare_module as mha_prepare,validate
        prepared=mha_prepare(module,spec["inputs"],spec["outputs"],approximations=spec["approximations"],
            resource_profile=profile,dtype_policy=spec["dtype_policy"])
    else:
'''+''.join("    "+line+"\n" for line in old.rstrip("\n").splitlines())+p[end:]
 p=replace(p,'    alias = metadata["ordered_alias_capture"]\n    adapter = alias["capture_adapter"]',
 '''    if spec["format"]==MHA_SOURCE_FORMAT:
        q=metadata["mha_source_capture"]["source_request"]
        adapter=dict(source_fx_text=q["source_fx_text"],normalized_fx_text=q["source_fx_text"])
    else:
        alias = metadata["ordered_alias_capture"]
        adapter = alias["capture_adapter"]''')
 outputs["pytorch_ordered_intake.py"]=p
 p=outputs["pytorch_source_contract.py"]
 p=replace(p,'def validate_output_tree(tree,names,*,allow_scalars=False):\n',
 '''def validate_output_tree(tree,names,*,allow_scalars=False,source_format=None):
    if source_format=="poseidon-trusted-pytorch-v16":
        from pytorch_mha_outputs import validate
        validate(tree)
        require(tree["source_format"]==source_format and tree["tensor_names"]==names,
                "Versioned MHA output signature")
        return tree
''')
 p=replace(p,'def restore_outputs(values,tree):\n',
 '''def restore_outputs(values,tree):
    if type(tree) is dict and tree.get("format")=="poseidon-static-output-tree-v16-draft-v1":
        from pytorch_mha_outputs import restore,SOURCE_FORMAT
        require(tree.get("source_format")==SOURCE_FORMAT,"Explicit MHA output restoration version")
        return restore(values,tree)
''')
 outputs["pytorch_source_contract.py"]=p
 for name,text in outputs.items():
  ast.parse(text,filename=name);compile(text,name,"exec")
 return originals,outputs,frozen

def build():
 originals,outputs,frozen=build_sources()
 dest=HERE/"candidate";require(not dest.exists(),"Fresh overlay candidate only")
 dest.mkdir()
 for name,text in outputs.items():(dest/name).write_text(text)
 snapshot=HERE/"baseline-snapshot";require(not snapshot.exists(),"Fresh baseline snapshot only")
 snapshot.mkdir()
 for name,text in originals.items():(snapshot/name).write_text(text)
 patch="".join("".join(difflib.unified_diff(
     originals.get(name,"").splitlines(True),text.splitlines(True),
     fromfile="a/scripts/baseline/"+name if name in originals else "/dev/null",
     tofile="b/scripts/baseline/"+name)) for name,text in outputs.items())
 (HERE/"integration.patch").write_text(patch)
 manifest=dict(format="poseidon-staged-mha-v16-integration-overlay-v1",
    targets={name:dict(before_sha256=sha(originals[name].encode()) if name in originals else None,
        after_sha256=sha(text.encode())) for name,text in outputs.items()},
    frozen_inputs=frozen,patch_sha256=sha(patch.encode()),
    production_applied=False,production_capture_validated=False,unified_ordered_integration_complete=False,
    api=0,nix_entries=0,compiler=0,encrypted_execution=0,
    scope="Review-only native-MHA capture boundary prototype; final architecture requires unified ordered per-node expansion")
 manifest["binding"]=sha(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode())
 (HERE/"overlay.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print(json.dumps(dict(targets=len(outputs),changed_existing=len(originals),binding=manifest["binding"],
                       patch_sha256=manifest["patch_sha256"],production_applied=False)))
if __name__=="__main__":build()
