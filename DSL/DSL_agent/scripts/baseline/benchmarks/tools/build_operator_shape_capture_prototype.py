"""Create an explicit staged Agent-only prototype; preserve production source bytes."""
import ast,difflib,hashlib,json
from pathlib import Path
from benchmark_graph import canonical,digest,require
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
VERSION="poseidon-operator-shape-capture-prototype-v1"
FUNCTIONAL="adaptive-pool-static-prototype-v1"
def build(output):
    output=Path(output);require(not output.exists(),"Fresh prototype")
    output.mkdir(parents=True)
    hashes={};originals={};patches={}
    for file in ("pytorch_static_trace.py","pytorch_model_capture.py","pytorch_model_intake.py"):
        path=ROOT/"scripts/baseline"/file;original=path.read_text();source=original
        def change(old,new,count=1):
            nonlocal source
            require(source.count(old)==count,"Exact prototype edit: "+old[:80])
            source=source.replace(old,new)
        if file=="pytorch_static_trace.py":
            change('ARITHMETIC_POLICY = "signature-static-arithmetic-shape-v1"',
                   'ARITHMETIC_POLICY = "signature-static-operator-shape-prototype-v1"')
            change('functions|={torch.subtract,torch.multiply}',
                   'functions|={torch.subtract,torch.multiply,F.instance_norm,\n'
                   '                            F.adaptive_avg_pool1d,F.adaptive_avg_pool2d}')
            insertion='''                if policy==ARITHMETIC_POLICY:
                    if type(child) in (torch.nn.InstanceNorm1d,torch.nn.InstanceNorm2d):
                        return x
                    if type(child) in (torch.nn.AdaptiveAvgPool1d,torch.nn.AdaptiveAvgPool2d):
                        fn=F.adaptive_avg_pool1d if type(child) is torch.nn.AdaptiveAvgPool1d else F.adaptive_avg_pool2d
                        return fn(x,child.output_size)
                    if type(child) in (torch.nn.ConstantPad1d,torch.nn.ConstantPad2d,torch.nn.ZeroPad2d,
                                      torch.nn.ReflectionPad1d,torch.nn.ReflectionPad2d,
                                      torch.nn.ReplicationPad1d,torch.nn.ReplicationPad2d):
                        mode=("reflect" if type(child) in (torch.nn.ReflectionPad1d,torch.nn.ReflectionPad2d)
                              else "replicate" if type(child) in (torch.nn.ReplicationPad1d,torch.nn.ReplicationPad2d)
                              else "constant")
                        return F.pad(x,child.padding,mode,getattr(child,"value",0.))
'''
            change('                if type(child) is torch.nn.Flatten:',insertion+'                if type(child) is torch.nn.Flatten:')
        elif file=="pytorch_model_capture.py":
            change('matrix_policy="rank-two-three-v1"):',
                   'matrix_policy="rank-two-three-v1", functional_policy="registered-v2"):')
            change('    from model_resources import resolve',
                   '    require(functional_policy in ("registered-v2","'+FUNCTIONAL+'"),"Functional capture policy")\n    from model_resources import resolve')
            insertion='''            elif functional_policy=="adaptive-pool-static-prototype-v1" and target in (
                    torch.nn.functional.adaptive_avg_pool1d,torch.nn.functional.adaptive_avg_pool2d):
                rank=1 if target is torch.nn.functional.adaptive_avg_pool1d else 2
                values=static_arguments(node,("input","output_size"),(None,None),"Functional adaptive average pool")
                require(isinstance(values["input"],torch.fx.Node),"Adaptive pool tensor input")
                require(len(shape(values["input"]))==rank+2,"Adaptive pool N,C,spatial rank")
                emit(node,"adaptive_avg_pool",[ref(values["input"])],
                     dict(output_size=pair(static_value(values["output_size"]),rank)))
'''
            change('            elif target is torch.nn.functional.batch_norm:',insertion+'            elif target is torch.nn.functional.batch_norm:')
        else:
            change('        matrix_policy="static-tensor-matmul-v1")',
                   '        matrix_policy="static-tensor-matmul-v1",\n'
                   '        functional_policy="'+FUNCTIONAL+'" if spec["format"]==SHAPE_FORMAT else "registered-v2")')
        ast.parse(source);target=output/file;target.write_text(source)
        hashes[file]=hashlib.sha256(target.read_bytes()).hexdigest()
        originals[file]=hashlib.sha256(original.encode()).hexdigest()
        patch="".join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile="a/scripts/baseline/"+file,tofile="b/scripts/baseline/"+file))
        patch_file=output/(file+".patch");patch_file.write_text(patch)
        patches[patch_file.name]=hashlib.sha256(patch.encode()).hexdigest()
    record=dict(format=VERSION,original_hashes=originals,staged_hashes=hashes,patch_hashes=patches,
        source_metadata_policy="signature-static-operator-shape-prototype-v1",
        functional_policy=FUNCTIONAL,production_integrated=False,backend_changes=0,reference_changes=0,
        scope="Known metadata rules and opt-in existing adaptive pool decomposition; staged Agent code only")
    record["binding"]=digest(record);(output/"prototype.json").write_bytes(canonical(record));return record
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.output)))
