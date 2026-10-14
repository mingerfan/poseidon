"""Audit checked-in ResNet20 weights, independent ReLU references and stem admission.

Developer-only: no provider, compiler, bootstrap, parameter change or CUDA run.
The local 1000-image subset is not the complete CIFAR-10 test set or calibration data.
"""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def main():
    import numpy as np
    import torch
    from torch import nn
    from benchmark_graph import canonical,digest,require,validate
    from model_decomposition import decompose,FORMAT
    from model_resources import identity as resource_identity
    from model_weight_package import regular_bytes
    from semantic_benchmark_execution import runtime_sources
    from component_contract import prepare_task
    from unified_graph_lowering import candidate_source
    from hecate_python_env import WORK,VENV
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,
            "Pinned pure Python required")
    require(torch.__version__=="2.0.1" and np.__version__=="1.25.2"
            and torch.version.cuda is None and sys.version_info[:3]==(3,10,14),"Pinned ARM CPU runtime")
    require(not a.output.exists() and a.output.parent.resolve().is_relative_to(WORK/"results"),
            "Fresh audit results")
    a.output.mkdir();start=time.monotonic();frozen=runtime_sources();files={}
    torch.set_num_threads(1);torch.manual_seed(0)
    data=ROOT/"benchmark/resnet20_gpu/data/resnet20"
    weights=data/"pretrained_parameters/resnet20_new"
    own_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def read(path,count,maximum):
        raw=regular_bytes(path,maximum)
        require(len(raw.split())==count,"Exact numeric element count: "+str(path))
        values=np.fromstring(raw.decode("ascii"),dtype=np.float64,sep=" ")
        require(values.size==count and np.isfinite(values).all(),"Finite weight/data values")
        files[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(raw).hexdigest(),
                                               bytes=len(raw),elements=count)
        return values
    bank={}
    def conv_bn(prefix,cin,cout):
        cname="conv1_weight.txt" if prefix=="stem" else prefix+"_weight.txt"
        bn="bn1" if prefix=="stem" else prefix.replace("_conv","_bn")
        values=dict(weight=read(weights/cname,cout*cin*9,1024**2).reshape(cout,cin,3,3))
        for role in ("bias","running_mean","running_var","weight"):
            values["bn_"+role]=read(weights/(bn+"_"+role+".txt"),cout,16384)
        require((values["bn_running_var"]>=0).all(),"Nonnegative frozen BN variance")
        bank[prefix]=values
        return values
    conv_bn("stem",3,16)
    blocks=[]
    previous=16
    for stage,cout in enumerate((16,32,64),1):
        for index in range(3):
            transition=stage>1 and index==0;stride=2 if transition else 1
            prefix="layer"+str(stage)+"_"+str(index)
            conv_bn(prefix+"_conv1",previous,cout);conv_bn(prefix+"_conv2",cout,cout)
            blocks.append(dict(prefix=prefix,input_channels=previous,output_channels=cout,
                               stride=stride,option_a=transition))
            previous=cout
    fc=read(weights/"linear_weight.txt",640,65536).reshape(10,64)
    bias=read(weights/"linear_bias.txt",10,16384)
    labels=read(data/"testFile/test_label.txt",1000,16384)
    require(np.array_equal(labels,labels.astype(np.int64)) and ((labels>=0)&(labels<10)).all(),
            "CIFAR class labels")
    images=read(data/"testFile/test_values.txt",1000*3072,64*1024**2).reshape(1000,3,32,32)
    source_files=["benchmark/resnet20_gpu/resnet20_weights.cpp",
                  "benchmark/resnet20_gpu/resnet20_topology.cpp",
                  "benchmark/resnet20_gpu/gpu_resnet20_inference.cpp"]
    for name in source_files:
        raw=regular_bytes(ROOT/name,131072)
        files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    class FrozenConvBN(nn.Module):
        def __init__(self,spec,stride):
            super().__init__();cout,cin=spec["weight"].shape[:2]
            self.conv=nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False).double()
            self.bn=nn.BatchNorm2d(cout,eps=1e-5).double()
            with torch.no_grad():
                self.conv.weight.copy_(torch.from_numpy(spec["weight"]))
                self.bn.weight.copy_(torch.from_numpy(spec["bn_weight"]))
                self.bn.bias.copy_(torch.from_numpy(spec["bn_bias"]))
                self.bn.running_mean.copy_(torch.from_numpy(spec["bn_running_mean"]))
                self.bn.running_var.copy_(torch.from_numpy(spec["bn_running_var"]))
        def forward(self,x):return self.bn(self.conv(x))
    class FrozenResNet20(nn.Module):
        def __init__(self):
            super().__init__();self.stem=FrozenConvBN(bank["stem"],1)
            self.branch1=nn.ModuleList([FrozenConvBN(bank[b["prefix"]+"_conv1"],b["stride"]) for b in blocks])
            self.branch2=nn.ModuleList([FrozenConvBN(bank[b["prefix"]+"_conv2"],1) for b in blocks])
            self.fc=nn.Linear(64,10).double()
            with torch.no_grad():
                self.fc.weight.copy_(torch.from_numpy(fc));self.fc.bias.copy_(torch.from_numpy(bias))
        def forward(self,x):
            x=torch.relu(self.stem(x))
            for spec,first,second in zip(blocks,self.branch1,self.branch2):
                shortcut=x
                if spec["option_a"]:
                    shortcut=shortcut[:,:,::2,::2]
                    shortcut=torch.cat([torch.zeros_like(shortcut[:,:spec["input_channels"]//2]),
                                        shortcut,torch.zeros_like(shortcut[:,:spec["input_channels"]//2])],dim=1)
                x=torch.relu(second(torch.relu(first(x)))+shortcut)
            return self.fc(x.mean(dim=(-2,-1)))
    model=FrozenResNet20().eval()
    def numpy_conv_bn(x,spec,stride):
        padded=np.pad(x,((0,0),(0,0),(1,1),(1,1)))
        windows=np.lib.stride_tricks.sliding_window_view(padded,(3,3),axis=(-2,-1))
        windows=windows[:,:,::stride,::stride,:,:]
        value=np.einsum("nchwkl,ockl->nohw",windows,spec["weight"],optimize=True)
        scale=spec["bn_weight"]/np.sqrt(spec["bn_running_var"]+1e-5)
        offset=spec["bn_bias"]-spec["bn_running_mean"]*scale
        return value*scale[None,:,None,None]+offset[None,:,None,None]
    def numpy_model(x):
        x=np.maximum(numpy_conv_bn(x,bank["stem"],1),0)
        for spec in blocks:
            shortcut=x
            if spec["option_a"]:
                shortcut=x[:,:,::2,::2];pad=spec["input_channels"]//2
                shortcut=np.pad(shortcut,((0,0),(pad,pad),(0,0),(0,0)))
            y=np.maximum(numpy_conv_bn(x,bank[spec["prefix"]+"_conv1"],spec["stride"]),0)
            x=np.maximum(numpy_conv_bn(y,bank[spec["prefix"]+"_conv2"],1)+shortcut,0)
        return x.mean(axis=(-2,-1))@fc.T+bias
    indices=[int(i) for label in range(10) for i in np.flatnonzero(labels==label)[:4]]
    require(len(indices)==40,"Four available images per class")
    selected=images[indices]
    lo=images.min(axis=(0,2,3),keepdims=True);hi=images.max(axis=(0,2,3),keepdims=True)
    # Observed-range probes are diagnostics, pending original preprocessing provenance.
    probes=np.concatenate([np.zeros((1,3,32,32)),np.broadcast_to((lo+hi)/2,(1,3,32,32)),
                           np.random.default_rng(20261002).uniform(lo,hi,size=(1,3,32,32)),
                           np.where(np.indices((1,3,32,32)).sum(axis=0)%2,lo,hi)])
    reference_inputs=np.concatenate([selected,probes])
    with torch.no_grad():
        original=model(torch.from_numpy(reference_inputs)).numpy()
        predictions=[]
        for offset in range(0,1000,16):
            predictions.extend(model(torch.from_numpy(images[offset:offset+16])).argmax(dim=1).tolist())
    independent=numpy_model(reference_inputs)
    require(original.shape==independent.shape==(44,10),"ResNet reference output binding")
    ratio=float(np.max(np.abs(original-independent)/(1e-12+1e-12*np.abs(independent))))
    np.testing.assert_allclose(original,independent,atol=1e-12,rtol=1e-12)
    np.save(a.output/"reference-inputs.npy",reference_inputs,allow_pickle=False)
    np.save(a.output/"standard-relu-logits.npy",original,allow_pickle=False)
    np.save(a.output/"independent-logits.npy",independent,allow_pickle=False)
    spec=bank["stem"]
    stem=dict(format=FORMAT,id="resnet20_frozen_stem_pre_relu",inputs=[dict(name="image",shape=[1,3,32,32])],
              constants={"weight":spec["weight"].tolist(),"mean":spec["bn_running_mean"].tolist(),
                         "variance":spec["bn_running_var"].tolist(),"gain":spec["bn_weight"].tolist(),
                         "bias":spec["bn_bias"].tolist()},
              nodes=[dict(id="stem_conv",op="conv2d",inputs=["image","weight"],
                          attrs=dict(stride=[1,1],padding=[1,1],dilation=[1,1],groups=1),outputs=["convolved"]),
                     dict(id="stem_bn",op="batch_norm",inputs=["convolved","mean","variance","gain","bias"],
                          attrs=dict(eps=1e-5),outputs=["normalized"]),
                     dict(id="diagnostic_mean",op="mean",inputs=["normalized"],
                          attrs=dict(axes=[2,3],keepdims=False),outputs=["channel_means"])],
              outputs=[dict(name="channel_means",value="channel_means")])
    lowered,binding=decompose(stem,resource_profile="resnet20-v1")
    (a.output/"stem-original.json").write_bytes(canonical(stem))
    (a.output/"stem-model.json").write_bytes(canonical(lowered))
    (a.output/"stem-decomposition.json").write_bytes(canonical(binding))
    checked=validate(lowered)
    norm_node=next(n for n in lowered["nodes"] if binding["source_map"][n["id"]]=="stem_bn")
    norm_shape=checked["shapes"][norm_node["outputs"][0]]
    admission=dict(model_sha256=digest(lowered),logical_shape_check="passed",
                   intermediate_shape=norm_shape,
                   intermediate_chunks=math.ceil(math.prod(norm_shape)/256),
                   scope="authentic full-size pre-ReLU stem; channel means are diagnostic outputs")
    try:
        req=prepare_task(lowered,dict(chunk_period=256,generation_guidance="explicit-v10"))
        admission.update(request_preparation="passed",request_id=req["request_id"],
                         input_ciphertexts=len(req["layout"]["inputs"]),output_ciphertexts=req["layout"]["output_ciphertexts"])
        (a.output/"stem-request.json").write_bytes(__import__("candidate_contract").request_file_bytes(req))
        candidate,strategy=candidate_source(req)
        (a.output/"stem-rule-candidate.py").write_text(candidate)
        admission.update(rule_construction="passed",strategy=strategy,source_bytes=len(candidate.encode()))
    except (ValueError,RuntimeError) as error:
        admission.update(request_preparation="blocked",diagnostic=str(error))
    require(runtime_sources()==frozen,"Runtime/source drift")
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==own_sha,"Developer tool drift")
    for name,entry in files.items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==entry["sha256"],"Data/source drift")
    report=dict(format="poseidon-resnet20-source-audit-v1",source_hashes=frozen,files=files,
        developer_sha256=own_sha,resource_profile=resource_identity("resnet20-v1"),
        topology=dict(input=[1,3,32,32],channels=[16,32,64],residual_blocks=9,convolutions=19,output=[1,10],
                      shortcut="Option A: even pixels and symmetric channel padding"),
        weight_elements=sum(entry.get("elements",0) for name,entry in files.items() if "pretrained_parameters" in name),
        weight_layout="OIHW convolutions; output-by-input Linear, checked against C++ index expressions",
        reference=dict(status="passed",function="standard ReLU ResNet20, CPU float64, unfused eval BatchNorm eps=1e-5",
                       input_preprocessing="checked-in CHW values without the GPU representation's divide-by-40",
                       images=40,per_class=4,indices=indices,observed_range_probes=4,values=440,
                       max_absolute_error=float(np.max(np.abs(original-independent))),max_tolerance_ratio=ratio,
                       tolerance=dict(abs=1e-12,rel=1e-12)),
        local_subset=dict(images=1000,class_counts=[int((labels==label).sum()) for label in range(10)],
                          correct=int((np.array(predictions)==labels).sum()),top1=float((np.array(predictions)==labels).mean()),
                          full_cifar10_test=False,training_calibration=False,
                          preprocessing_provenance_complete=False),
        stem=admission,compiled=False,encrypted_execution=False,agent_generated=False,paid_calls=0,
        approximation_calibrated=False,bootstrap_certified=False,seconds=time.monotonic()-start)
    report["binding"]=digest(report)
    (a.output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k not in ("files","source_hashes")}),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
