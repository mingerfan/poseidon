"""Small independent regression graphs for the explicit rank 1..32 profile."""
import copy
from benchmark_graph import CAPACITY_FORMAT, infer, shape, validate

PROFILE = "general-tensors-v1"


class Builder:
    def __init__(self, name, shapes):
        self.graph = dict(format=CAPACITY_FORMAT, id=name, resource_profile=PROFILE,
                          inputs=[dict(name="input"+str(i), shape=list(s)) for i,s in enumerate(shapes)],
                          constants={}, nodes=[], outputs=[])
        self.shapes = {s["name"]:tuple(s["shape"]) for s in self.graph["inputs"]}

    def constant(self, value):
        name = "c"+str(len(self.graph["constants"]))
        self.graph["constants"][name] = value
        self.shapes[name] = shape(value, resource_profile=PROFILE)
        return name

    def node(self, op, inputs, **attrs):
        name = "n"+str(len(self.graph["nodes"]))
        inferred = infer(op, [self.shapes[x] for x in inputs], attrs, resource_profile=PROFILE)
        outputs = [name+"v"+str(i) for i in range(len(inferred))]
        self.graph["nodes"].append(dict(id=name, op=op, inputs=list(inputs), attrs=attrs, outputs=outputs))
        self.shapes.update(zip(outputs,inferred))
        return outputs[0] if len(outputs)==1 else outputs

    def finish(self, *values):
        self.graph["outputs"] = [dict(name="output"+str(i), value=x) for i,x in enumerate(values)]
        validate(self.graph)
        return copy.deepcopy(self.graph)


def tensor_shape(rank):
    return [4] if rank==1 else [1]*(rank-2)+[2,2]


def models():
    cases = []
    b=Builder("general_rank1_affine",[tensor_shape(1)])
    x=b.node("multiply",["input0",b.constant(0.125)])
    cases.append(b.finish(b.node("add",[x,b.constant(-0.03125)])))
    b=Builder("general_rank4_linear",[tensor_shape(4)])
    x=b.node("linear",["input0",b.constant([[0.125,-0.0625],[-0.03125,0.09375]]),
                       b.constant([0.03125,-0.015625])])
    cases.append(b.finish(x))
    b=Builder("general_rank5_transpose",[tensor_shape(5)])
    x=b.node("transpose",["input0"],dim0=-2,dim1=-1)
    cases.append(b.finish(b.node("negate",[x])))
    b=Builder("general_rank8_partial_mean",[tensor_shape(8)])
    x=b.node("square",["input0"])
    cases.append(b.finish(b.node("mean",[x],axes=[-1],keepdims=True)))
    b=Builder("general_rank16_multi_join",[tensor_shape(16)]*2)
    a=b.node("add",["input0","input1"])
    c=b.node("subtract",["input0","input1"])
    x=b.node("stack",[a,c],axis=-1)
    cases.append(b.finish(x,c))
    b=Builder("general_rank32_reshape",[tensor_shape(32)])
    x=b.node("permute",["input0"],dims=[*range(30),31,30])
    x=b.node("reshape",[x],shape=[1]*31+[4])
    cases.append(b.finish(b.node("add",[x,b.constant(0.0625)])))
    b=Builder("general_rank6_transpose_tail",[[2,1,1,1,1,3]])
    x=b.node("transpose",["input0"],dim0=0,dim1=-1)
    cases.append(b.finish(b.node("add",[x,b.constant(0.125)])))
    b=Builder("general_rank32_transpose_tail",[[1]*29+[2,1,3]])
    x=b.node("square",["input0"])
    cases.append(b.finish(b.node("transpose",[x],dim0=-3,dim1=-1)))
    b=Builder("general_rank8_permute_tail",[[2,1,1,1,1,1,1,3]])
    x=b.node("permute",["input0"],dims=[7,0,1,2,3,4,5,6])
    cases.append(b.finish(b.node("square",[x])))
    b=Builder("general_rank16_permute_tail",[[1]*13+[2,1,3]])
    x=b.node("permute",["input0"],dims=[*range(13),15,14,13])
    cases.append(b.finish(b.node("negate",[x])))
    return cases
