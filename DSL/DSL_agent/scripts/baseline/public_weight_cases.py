"""Small deterministic nonrepeating-weight regressions, plus a 10k-weight request."""
import random
from general_tensor_cases import Builder

def models():
    rng = random.Random(619)
    def matrix(rows,cols):
        return [[rng.uniform(-0.125,0.125) for _ in range(cols)] for _ in range(rows)]
    def bias(n):
        return [rng.uniform(-0.03125,0.03125) for _ in range(n)]
    def linear(b,x,rows,cols):
        return b.node("linear",[x,b.constant(matrix(rows,cols)),b.constant(bias(rows))])
    result=[]
    b=Builder("weights_linear",[[4]])
    result.append(b.finish(linear(b,"input0",3,4)))
    b=Builder("weights_mlp",[[4]])
    x=b.node("square",[linear(b,"input0",4,4)])
    result.append(b.finish(linear(b,x,2,4)))
    b=Builder("weights_residual",[[4]])
    x=b.node("add",["input0",linear(b,"input0",4,4)])
    result.append(b.finish(b.node("add",[x,linear(b,"input0",4,4)])))
    b=Builder("weights_channel_linear",[[1,1,2,2]])
    result.append(b.finish(linear(b,"input0",3,2)))
    b=Builder("weights_batched_linear",[[2,2]])
    result.append(b.finish(linear(b,"input0",3,2)))
    b=Builder("weights_multi_input",[[3],[3]])
    x=b.node("concat",["input0","input1"],axis=0)
    result.append(b.finish(linear(b,x,4,6)))
    b=Builder("weights_shared_outputs",[[4]])
    x=linear(b,"input0",3,4)
    result.append(b.finish(b.node("square",[x]),b.node("negate",[x])))
    b=Builder("weights_conv_pool",[[1,1,3,3]])
    kernel=[[[row for row in matrix(2,2)]] for _ in range(2)]
    x=b.node("conv2d",["input0",b.constant(kernel),b.constant(bias(2))],
             stride=[1,1],padding=[0,0],dilation=[1,1],groups=1)
    x=b.node("avg_pool2d",[x],kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True)
    result.append(b.finish(x))
    b=Builder("weights_linear_10k",[[100]])
    result.append(b.finish(linear(b,"input0",100,100)))
    return result

def period(model):
    return 128 if model["inputs"][0]["shape"]==[100] else 4
