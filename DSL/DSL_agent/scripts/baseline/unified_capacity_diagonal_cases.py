"""Cross-input/output chunk Linear contexts; separate from the frozen benchmark."""
from benchmark_graph import CAPACITY_FORMAT,validate

def models():
    result=[]
    for index,(shape,width) in enumerate((([128,4],4),([1,130,3],5),([1,2,64,4],6),([1,192,3],3),([1,2,128,4],4))):
        inputs=shape[-1]
        g=dict(format=CAPACITY_FORMAT,resource_profile="medium-v1",id="capacity_batched_linear_"+str(index),
          inputs=[dict(name="input",shape=shape)],
          constants=dict(offset=0.0625,weight=[[(i*inputs+j+1)/128. for j in range(inputs)] for i in range(width)],
                         bias=[(i-2)/256. for i in range(width)]),
          nodes=[dict(id="n0",op="add",inputs=["input","offset"],attrs={},outputs=["shifted"]),
                 dict(id="n1",op="linear",inputs=["shifted","weight","bias"],attrs={},outputs=["result"])],
          outputs=[dict(name="result",value="result")])
        validate(g);result.append(g)
    return result
