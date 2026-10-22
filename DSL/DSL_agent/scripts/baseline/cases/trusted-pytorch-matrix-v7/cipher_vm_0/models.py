"""Static matrix models for the explicit trusted PyTorch v7 interface."""
import torch

PARTITIONS = {
 "cipher_vv": [([3],[3]),([5],[5]),([7],[7])],
 "cipher_mv": [([2,3],[3]),([2,1,4],[4]),([2,1,2,2],[2])],
 "cipher_vm": [([3],[3,2]),([4],[2,4,1]),([2],[2,1,2,3])],
 "cipher_batch": [([2,1,2,3],[1,3,3,2]),([1,2,1,2,2],[2,1,3,2,1]),([1]*30+[2,2],[2,2])],
 "public_vector_l": [([3],[3,2]),([2],[2,2,3]),([4],[2,1,4,2])],
 "public_vector_r": [([2,3],[3]),([2,1,2],[2]),([2,1,1,4],[4])],
 "public_matrix_l": [([2,3],[3]),([2,2],[2,1,2,3]),([3,2],[2,1])],
 "public_matrix_r": [([3],[3,2]),([2,1,2,3],[3,2]),([1]*30+[1,2],[2,3])],
 "public_batch_l": [([2,2,3],[3,2]),([2,1,1,2],[1,2,2,3]),([1,2,1,2,2],[2,1,2,1])],
 "public_batch_r": [([2,3],[2,3,2]),([2,1,2,3],[1,2,3,1]),([2],[2,1,2,3])]
}
def product(x,y,spelling):
    if spelling==0:return x@y
    if spelling==1:return torch.matmul(input=x,other=y,out=None)
    return x.matmul(y)

def output(value,structure):
    if structure=="dict":return {"value":value,"square":value.square()*.125+value}
    if structure=="tuple":return value,value.sum()
    return value

class Cipher(torch.nn.Module):
    def __init__(self,spelling,structure):
        super().__init__();self.spelling=spelling;self.structure=structure
    def forward(self,x,y):return output(product(x,y,self.spelling),self.structure)

class Public(torch.nn.Module):
    def __init__(self,shape,side,spelling,structure):
        super().__init__();self.side=side;self.spelling=spelling;self.structure=structure
        size=__import__("math").prod(shape)
        value=((torch.arange(size,dtype=torch.float64)%11)-5)*.03125+.125
        self.register_buffer("weight",value.reshape(shape))
    def forward(self,x):
        value=product(self.weight,x,self.spelling) if self.side=="left" else product(x,self.weight,self.spelling)
        return output(value,self.structure)

def make_model(partition,index,structure="tensor"):
    first,second=PARTITIONS[partition][index]
    side="left" if partition.startswith("public_") and partition.endswith("_l") else "right" if partition.startswith("public_") else None
    return Cipher(index,structure) if side is None else Public(first if side=="left" else second,side,index,structure)

def signature(partition,index):
    first,second=PARTITIONS[partition][index]
    if partition.startswith("public_"):
        return [dict(name="x",shape=second if partition.endswith("_l") else first)]
    return [dict(name="x",shape=first),dict(name="y",shape=second)]
