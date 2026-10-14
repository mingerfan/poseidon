import torch
from torch.nn import functional as F
class Model(torch.nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.register_buffer("std",torch.tensor([.5,-2.,3.],dtype=torch.float64))
        if mode=="batch_norm_module":
            self.bn=torch.nn.BatchNorm1d(3,dtype=torch.float64)
            self.bn.weight.data.copy_(torch.tensor([.7,-.8,1.2],dtype=torch.float64))
            self.bn.bias.data.copy_(torch.tensor([.1,-.2,.3],dtype=torch.float64))
            self.bn.running_mean.copy_(torch.tensor([.2,-.4,.1],dtype=torch.float64))
            self.bn.running_var.copy_(torch.tensor([1.2,.8,2.],dtype=torch.float64))
            self.bn.num_batches_tracked.fill_(7)
        if mode=="batch_norm_function":
            for n,v in {"weight":[.7,-.8,1.2],"bias":[.1,-.2,.3],
              "mean":[.2,-.4,.1],"var":[1.2,.8,2.]}.items():
                self.register_buffer(n,torch.tensor(v,dtype=torch.float64))
            self.register_buffer("unused_counter",torch.tensor(7,dtype=torch.int64))
        if mode in ("index_select_function","mask_index_residual"):
            self.register_buffer("index",torch.tensor([2,0],dtype=torch.int64))
        if mode=="index_method_nested":
            self.public=torch.nn.Module()
            self.public.register_buffer("index",torch.tensor([1,2],dtype=torch.int32))
            self.register_buffer("unused_mask",torch.tensor([True,False],dtype=torch.bool))
        if mode in ("bool_multiply","mask_index_residual"):
            self.register_buffer("mask",torch.tensor([True,False,True],dtype=torch.bool))
        if mode=="divided_relu_residual":
            self.register_buffer("unused_counter",torch.tensor(7,dtype=torch.int64))
        if mode=="divided_gelu_linear":
            self.proj=torch.nn.Linear(3,2,dtype=torch.float64)
            self.proj.weight.data.copy_(torch.tensor([[.23,-.17,.31],[-.42,.15,.07]],dtype=torch.float64))
            self.proj.bias.data.copy_(torch.tensor([.05,-.11],dtype=torch.float64))
        if mode=="divided_sigmoid_tuple":
            self.register_buffer("unused_mask",torch.tensor([True,False,True],dtype=torch.bool))
    def forward(self,x):
        z=x/self.std
        if self.mode=="vector_function":return z
        if self.mode=="batch_norm_module":return self.bn(z)
        if self.mode=="batch_norm_function":
            return F.batch_norm(z,self.mean,self.var,self.weight,self.bias,training=False)
        if self.mode=="index_select_function":return torch.index_select(z,-1,self.index)
        if self.mode=="index_method_nested":return z.index_select(1,self.public.index)
        if self.mode=="bool_multiply":return z*self.mask
        if self.mode=="mask_index_residual":return {"selected":z[:,self.index],"masked":z*self.mask}
        if self.mode=="divided_relu_residual":return F.relu(z)+x
        if self.mode=="divided_gelu_linear":return self.proj(F.gelu(z,approximate="none"))
        if self.mode=="divided_sigmoid_tuple":return (z,torch.sigmoid(z))
        raise ValueError("Unknown fixture mode")
def make(mode):
    aliases={"float32_vector":"vector_function","float32_batch_norm":"batch_norm_module",
      "float32_gelu_projection":"divided_gelu_linear"}
    model=Model(aliases.get(mode,mode)).eval()
    if mode in aliases:model.float()
    if mode=="float32_vector":
        model.std.copy_(torch.tensor([.7,-1.3,2.1],dtype=torch.float32))
    return model
