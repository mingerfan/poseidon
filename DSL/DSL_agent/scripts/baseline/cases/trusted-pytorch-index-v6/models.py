import torch

class IndexModel(torch.nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        vectors={"vector_1":[-1,0,-1],"vector_2":[2,0,2],
                 "index_select_0":[3,0,2],"index_select_1":[2,0,1],"index_select_2":[2,0]}
        if mode in vectors:self.register_buffer("idx",torch.tensor(vectors[mode],dtype=torch.int64))
    def forward(self,x):
        if self.mode=="slicing_0":
            y=x[:,1::2]
            return {"features":y,"energy":y.square().mean(dim=-1)}
        if self.mode=="slicing_1":return x[...,1]+.25
        if self.mode=="slicing_2":return x[-3:-1].square()
        if self.mode=="axes_0":return {"features":x[None,:,None,1:3],"score":x[-1,0]}
        if self.mode=="axes_1":
            y=torch.select(x,dim=-1,index=-1)
            return y.reshape(y.shape[0],-1).square()
        if self.mode=="axes_2":return (x.select(index=1,dim=0),x.select(0,-1))
        if self.mode=="narrow_0":
            y=torch.narrow(input=x,dim=1,start=-3,length=2)
            return (y,y.mean(dim=-1))
        if self.mode=="narrow_1":return x.narrow(dim=0,start=1,length=2)[:,1::2].square()
        if self.mode=="narrow_2":return x.narrow(-1,0,1).sum(dim=1)
        if self.mode=="vector_0":return x[:,[3,0,3]]+.125
        if self.mode=="vector_1":return x[...,self.idx]
        if self.mode=="vector_2":return x[self.idx,:,None].square()
        if self.mode=="index_select_0":
            y=torch.index_select(input=x,dim=-1,index=self.idx)
            return {"features":y,"total":y.sum(dim=-1)}
        if self.mode=="index_select_1":return x.index_select(index=self.idx,dim=0).square()
        if self.mode=="index_select_2":return torch.index_select(x,1,self.idx)
        raise ValueError("Unknown public fixture mode")

def make(mode):return IndexModel(mode)
