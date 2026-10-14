import torch
class Model(torch.nn.Module):
    def __init__(self,method,context):
        super().__init__();self.method=method;self.context=context
        if context=="scalar_linear":self.linear=torch.nn.Linear(3,2,bias=True,dtype=torch.float64)
    def forward(self,x,y):
        if self.context=="arithmetic":
            source=x
        elif self.context=="transpose_alias":
            source=x.transpose(0,1).clone()
        else:
            source=x.mean()
        z=source.expand_as(other=y) if self.method=="expand_as" else source.broadcast_to(size=y.shape)
        if self.context=="arithmetic":return z+y
        if self.context=="transpose_alias":
            source.add_(0.25)
            return (z*y,z.mean(dim=0))
        return {"result":self.linear(z)+y[:,:2]}
def build(method,context):return Model(method,context).double().eval()
