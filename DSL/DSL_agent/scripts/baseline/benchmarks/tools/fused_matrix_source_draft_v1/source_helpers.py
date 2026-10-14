import torch
def fused(op,style,bias,left,right):
    if style=="method":return getattr(bias,op)(left,right,alpha=1.25,beta=-.5)
    if style=="keywords":
        if op=="addmm":return torch.addmm(input=bias,mat1=left,mat2=right,alpha=1.25,beta=-.5)
        if op=="baddbmm":return torch.baddbmm(input=bias,batch1=left,batch2=right,alpha=1.25,beta=-.5)
        return torch.addbmm(input=bias,batch1=left,batch2=right,alpha=1.25,beta=-.5)
    return getattr(torch,op)(bias,left,right,alpha=1.25,beta=-.5)
