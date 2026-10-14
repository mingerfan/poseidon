def masked_selection(x,z,mask,w):
    return (x+z).where(condition=mask.flip(0).reshape(2,1),other=w)
