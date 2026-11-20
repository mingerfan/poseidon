"""Reuse the trusted original Torch graph interpreter with owned weight views.

Original file mappings remain read-only.  Ordinary arrays are copied into
private immutable ownership.  The original interpreter code and operations are
reused unchanged, with a local NumPy facade avoiding its full weight copies.
No model-provided or language-model-provided source is executed here.
"""

import copy
import types


class MappedReference:
    def __init__(self, graph):
        import numpy as np
        from .graph import Graph,Node
        from . import graph_exec
        weights={}
        mapped=private=0
        for name,value in graph.weights.items():
            if not isinstance(value,np.ndarray) or value.dtype != np.float64:
                raise ValueError('The original graph requires float64 weight arrays')
            if isinstance(value,np.memmap) and value.mode=='r' and not value.flags.writeable:
                weights[name]=value;mapped+=1
            else:
                weights[name]=np.array(value,copy=True);weights[name].flags.writeable=False;private+=1
        nodes=[Node(node.id,node.op,list(node.inputs),list(node.outputs),copy.deepcopy(node.attrs),node.src)
               for node in graph.nodes]
        self._graph=Graph(graph.name,nodes,copy.deepcopy(graph.tensors),weights,list(graph.inputs),list(graph.outputs))
        self.ownership=dict(mapped_weight_views=mapped,private_weight_copies=private,
            weight_tensors=len(weights),all_numpy_weights_readonly=all(not value.flags.writeable for value in weights.values()),
            exposed_weight_tensors=False,original_interpreter_code_reused=True)
        self._view_calls=0
        owner=self
        class ArrayFacade:
            def array(self,value,*args,**kwargs):
                if id(value) in owned and not args and not kwargs:
                    owner._view_calls+=1
                    return np.asarray(value)
                return np.array(value,*args,**kwargs)
            def __getattr__(self,name):return getattr(np,name)
        owned={id(value) for value in weights.values()}
        original=graph_exec.forward
        namespace=dict(original.__globals__,np=ArrayFacade())
        self._forward=types.FunctionType(original.__code__,namespace,name=original.__name__,argdefs=original.__defaults__,closure=original.__closure__)
        self._forward.__kwdefaults__=copy.copy(original.__kwdefaults__)

    def forward(self,inputs,*,keep=False):
        import numpy as np
        import torch
        # Inputs are small real samples, independently owned during execution.
        private={name:(value.detach().clone() if torch.is_tensor(value) else np.array(value,dtype=np.float64,copy=True))
                 for name,value in inputs.items()}
        with torch.no_grad():result=self._forward(self._graph,private,keep=keep)
        if keep:
            # Calibration receives activations; weight views stay private.
            return {name:value for name,value in result.items() if name not in self._graph.weights}
        return result

    @property
    def observation(self):
        return dict(self.ownership,interpreter_weight_view_calls=self._view_calls)
