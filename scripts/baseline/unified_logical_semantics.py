"""Versioned public mathematical input specification; never a DSL recipe or test oracle."""
from benchmark_graph import OPS,validate
OPERATIONS={
 "add":"Elementwise x + y with NumPy/PyTorch right-aligned broadcasting.",
 "subtract":"Elementwise x - y in the listed operand order, with right-aligned broadcasting.",
 "multiply":"Elementwise x * y with right-aligned broadcasting; this is not matrix multiplication.",
 "negate":"Elementwise -x.",
 "square":"Elementwise x squared.",
 "power":"Elementwise x raised to attrs.exponent, which is exactly 2 or 4.",
 "linear":"Inputs are x, public weight W, and optional public bias b. W has shape [out_features,in_features]. y[...,j]=sum_k x[...,k]*W[j,k]+b[j]; omit b if absent. Contract the last axis of x.",
 "flatten":"Flatten every logical tensor axis into one dimension using C order (last axis varies fastest).",
 "reshape":"Keep the C-order sequence of logical values and reinterpret attrs.shape. At most one -1 infers its dimension from the unchanged element count.",
 "transpose":"Swap logical axes attrs.dim0 and attrs.dim1, then represent the resulting tensor in C order. This is a logical value permutation, not merely renaming an Expr object-array shape.",
 "permute":"Output axis i is input axis attrs.dims[i]. The resulting logical tensor is represented in C order. Axes use normalized Python negative indexing.",
 "rotate":"Let N be the number of logical elements and v the C-order flattened input. Output flat element j is v[(j+attrs.step) modulo N], with the same logical shape. Positive step reads a later logical element. N is not the padded ciphertext period P; padded slots are never part of this logical rotation.",
 "concat":"Concatenate the listed logical tensors in input-list order along existing attrs.axis; all other dimensions must match.",
 "stack":"Insert a new axis attrs.axis and stack equal-shaped logical tensors in input-list order. Negative axis is normalized against output rank.",
 "slice":"Slice only attrs.axis with ordinary Python slice(attrs.start,attrs.stop,attrs.step) semantics, including negative bounds/step and clipping. All other axes are unchanged. The validated result is nonempty.",
 "split":"Partition attrs.axis into consecutive lengths attrs.sections in order; bind each part to the corresponding entry of node.outputs.",
 "sum":"Sum over all attrs.axes. With attrs.keepdims=true, reduced axes remain size 1; otherwise remove them. A complete scalar reduction is represented by logical shape [1], never rank 0.",
 "mean":"Arithmetic mean over attrs.axes, dividing the sum by the product of the original reduced dimensions. Output dimensions follow sum/keepdims, including logical shape [1] for a scalar.",
 "batch_norm":"Inference only. Inputs are x, public mean, variance, gamma, beta in that order. Channel is x axis 1. y=(x-mean[channel])*gamma[channel]/sqrt(variance[channel]+attrs.eps)+beta[channel]. No training or batch statistics recomputation.",
 "polynomial":"Inputs are x and a public 1-D coefficient vector c in ascending degree order. basis=power means sum_k c[k]*x**k. basis=chebyshev means sum_k c[k]*T_k(x), where T_0=1, T_1=x, T_(k+1)=2*x*T_k-T_(k-1). Evaluate this polynomial itself; do not substitute an original nonlinear function.",
}
CONV=("Cross-correlation, no kernel reversal. Input is [C,spatial...] or [N,C,spatial...]. Public weight is [Cout,Cin/groups,kernel...] and optional bias [Cout]. "
"For each output channel, sum only the corresponding contiguous input-channel group. Source coordinate is dst*stride-padding+kernel_index*dilation on each spatial axis; out-of-range input is zero. "
"Add bias once per output value. Spatial size is floor((input_size+2*padding-dilation*(kernel_size-1)-1)/stride)+1.")
POOL=("Average pooling independently per channel and batch on [C,spatial...] or [N,C,spatial...]. "
"Source coordinate is dst*stride-padding+kernel_index, with zero outside the input. "
"If count_include_pad=true, divide by the full kernel element count; otherwise divide by the count of valid input locations in that window. "
"No dilation or ceil mode. Spatial size is floor((input_size+2*padding-kernel_size)/stride)+1.")
OPERATIONS.update(conv1d=CONV,conv2d=CONV,avg_pool1d=POOL,avg_pool2d=POOL)

def specification(model):
 validate(model)
 if set(OPERATIONS)!=set(OPS):raise ValueError("Logical operator specification coverage changed")
 names=sorted({node["op"] for node in model["nodes"]})
 return dict(version="logical-tensors-v1",
  scope="Mathematical definition of the public input graph; no DSL implementation or test answers.",
  ordering="Named input/output order and node output order are significant. Tensor flattening always uses C order.",
  attributes="All required attributes are explicit in each node; no implicit framework defaults are to be inferred.",
  broadcasting="Align dimensions from the right. Each pair must be equal or contain 1. Public broadcasting cannot expand the encrypted operand shape.",
  axes="Negative existing axes are normalized against input rank; stack inserts an axis and normalizes against output rank.",
  layout_boundary="Model operators act on logical tensors. Padding, repeated periods, chunks and Expr object-array storage are representation details, not extra mathematical elements.",
  operations={name:OPERATIONS[name] for name in names})
