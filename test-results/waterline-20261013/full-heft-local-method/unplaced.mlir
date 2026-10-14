module {
func.func @heft_waterline(%arg0: tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>> {
%w0 = "ckks.encode"() <{payload = dense<1.0> : tensor<1xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 5>>
%w1 = "ckks.encode"() <{payload = dense<2.0> : tensor<1xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 5>>
%r0 = "ckks.rotatec"(%arg0) <{offset = array<i64: 1>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a0 = "ckks.addcp"(%r0, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b0 = "ckks.addcp"(%a0, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r1 = "ckks.rotatec"(%arg0) <{offset = array<i64: 2>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a1 = "ckks.addcp"(%r1, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b1 = "ckks.addcp"(%a1, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r2 = "ckks.rotatec"(%arg0) <{offset = array<i64: 3>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a2 = "ckks.addcp"(%r2, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b2 = "ckks.addcp"(%a2, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r3 = "ckks.rotatec"(%arg0) <{offset = array<i64: 4>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a3 = "ckks.addcp"(%r3, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b3 = "ckks.addcp"(%a3, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r4 = "ckks.rotatec"(%arg0) <{offset = array<i64: 5>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a4 = "ckks.addcp"(%r4, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b4 = "ckks.addcp"(%a4, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r5 = "ckks.rotatec"(%arg0) <{offset = array<i64: 6>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a5 = "ckks.addcp"(%r5, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b5 = "ckks.addcp"(%a5, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r6 = "ckks.rotatec"(%arg0) <{offset = array<i64: 7>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a6 = "ckks.addcp"(%r6, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b6 = "ckks.addcp"(%a6, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r7 = "ckks.rotatec"(%arg0) <{offset = array<i64: -1>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a7 = "ckks.addcp"(%r7, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b7 = "ckks.addcp"(%a7, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r8 = "ckks.rotatec"(%arg0) <{offset = array<i64: -2>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a8 = "ckks.addcp"(%r8, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b8 = "ckks.addcp"(%a8, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r9 = "ckks.rotatec"(%arg0) <{offset = array<i64: -3>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a9 = "ckks.addcp"(%r9, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b9 = "ckks.addcp"(%a9, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r10 = "ckks.rotatec"(%arg0) <{offset = array<i64: -4>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a10 = "ckks.addcp"(%r10, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b10 = "ckks.addcp"(%a10, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r11 = "ckks.rotatec"(%arg0) <{offset = array<i64: -5>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a11 = "ckks.addcp"(%r11, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b11 = "ckks.addcp"(%a11, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r12 = "ckks.rotatec"(%arg0) <{offset = array<i64: -6>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a12 = "ckks.addcp"(%r12, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b12 = "ckks.addcp"(%a12, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r13 = "ckks.rotatec"(%arg0) <{offset = array<i64: -7>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a13 = "ckks.addcp"(%r13, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b13 = "ckks.addcp"(%a13, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r14 = "ckks.rotatec"(%arg0) <{offset = array<i64: 1>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a14 = "ckks.addcp"(%r14, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b14 = "ckks.addcp"(%a14, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%r15 = "ckks.rotatec"(%arg0) <{offset = array<i64: 2>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%a15 = "ckks.addcp"(%r15, %w0) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%b15 = "ckks.addcp"(%a15, %w1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum0 = "ckks.addcc"(%b0, %b1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum1 = "ckks.addcc"(%b2, %b3) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum2 = "ckks.addcc"(%b4, %b5) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum3 = "ckks.addcc"(%b6, %b7) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum4 = "ckks.addcc"(%b8, %b9) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum5 = "ckks.addcc"(%b10, %b11) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum6 = "ckks.addcc"(%b12, %b13) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum7 = "ckks.addcc"(%b14, %b15) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum8 = "ckks.addcc"(%sum0, %sum1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum9 = "ckks.addcc"(%sum2, %sum3) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum10 = "ckks.addcc"(%sum4, %sum5) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum11 = "ckks.addcc"(%sum6, %sum7) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum12 = "ckks.addcc"(%sum8, %sum9) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum13 = "ckks.addcc"(%sum10, %sum11) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%sum14 = "ckks.addcc"(%sum12, %sum13) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%boot = "ckks.bootstrapc"(%sum14) : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
%result = "ckks.negatec"(%boot) : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
return %result : tensor<1x!ckks.poly<2 * 40 * 5>>
}
}