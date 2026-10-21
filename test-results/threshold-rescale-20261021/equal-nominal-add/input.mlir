module { func.func @chains(%x: tensor<1x!earth.ci<0 * 0>>, %y: tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) {
%p = "earth.constant"() {value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%a = "earth.mul"(%x, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b = "earth.mul"(%a, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%c = "earth.mul"(%b, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%d = "earth.add"(%c, %x) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%e = "earth.mul"(%d, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %d, %e : tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>
}}
