module { func.func @graph(%x: tensor<1x!earth.ci<0 * 0>>, %y: tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>> {
%quarter = "earth.constant"() {value = dense<0.25> : tensor<8192xf64>, rms_var = 0.25 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%threequarter = "earth.constant"() {value = dense<0.75> : tensor<8192xf64>, rms_var = 0.75 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%half = "earth.constant"() {value = dense<0.5> : tensor<8192xf64>, rms_var = 0.5 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%one = "earth.constant"() {value = dense<1.0> : tensor<8192xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%sq0 = "earth.mul"(%x, %x) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a0 = "earth.mul"(%x, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b0 = "earth.add"(%a0, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r0 = "earth.mul"(%sq0, %b0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%c0 = "earth.mul"(%r0, %half) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%d0 = "earth.add"(%c0, %one) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%y0 = "earth.mul"(%y, %d0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %y0 : tensor<1x!earth.ci<0 * 0>>
}}
