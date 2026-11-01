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
%sq1 = "earth.mul"(%r0, %r0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a1 = "earth.mul"(%r0, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b1 = "earth.add"(%a1, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r1 = "earth.mul"(%sq1, %b1) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%c1 = "earth.mul"(%r1, %half) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%d1 = "earth.add"(%c1, %one) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%y1 = "earth.mul"(%y0, %d1) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq2 = "earth.mul"(%r1, %r1) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a2 = "earth.mul"(%r1, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b2 = "earth.add"(%a2, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r2 = "earth.mul"(%sq2, %b2) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%c2 = "earth.mul"(%r2, %half) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%d2 = "earth.add"(%c2, %one) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%y2 = "earth.mul"(%y1, %d2) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %y2 : tensor<1x!earth.ci<0 * 0>>
}}
