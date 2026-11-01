module { func.func @graph(%x: tensor<1x!earth.ci<0 * 0>>, %y: tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>> {
%quarter = "earth.constant"() {value = dense<0.25> : tensor<8192xf64>, rms_var = 0.25 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%threequarter = "earth.constant"() {value = dense<0.75> : tensor<8192xf64>, rms_var = 0.75 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%half = "earth.constant"() {value = dense<0.5> : tensor<8192xf64>, rms_var = 0.5 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%one = "earth.constant"() {value = dense<1.0> : tensor<8192xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%sq0 = "earth.mul"(%x, %x) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a0 = "earth.mul"(%x, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b0 = "earth.add"(%a0, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r0 = "earth.mul"(%sq0, %b0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq1 = "earth.mul"(%r0, %r0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a1 = "earth.mul"(%r0, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b1 = "earth.add"(%a1, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r1 = "earth.mul"(%sq1, %b1) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq2 = "earth.mul"(%r1, %r1) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a2 = "earth.mul"(%r1, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b2 = "earth.add"(%a2, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r2 = "earth.mul"(%sq2, %b2) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq3 = "earth.mul"(%r2, %r2) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a3 = "earth.mul"(%r2, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b3 = "earth.add"(%a3, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r3 = "earth.mul"(%sq3, %b3) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq4 = "earth.mul"(%r3, %r3) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a4 = "earth.mul"(%r3, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b4 = "earth.add"(%a4, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r4 = "earth.mul"(%sq4, %b4) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq5 = "earth.mul"(%r4, %r4) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a5 = "earth.mul"(%r4, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b5 = "earth.add"(%a5, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r5 = "earth.mul"(%sq5, %b5) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq6 = "earth.mul"(%r5, %r5) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a6 = "earth.mul"(%r5, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b6 = "earth.add"(%a6, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r6 = "earth.mul"(%sq6, %b6) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%sq7 = "earth.mul"(%r6, %r6) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%a7 = "earth.mul"(%r6, %quarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%b7 = "earth.add"(%a7, %threequarter) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%r7 = "earth.mul"(%sq7, %b7) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %r7 : tensor<1x!earth.ci<0 * 0>>
}}
