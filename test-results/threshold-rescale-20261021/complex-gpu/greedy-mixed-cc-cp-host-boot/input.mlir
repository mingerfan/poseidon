module { func.func @graph(%x: tensor<1x!earth.ci<0 * 0>>, %y: tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>) {
%p = "earth.constant"() {value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%v0 = "earth.mul"(%x, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v1 = "earth.mul"(%v0, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v2 = "earth.mul"(%v1, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v3 = "earth.mul"(%v2, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v4 = "earth.mul"(%v3, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v5 = "earth.mul"(%v4, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v6 = "earth.mul"(%v5, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v7 = "earth.mul"(%v6, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v8 = "earth.mul"(%v7, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v9 = "earth.mul"(%v8, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v10 = "earth.mul"(%v9, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v11 = "earth.mul"(%v10, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v12 = "earth.mul"(%v11, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v13 = "earth.mul"(%v12, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v14 = "earth.mul"(%v13, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v15 = "earth.mul"(%v14, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v16 = "earth.mul"(%v15, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v17 = "earth.mul"(%v16, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v18 = "earth.mul"(%v17, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v19 = "earth.mul"(%v18, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v20 = "earth.mul"(%v19, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v21 = "earth.mul"(%v20, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v22 = "earth.mul"(%v21, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v23 = "earth.mul"(%v22, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v24 = "earth.mul"(%v23, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v25 = "earth.mul"(%v24, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v26 = "earth.mul"(%v25, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v27 = "earth.mul"(%v26, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v28 = "earth.mul"(%v27, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v29 = "earth.mul"(%v28, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v30 = "earth.mul"(%v29, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v31 = "earth.mul"(%v30, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v32 = "earth.mul"(%v31, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v33 = "earth.mul"(%v32, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v34 = "earth.mul"(%v33, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v35 = "earth.mul"(%v34, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v36 = "earth.mul"(%v35, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v37 = "earth.mul"(%v36, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v38 = "earth.mul"(%v37, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v39 = "earth.mul"(%v38, %y) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %v39 : tensor<1x!earth.ci<0 * 0>>
}}
