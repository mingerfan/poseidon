module {
  func.func @probe(%arg0: tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 51 * 29>> attributes {absorb_coefficient_scale, arg_scale = array<i64: 40>, earth.arg_scales = array<f64: 4.000000e+01>, earth.local_scale_policy, earth.native_level_anchor = 30 : i64, earth.native_output_level = 30 : i64, earth.native_scale_moduli = array<i64: 1073709057, 2147418113, 4294868993, 1073610753, 2147319809, 4294770689, 1073512449, 2147221505, 4294672385, 1073414145, 2147123201, 4294574081, 1073315841, 2147024897, 4294475777, 1073217537, 2146926593, 4294377473, 1073119233, 2146828289, 4294279169, 1073020929, 2146729985, 4294180865, 1072922625, 2146631681, 4294082561, 1072824321, 2146533377, 4293984257, 1072726017, 2146435073, 4293885953, 1072627713, 2146336769, 4293787649, 1072529409, 2146238465, 4293689345, 1072431105, 2146140161, 4293591041, 1072332801, 2146041857, 4293492737, 1072234497, 2145943553, 4293394433, 1072136193, 2145845249>, earth.result_scales = array<f64: 50.00136549898658>, earth.scale_waterline = 4.000000e+01 : f64, init_level = 30 : i64, res_scale = array<i64: 51>} {
    %0 = "ckks.encode"() <{payload = dense<1.250000e-01> : tensor<1xf64>}> {runtime.scale_log2 = 4.000000e+01 : f64} : () -> tensor<1x!ckks.poly<1 * 40 * 30>>
    %1 = "ckks.mulcp"(%arg0, %0) : (tensor<1x!ckks.poly<2 * 40 * 30>>, tensor<1x!ckks.poly<1 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 80 * 30>>
    %2 = "ckks.rescalec"(%1) : (tensor<1x!ckks.poly<2 * 80 * 30>>) -> tensor<1x!ckks.poly<2 * 51 * 29>>
    return %2 : tensor<1x!ckks.poly<2 * 51 * 29>>
  }
}

