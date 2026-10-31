module {
  func.func @one_relu(%arg0: tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 41 * 8>> attributes {absorb_coefficient_scale, arg_scale = array<i64: 40>, earth.arg_scales = array<f64: 4.000000e+01>, earth.native_level_anchor = 30 : i64, earth.native_output_level = 30 : i64, earth.native_scale_moduli = array<i64: 4255252481, 4255645697, 4258922497, 4260364289, 4261675009, 4264427521, 4265476097, 4266393601, 4267442177, 1040842753, 1042415617, 1043464193, 1045430273, 1048707073, 1049100289, 1051721729, 1052508161, 1053818881, 1054212097, 1055260673, 1056178177, 1056440321, 1060765697, 1062469633, 1062862849, 1064697857, 1065484289, 1068236801, 1070727169, 1071513601, 1073479681, 2135162881, 4272291841, 4272685057, 1040056321, 2135818241, 4273340417, 2142502913, 4274126849, 4276092929, 2144468993, 2146041857, 2146959361, 4277403649, 4279369729, 2147352577, 4280025089, 4280156161, 4281204737, 1038745601>, earth.region_scale_policy, earth.regional_boot_count = 0 : i64, earth.regional_queries = 24695 : i64, earth.regional_regions = 22 : i64, earth.result_scales = array<f64: 40.000000001044064>, earth.scale_waterline = 4.000000e+01 : f64, init_level = 30 : i64, res_scale = array<i64: 41>} {
    %0 = "ckks.addcc"(%arg0, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 30>>, tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 30>>
    %1 = "ckks.mulcc"(%0, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 30>>, tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<3 * 80 * 30>>
    %2 = "ckks.relinearize"(%1) : (tensor<1x!ckks.poly<3 * 80 * 30>>) -> tensor<1x!ckks.poly<2 * 80 * 30>>
    %3 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 8.000000e+01 : f64} : () -> tensor<1x!ckks.poly<1 * 80 * 30>>
    %4 = "ckks.addcp"(%2, %3) : (tensor<1x!ckks.poly<2 * 80 * 30>>, tensor<1x!ckks.poly<1 * 80 * 30>>) -> tensor<1x!ckks.poly<2 * 80 * 30>>
    %5 = "ckks.rescalec"(%4) : (tensor<1x!ckks.poly<2 * 80 * 30>>) -> tensor<1x!ckks.poly<2 * 51 * 29>>
    %6 = "ckks.modswitchc"(%0) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 29>>
    %7 = "ckks.mulcc"(%6, %5) : (tensor<1x!ckks.poly<2 * 40 * 29>>, tensor<1x!ckks.poly<2 * 51 * 29>>) -> tensor<1x!ckks.poly<3 * 91 * 29>>
    %8 = "ckks.relinearize"(%7) : (tensor<1x!ckks.poly<3 * 91 * 29>>) -> tensor<1x!ckks.poly<2 * 91 * 29>>
    %9 = "ckks.negatec"(%arg0) : (tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 30>>
    %10 = "ckks.modswitchc"(%9) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 29>>
    %11 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 50.000352262127691 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 29>>
    %12 = "ckks.mulcp"(%10, %11) : (tensor<1x!ckks.poly<2 * 40 * 29>>, tensor<1x!ckks.poly<1 * 51 * 29>>) -> tensor<1x!ckks.poly<2 * 91 * 29>>
    %13 = "ckks.addcc"(%8, %12) : (tensor<1x!ckks.poly<2 * 91 * 29>>, tensor<1x!ckks.poly<2 * 91 * 29>>) -> tensor<1x!ckks.poly<2 * 91 * 29>>
    %14 = "ckks.rescalec"(%13) : (tensor<1x!ckks.poly<2 * 91 * 29>>) -> tensor<1x!ckks.poly<2 * 61 * 28>>
    %15 = "ckks.addcc"(%5, %5) : (tensor<1x!ckks.poly<2 * 51 * 29>>, tensor<1x!ckks.poly<2 * 51 * 29>>) -> tensor<1x!ckks.poly<2 * 51 * 29>>
    %16 = "ckks.mulcc"(%15, %5) : (tensor<1x!ckks.poly<2 * 51 * 29>>, tensor<1x!ckks.poly<2 * 51 * 29>>) -> tensor<1x!ckks.poly<3 * 102 * 29>>
    %17 = "ckks.relinearize"(%16) : (tensor<1x!ckks.poly<3 * 102 * 29>>) -> tensor<1x!ckks.poly<2 * 102 * 29>>
    %18 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 100.00070452425538 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 29>>
    %19 = "ckks.addcp"(%17, %18) : (tensor<1x!ckks.poly<2 * 102 * 29>>, tensor<1x!ckks.poly<1 * 102 * 29>>) -> tensor<1x!ckks.poly<2 * 102 * 29>>
    %20 = "ckks.rescalec"(%19) : (tensor<1x!ckks.poly<2 * 102 * 29>>) -> tensor<1x!ckks.poly<2 * 71 * 28>>
    %21 = "ckks.rescalec"(%20) : (tensor<1x!ckks.poly<2 * 71 * 28>>) -> tensor<1x!ckks.poly<2 * 41 * 27>>
    %22 = "ckks.addcc"(%21, %21) : (tensor<1x!ckks.poly<2 * 41 * 27>>, tensor<1x!ckks.poly<2 * 41 * 27>>) -> tensor<1x!ckks.poly<2 * 41 * 27>>
    %23 = "ckks.mulcc"(%22, %21) : (tensor<1x!ckks.poly<2 * 41 * 27>>, tensor<1x!ckks.poly<2 * 41 * 27>>) -> tensor<1x!ckks.poly<3 * 82 * 27>>
    %24 = "ckks.relinearize"(%23) : (tensor<1x!ckks.poly<3 * 82 * 27>>) -> tensor<1x!ckks.poly<2 * 82 * 27>>
    %25 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.015515477047628 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 27>>
    %26 = "ckks.addcp"(%24, %25) : (tensor<1x!ckks.poly<2 * 82 * 27>>, tensor<1x!ckks.poly<1 * 82 * 27>>) -> tensor<1x!ckks.poly<2 * 82 * 27>>
    %27 = "ckks.rescalec"(%26) : (tensor<1x!ckks.poly<2 * 82 * 27>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %28 = "ckks.encode"() <{payload = dense<-4.0270201200636233> : tensor<32768xf64>}> {runtime.scale_log2 = 40.030796841152579 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 30>>
    %29 = "ckks.mulcp"(%arg0, %28) : (tensor<1x!ckks.poly<2 * 40 * 30>>, tensor<1x!ckks.poly<1 * 41 * 30>>) -> tensor<1x!ckks.poly<2 * 81 * 30>>
    %30 = "ckks.rescalec"(%29) : (tensor<1x!ckks.poly<2 * 81 * 30>>) -> tensor<1x!ckks.poly<2 * 51 * 29>>
    %31 = "ckks.mulcc"(%30, %5) : (tensor<1x!ckks.poly<2 * 51 * 29>>, tensor<1x!ckks.poly<2 * 51 * 29>>) -> tensor<1x!ckks.poly<3 * 102 * 29>>
    %32 = "ckks.relinearize"(%31) : (tensor<1x!ckks.poly<3 * 102 * 29>>) -> tensor<1x!ckks.poly<2 * 102 * 29>>
    %33 = "ckks.rescalec"(%32) : (tensor<1x!ckks.poly<2 * 102 * 29>>) -> tensor<1x!ckks.poly<2 * 71 * 28>>
    %34 = "ckks.rescalec"(%33) : (tensor<1x!ckks.poly<2 * 71 * 28>>) -> tensor<1x!ckks.poly<2 * 41 * 27>>
    %35 = "ckks.modswitchc"(%arg0) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 29>>
    %36 = "ckks.encode"() <{payload = dense<2.243187978526648> : tensor<32768xf64>}> {runtime.scale_log2 = 60.031501365407962 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 29>>
    %37 = "ckks.mulcp"(%35, %36) : (tensor<1x!ckks.poly<2 * 40 * 29>>, tensor<1x!ckks.poly<1 * 61 * 29>>) -> tensor<1x!ckks.poly<2 * 101 * 29>>
    %38 = "ckks.rescalec"(%37) : (tensor<1x!ckks.poly<2 * 101 * 29>>) -> tensor<1x!ckks.poly<2 * 71 * 28>>
    %39 = "ckks.rescalec"(%38) : (tensor<1x!ckks.poly<2 * 71 * 28>>) -> tensor<1x!ckks.poly<2 * 41 * 27>>
    %40 = "ckks.addcc"(%34, %39) : (tensor<1x!ckks.poly<2 * 41 * 27>>, tensor<1x!ckks.poly<2 * 41 * 27>>) -> tensor<1x!ckks.poly<2 * 41 * 27>>
    %41 = "ckks.mulcc"(%40, %21) : (tensor<1x!ckks.poly<2 * 41 * 27>>, tensor<1x!ckks.poly<2 * 41 * 27>>) -> tensor<1x!ckks.poly<3 * 82 * 27>>
    %42 = "ckks.relinearize"(%41) : (tensor<1x!ckks.poly<3 * 82 * 27>>) -> tensor<1x!ckks.poly<2 * 82 * 27>>
    %43 = "ckks.rescalec"(%42) : (tensor<1x!ckks.poly<2 * 82 * 27>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %44 = "ckks.modswitchc"(%arg0) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    %45 = "ckks.encode"() <{payload = dense<1.1583708205212988> : tensor<32768xf64>}> {runtime.scale_log2 = 40.046312318200137 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 27>>
    %46 = "ckks.mulcp"(%44, %45) : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 41 * 27>>) -> tensor<1x!ckks.poly<2 * 81 * 27>>
    %47 = "ckks.rescalec"(%46) : (tensor<1x!ckks.poly<2 * 81 * 27>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %48 = "ckks.encode"() <{payload = dense<-0.24378614014677616> : tensor<32768xf64>}> {runtime.scale_log2 = 50.038906841804078 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 28>>
    %49 = "ckks.mulcp"(%14, %48) : (tensor<1x!ckks.poly<2 * 61 * 28>>, tensor<1x!ckks.poly<1 * 51 * 28>>) -> tensor<1x!ckks.poly<2 * 112 * 28>>
    %50 = "ckks.rescalec"(%49) : (tensor<1x!ckks.poly<2 * 112 * 28>>) -> tensor<1x!ckks.poly<2 * 81 * 27>>
    %51 = "ckks.rescalec"(%50) : (tensor<1x!ckks.poly<2 * 81 * 27>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %52 = "ckks.addcc"(%47, %51) : (tensor<1x!ckks.poly<2 * 51 * 26>>, tensor<1x!ckks.poly<2 * 51 * 26>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %53 = "ckks.addcc"(%43, %52) : (tensor<1x!ckks.poly<2 * 51 * 26>>, tensor<1x!ckks.poly<2 * 51 * 26>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %54 = "ckks.mulcc"(%53, %27) : (tensor<1x!ckks.poly<2 * 51 * 26>>, tensor<1x!ckks.poly<2 * 51 * 26>>) -> tensor<1x!ckks.poly<3 * 102 * 26>>
    %55 = "ckks.relinearize"(%54) : (tensor<1x!ckks.poly<3 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 102 * 26>>
    %56 = "ckks.rescalec"(%55) : (tensor<1x!ckks.poly<2 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 71 * 25>>
    %57 = "ckks.rescalec"(%56) : (tensor<1x!ckks.poly<2 * 71 * 25>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %58 = "ckks.encode"() <{payload = dense<0.38878894178135931> : tensor<32768xf64>}> {runtime.scale_log2 = 50.061485711367624 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 27>>
    %59 = "ckks.mulcp"(%44, %58) : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 51 * 27>>) -> tensor<1x!ckks.poly<2 * 91 * 27>>
    %60 = "ckks.encode"() <{payload = dense<-0.34103182272202132> : tensor<32768xf64>}> {runtime.scale_log2 = 60.054080234971508 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 28>>
    %61 = "ckks.mulcp"(%14, %60) : (tensor<1x!ckks.poly<2 * 61 * 28>>, tensor<1x!ckks.poly<1 * 61 * 28>>) -> tensor<1x!ckks.poly<2 * 122 * 28>>
    %62 = "ckks.rescalec"(%61) : (tensor<1x!ckks.poly<2 * 122 * 28>>) -> tensor<1x!ckks.poly<2 * 91 * 27>>
    %63 = "ckks.addcc"(%59, %62) : (tensor<1x!ckks.poly<2 * 91 * 27>>, tensor<1x!ckks.poly<2 * 91 * 27>>) -> tensor<1x!ckks.poly<2 * 91 * 27>>
    %64 = "ckks.mulcc"(%63, %21) : (tensor<1x!ckks.poly<2 * 91 * 27>>, tensor<1x!ckks.poly<2 * 41 * 27>>) -> tensor<1x!ckks.poly<3 * 132 * 27>>
    %65 = "ckks.relinearize"(%64) : (tensor<1x!ckks.poly<3 * 132 * 27>>) -> tensor<1x!ckks.poly<2 * 132 * 27>>
    %66 = "ckks.rescalec"(%65) : (tensor<1x!ckks.poly<2 * 132 * 27>>) -> tensor<1x!ckks.poly<2 * 101 * 26>>
    %67 = "ckks.rescalec"(%66) : (tensor<1x!ckks.poly<2 * 101 * 26>>) -> tensor<1x!ckks.poly<2 * 71 * 25>>
    %68 = "ckks.rescalec"(%67) : (tensor<1x!ckks.poly<2 * 71 * 25>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %69 = "ckks.modswitchc"(%arg0) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 26>>
    %70 = "ckks.encode"() <{payload = dense<1.3110156470890746> : tensor<32768xf64>}> {runtime.scale_log2 = 60.076659104535082 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 26>>
    %71 = "ckks.mulcp"(%69, %70) : (tensor<1x!ckks.poly<2 * 40 * 26>>, tensor<1x!ckks.poly<1 * 61 * 26>>) -> tensor<1x!ckks.poly<2 * 101 * 26>>
    %72 = "ckks.rescalec"(%71) : (tensor<1x!ckks.poly<2 * 101 * 26>>) -> tensor<1x!ckks.poly<2 * 71 * 25>>
    %73 = "ckks.rescalec"(%72) : (tensor<1x!ckks.poly<2 * 71 * 25>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %74 = "ckks.modswitchc"(%14) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 28>>) -> tensor<1x!ckks.poly<2 * 61 * 26>>
    %75 = "ckks.encode"() <{payload = dense<-0.46554445822061141> : tensor<32768xf64>}> {runtime.scale_log2 = 40.073309859026608 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 26>>
    %76 = "ckks.mulcp"(%74, %75) : (tensor<1x!ckks.poly<2 * 61 * 26>>, tensor<1x!ckks.poly<1 * 41 * 26>>) -> tensor<1x!ckks.poly<2 * 102 * 26>>
    %77 = "ckks.rescalec"(%76) : (tensor<1x!ckks.poly<2 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 71 * 25>>
    %78 = "ckks.rescalec"(%77) : (tensor<1x!ckks.poly<2 * 71 * 25>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %79 = "ckks.addcc"(%73, %78) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %80 = "ckks.addcc"(%68, %79) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %81 = "ckks.addcc"(%57, %80) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %82 = "ckks.addcc"(%81, %81) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %83 = "ckks.mulcc"(%82, %81) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<3 * 82 * 24>>
    %84 = "ckks.relinearize"(%83) : (tensor<1x!ckks.poly<3 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 82 * 24>>
    %85 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.200000000000216 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 24>>
    %86 = "ckks.addcp"(%84, %85) : (tensor<1x!ckks.poly<2 * 82 * 24>>, tensor<1x!ckks.poly<1 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 82 * 24>>
    %87 = "ckks.rescalec"(%86) : (tensor<1x!ckks.poly<2 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %88 = "ckks.modswitchc"(%82) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 23>>
    %89 = "ckks.mulcc"(%88, %87) : (tensor<1x!ckks.poly<2 * 41 * 23>>, tensor<1x!ckks.poly<2 * 51 * 23>>) -> tensor<1x!ckks.poly<3 * 92 * 23>>
    %90 = "ckks.relinearize"(%89) : (tensor<1x!ckks.poly<3 * 92 * 23>>) -> tensor<1x!ckks.poly<2 * 92 * 23>>
    %91 = "ckks.negatec"(%81) : (tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %92 = "ckks.modswitchc"(%91) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 23>>
    %93 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 50.214691701386776 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 23>>
    %94 = "ckks.mulcp"(%92, %93) : (tensor<1x!ckks.poly<2 * 41 * 23>>, tensor<1x!ckks.poly<1 * 51 * 23>>) -> tensor<1x!ckks.poly<2 * 92 * 23>>
    %95 = "ckks.addcc"(%90, %94) : (tensor<1x!ckks.poly<2 * 92 * 23>>, tensor<1x!ckks.poly<2 * 92 * 23>>) -> tensor<1x!ckks.poly<2 * 92 * 23>>
    %96 = "ckks.rescalec"(%95) : (tensor<1x!ckks.poly<2 * 92 * 23>>) -> tensor<1x!ckks.poly<2 * 61 * 22>>
    %97 = "ckks.addcc"(%87, %87) : (tensor<1x!ckks.poly<2 * 51 * 23>>, tensor<1x!ckks.poly<2 * 51 * 23>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %98 = "ckks.mulcc"(%97, %87) : (tensor<1x!ckks.poly<2 * 51 * 23>>, tensor<1x!ckks.poly<2 * 51 * 23>>) -> tensor<1x!ckks.poly<3 * 102 * 23>>
    %99 = "ckks.relinearize"(%98) : (tensor<1x!ckks.poly<3 * 102 * 23>>) -> tensor<1x!ckks.poly<2 * 102 * 23>>
    %100 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 100.42938340277399 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 23>>
    %101 = "ckks.addcp"(%99, %100) : (tensor<1x!ckks.poly<2 * 102 * 23>>, tensor<1x!ckks.poly<1 * 102 * 23>>) -> tensor<1x!ckks.poly<2 * 102 * 23>>
    %102 = "ckks.rescalec"(%101) : (tensor<1x!ckks.poly<2 * 102 * 23>>) -> tensor<1x!ckks.poly<2 * 71 * 22>>
    %103 = "ckks.rescalec"(%102) : (tensor<1x!ckks.poly<2 * 71 * 22>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %104 = "ckks.addcc"(%103, %103) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %105 = "ckks.mulcc"(%104, %103) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<3 * 82 * 21>>
    %106 = "ckks.relinearize"(%105) : (tensor<1x!ckks.poly<3 * 82 * 21>>) -> tensor<1x!ckks.poly<2 * 82 * 21>>
    %107 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.924300119763757 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 21>>
    %108 = "ckks.addcp"(%106, %107) : (tensor<1x!ckks.poly<2 * 82 * 21>>, tensor<1x!ckks.poly<1 * 82 * 21>>) -> tensor<1x!ckks.poly<2 * 82 * 21>>
    %109 = "ckks.rescalec"(%108) : (tensor<1x!ckks.poly<2 * 82 * 21>>) -> tensor<1x!ckks.poly<2 * 51 * 20>>
    %110 = "ckks.encode"() <{payload = dense<-4.1144251614039149> : tensor<32768xf64>}> {runtime.scale_log2 = 48.155686376451889 : f64} : () -> tensor<1x!ckks.poly<1 * 49 * 24>>
    %111 = "ckks.mulcp"(%81, %110) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<1 * 49 * 24>>) -> tensor<1x!ckks.poly<2 * 90 * 24>>
    %112 = "ckks.rescalec"(%111) : (tensor<1x!ckks.poly<2 * 90 * 24>>) -> tensor<1x!ckks.poly<2 * 59 * 23>>
    %113 = "ckks.mulcc"(%112, %87) : (tensor<1x!ckks.poly<2 * 59 * 23>>, tensor<1x!ckks.poly<2 * 51 * 23>>) -> tensor<1x!ckks.poly<3 * 110 * 23>>
    %114 = "ckks.relinearize"(%113) : (tensor<1x!ckks.poly<3 * 110 * 23>>) -> tensor<1x!ckks.poly<2 * 110 * 23>>
    %115 = "ckks.rescalec"(%114) : (tensor<1x!ckks.poly<2 * 110 * 23>>) -> tensor<1x!ckks.poly<2 * 79 * 22>>
    %116 = "ckks.rescalec"(%115) : (tensor<1x!ckks.poly<2 * 79 * 22>>) -> tensor<1x!ckks.poly<2 * 49 * 21>>
    %117 = "ckks.modswitchc"(%81) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 23>>
    %118 = "ckks.encode"() <{payload = dense<0.93680785180227221> : tensor<32768xf64>}> {runtime.scale_log2 = 68.385069779225461 : f64} : () -> tensor<1x!ckks.poly<1 * 69 * 23>>
    %119 = "ckks.mulcp"(%117, %118) : (tensor<1x!ckks.poly<2 * 41 * 23>>, tensor<1x!ckks.poly<1 * 69 * 23>>) -> tensor<1x!ckks.poly<2 * 110 * 23>>
    %120 = "ckks.rescalec"(%119) : (tensor<1x!ckks.poly<2 * 110 * 23>>) -> tensor<1x!ckks.poly<2 * 79 * 22>>
    %121 = "ckks.rescalec"(%120) : (tensor<1x!ckks.poly<2 * 79 * 22>>) -> tensor<1x!ckks.poly<2 * 49 * 21>>
    %122 = "ckks.addcc"(%116, %121) : (tensor<1x!ckks.poly<2 * 49 * 21>>, tensor<1x!ckks.poly<2 * 49 * 21>>) -> tensor<1x!ckks.poly<2 * 49 * 21>>
    %123 = "ckks.mulcc"(%122, %103) : (tensor<1x!ckks.poly<2 * 49 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<3 * 90 * 21>>
    %124 = "ckks.relinearize"(%123) : (tensor<1x!ckks.poly<3 * 90 * 21>>) -> tensor<1x!ckks.poly<2 * 90 * 21>>
    %125 = "ckks.rescalec"(%124) : (tensor<1x!ckks.poly<2 * 90 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 20>>
    %126 = "ckks.modswitchc"(%81) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %127 = "ckks.encode"() <{payload = dense<0.13312678038183748> : tensor<32768xf64>}> {runtime.scale_log2 = 48.879986496214705 : f64} : () -> tensor<1x!ckks.poly<1 * 49 * 21>>
    %128 = "ckks.mulcp"(%126, %127) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<1 * 49 * 21>>) -> tensor<1x!ckks.poly<2 * 90 * 21>>
    %129 = "ckks.rescalec"(%128) : (tensor<1x!ckks.poly<2 * 90 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 20>>
    %130 = "ckks.encode"() <{payload = dense<-0.45988074662238737> : tensor<32768xf64>}> {runtime.scale_log2 = 58.632528137720044 : f64} : () -> tensor<1x!ckks.poly<1 * 59 * 22>>
    %131 = "ckks.mulcp"(%96, %130) : (tensor<1x!ckks.poly<2 * 61 * 22>>, tensor<1x!ckks.poly<1 * 59 * 22>>) -> tensor<1x!ckks.poly<2 * 120 * 22>>
    %132 = "ckks.rescalec"(%131) : (tensor<1x!ckks.poly<2 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 89 * 21>>
    %133 = "ckks.rescalec"(%132) : (tensor<1x!ckks.poly<2 * 89 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 20>>
    %134 = "ckks.addcc"(%129, %133) : (tensor<1x!ckks.poly<2 * 60 * 20>>, tensor<1x!ckks.poly<2 * 60 * 20>>) -> tensor<1x!ckks.poly<2 * 60 * 20>>
    %135 = "ckks.addcc"(%125, %134) : (tensor<1x!ckks.poly<2 * 60 * 20>>, tensor<1x!ckks.poly<2 * 60 * 20>>) -> tensor<1x!ckks.poly<2 * 60 * 20>>
    %136 = "ckks.mulcc"(%135, %109) : (tensor<1x!ckks.poly<2 * 60 * 20>>, tensor<1x!ckks.poly<2 * 51 * 20>>) -> tensor<1x!ckks.poly<3 * 111 * 20>>
    %137 = "ckks.relinearize"(%136) : (tensor<1x!ckks.poly<3 * 111 * 20>>) -> tensor<1x!ckks.poly<2 * 111 * 20>>
    %138 = "ckks.rescalec"(%137) : (tensor<1x!ckks.poly<2 * 111 * 20>>) -> tensor<1x!ckks.poly<2 * 80 * 19>>
    %139 = "ckks.rescalec"(%138) : (tensor<1x!ckks.poly<2 * 80 * 19>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %140 = "ckks.encode"() <{payload = dense<-0.058402193560527074> : tensor<32768xf64>}> {runtime.scale_log2 = 59.365572431315165 : f64} : () -> tensor<1x!ckks.poly<1 * 60 * 21>>
    %141 = "ckks.mulcp"(%126, %140) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<1 * 60 * 21>>) -> tensor<1x!ckks.poly<2 * 101 * 21>>
    %142 = "ckks.encode"() <{payload = dense<-0.56048531218535647> : tensor<32768xf64>}> {runtime.scale_log2 = 69.118114072820518 : f64} : () -> tensor<1x!ckks.poly<1 * 70 * 22>>
    %143 = "ckks.mulcp"(%96, %142) : (tensor<1x!ckks.poly<2 * 61 * 22>>, tensor<1x!ckks.poly<1 * 70 * 22>>) -> tensor<1x!ckks.poly<2 * 131 * 22>>
    %144 = "ckks.rescalec"(%143) : (tensor<1x!ckks.poly<2 * 131 * 22>>) -> tensor<1x!ckks.poly<2 * 100 * 21>>
    %145 = "ckks.addcc"(%141, %144) : (tensor<1x!ckks.poly<2 * 101 * 21>>, tensor<1x!ckks.poly<2 * 100 * 21>>) -> tensor<1x!ckks.poly<2 * 101 * 21>>
    %146 = "ckks.mulcc"(%145, %103) : (tensor<1x!ckks.poly<2 * 101 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<3 * 142 * 21>>
    %147 = "ckks.relinearize"(%146) : (tensor<1x!ckks.poly<3 * 142 * 21>>) -> tensor<1x!ckks.poly<2 * 142 * 21>>
    %148 = "ckks.rescalec"(%147) : (tensor<1x!ckks.poly<2 * 142 * 21>>) -> tensor<1x!ckks.poly<2 * 110 * 20>>
    %149 = "ckks.rescalec"(%148) : (tensor<1x!ckks.poly<2 * 110 * 20>>) -> tensor<1x!ckks.poly<2 * 80 * 19>>
    %150 = "ckks.rescalec"(%149) : (tensor<1x!ckks.poly<2 * 80 * 19>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %151 = "ckks.modswitchc"(%81) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 20>>
    %152 = "ckks.encode"() <{payload = dense<0.77658111917095896> : tensor<32768xf64>}> {runtime.scale_log2 = 69.851158366415717 : f64} : () -> tensor<1x!ckks.poly<1 * 70 * 20>>
    %153 = "ckks.mulcp"(%151, %152) : (tensor<1x!ckks.poly<2 * 41 * 20>>, tensor<1x!ckks.poly<1 * 70 * 20>>) -> tensor<1x!ckks.poly<2 * 111 * 20>>
    %154 = "ckks.rescalec"(%153) : (tensor<1x!ckks.poly<2 * 111 * 20>>) -> tensor<1x!ckks.poly<2 * 80 * 19>>
    %155 = "ckks.rescalec"(%154) : (tensor<1x!ckks.poly<2 * 80 * 19>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %156 = "ckks.modswitchc"(%96) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 22>>) -> tensor<1x!ckks.poly<2 * 61 * 20>>
    %157 = "ckks.encode"() <{payload = dense<-0.68184097760015783> : tensor<32768xf64>}> {runtime.scale_log2 = 49.621241126433645 : f64} : () -> tensor<1x!ckks.poly<1 * 50 * 20>>
    %158 = "ckks.mulcp"(%156, %157) : (tensor<1x!ckks.poly<2 * 61 * 20>>, tensor<1x!ckks.poly<1 * 50 * 20>>) -> tensor<1x!ckks.poly<2 * 111 * 20>>
    %159 = "ckks.rescalec"(%158) : (tensor<1x!ckks.poly<2 * 111 * 20>>) -> tensor<1x!ckks.poly<2 * 80 * 19>>
    %160 = "ckks.rescalec"(%159) : (tensor<1x!ckks.poly<2 * 80 * 19>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %161 = "ckks.addcc"(%155, %160) : (tensor<1x!ckks.poly<2 * 51 * 18>>, tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %162 = "ckks.addcc"(%150, %161) : (tensor<1x!ckks.poly<2 * 51 * 18>>, tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %163 = "ckks.addcc"(%139, %162) : (tensor<1x!ckks.poly<2 * 51 * 18>>, tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %164 = "ckks.addcc"(%163, %163) : (tensor<1x!ckks.poly<2 * 51 * 18>>, tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %165 = "ckks.mulcc"(%164, %163) : (tensor<1x!ckks.poly<2 * 51 * 18>>, tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<3 * 102 * 18>>
    %166 = "ckks.relinearize"(%165) : (tensor<1x!ckks.poly<3 * 102 * 18>>) -> tensor<1x!ckks.poly<2 * 102 * 18>>
    %167 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 100.00000000000327 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 18>>
    %168 = "ckks.addcp"(%166, %167) : (tensor<1x!ckks.poly<2 * 102 * 18>>, tensor<1x!ckks.poly<1 * 102 * 18>>) -> tensor<1x!ckks.poly<2 * 102 * 18>>
    %169 = "ckks.rescalec"(%168) : (tensor<1x!ckks.poly<2 * 102 * 18>>) -> tensor<1x!ckks.poly<2 * 71 * 17>>
    %170 = "ckks.rescalec"(%169) : (tensor<1x!ckks.poly<2 * 71 * 17>>) -> tensor<1x!ckks.poly<2 * 41 * 16>>
    %171 = "ckks.negatec"(%163) : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 18>>
    %172 = "ckks.modswitchc"(%164) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 16>>
    %173 = "ckks.mulcc"(%172, %170) : (tensor<1x!ckks.poly<2 * 51 * 16>>, tensor<1x!ckks.poly<2 * 41 * 16>>) -> tensor<1x!ckks.poly<3 * 92 * 16>>
    %174 = "ckks.relinearize"(%173) : (tensor<1x!ckks.poly<3 * 92 * 16>>) -> tensor<1x!ckks.poly<2 * 92 * 16>>
    %175 = "ckks.modswitchc"(%171) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 16>>
    %176 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 40.053502206492468 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 16>>
    %177 = "ckks.mulcp"(%175, %176) : (tensor<1x!ckks.poly<2 * 51 * 16>>, tensor<1x!ckks.poly<1 * 41 * 16>>) -> tensor<1x!ckks.poly<2 * 92 * 16>>
    %178 = "ckks.addcc"(%174, %177) : (tensor<1x!ckks.poly<2 * 92 * 16>>, tensor<1x!ckks.poly<2 * 92 * 16>>) -> tensor<1x!ckks.poly<2 * 92 * 16>>
    %179 = "ckks.rescalec"(%178) : (tensor<1x!ckks.poly<2 * 92 * 16>>) -> tensor<1x!ckks.poly<2 * 61 * 15>>
    %180 = "ckks.addcc"(%170, %170) : (tensor<1x!ckks.poly<2 * 41 * 16>>, tensor<1x!ckks.poly<2 * 41 * 16>>) -> tensor<1x!ckks.poly<2 * 41 * 16>>
    %181 = "ckks.mulcc"(%180, %170) : (tensor<1x!ckks.poly<2 * 41 * 16>>, tensor<1x!ckks.poly<2 * 41 * 16>>) -> tensor<1x!ckks.poly<3 * 82 * 16>>
    %182 = "ckks.relinearize"(%181) : (tensor<1x!ckks.poly<3 * 82 * 16>>) -> tensor<1x!ckks.poly<2 * 82 * 16>>
    %183 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.107004412991472 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 16>>
    %184 = "ckks.addcp"(%182, %183) : (tensor<1x!ckks.poly<2 * 82 * 16>>, tensor<1x!ckks.poly<1 * 82 * 16>>) -> tensor<1x!ckks.poly<2 * 82 * 16>>
    %185 = "ckks.rescalec"(%184) : (tensor<1x!ckks.poly<2 * 82 * 16>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %186 = "ckks.modswitchc"(%164) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %187 = "ckks.mulcc"(%186, %185) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<3 * 102 * 15>>
    %188 = "ckks.relinearize"(%187) : (tensor<1x!ckks.poly<3 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 102 * 15>>
    %189 = "ckks.negatec"(%179) : (tensor<1x!ckks.poly<2 * 61 * 15>>) -> tensor<1x!ckks.poly<2 * 61 * 15>>
    %190 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 40.053502206492468 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 15>>
    %191 = "ckks.mulcp"(%189, %190) : (tensor<1x!ckks.poly<2 * 61 * 15>>, tensor<1x!ckks.poly<1 * 41 * 15>>) -> tensor<1x!ckks.poly<2 * 102 * 15>>
    %192 = "ckks.addcc"(%188, %191) : (tensor<1x!ckks.poly<2 * 102 * 15>>, tensor<1x!ckks.poly<2 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 102 * 15>>
    %193 = "ckks.rescalec"(%192) : (tensor<1x!ckks.poly<2 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 71 * 14>>
    %194 = "ckks.rescalec"(%193) : (tensor<1x!ckks.poly<2 * 71 * 14>>) -> tensor<1x!ckks.poly<2 * 41 * 13>>
    %195 = "ckks.addcc"(%179, %179) : (tensor<1x!ckks.poly<2 * 61 * 15>>, tensor<1x!ckks.poly<2 * 61 * 15>>) -> tensor<1x!ckks.poly<2 * 61 * 15>>
    %196 = "ckks.mulcc"(%195, %185) : (tensor<1x!ckks.poly<2 * 61 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<3 * 112 * 15>>
    %197 = "ckks.relinearize"(%196) : (tensor<1x!ckks.poly<3 * 112 * 15>>) -> tensor<1x!ckks.poly<2 * 112 * 15>>
    %198 = "ckks.modswitchc"(%171) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %199 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 60.218138071926617 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 15>>
    %200 = "ckks.mulcp"(%198, %199) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<1 * 61 * 15>>) -> tensor<1x!ckks.poly<2 * 112 * 15>>
    %201 = "ckks.addcc"(%197, %200) : (tensor<1x!ckks.poly<2 * 112 * 15>>, tensor<1x!ckks.poly<2 * 112 * 15>>) -> tensor<1x!ckks.poly<2 * 112 * 15>>
    %202 = "ckks.rescalec"(%201) : (tensor<1x!ckks.poly<2 * 112 * 15>>) -> tensor<1x!ckks.poly<2 * 81 * 14>>
    %203 = "ckks.rescalec"(%202) : (tensor<1x!ckks.poly<2 * 81 * 14>>) -> tensor<1x!ckks.poly<2 * 51 * 13>>
    %204 = "ckks.addcc"(%185, %185) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %205 = "ckks.mulcc"(%204, %185) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<3 * 102 * 15>>
    %206 = "ckks.relinearize"(%205) : (tensor<1x!ckks.poly<3 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 102 * 15>>
    %207 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 100.27164027843216 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 15>>
    %208 = "ckks.addcp"(%206, %207) : (tensor<1x!ckks.poly<2 * 102 * 15>>, tensor<1x!ckks.poly<1 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 102 * 15>>
    %209 = "ckks.rescalec"(%208) : (tensor<1x!ckks.poly<2 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 71 * 14>>
    %210 = "ckks.rescalec"(%209) : (tensor<1x!ckks.poly<2 * 71 * 14>>) -> tensor<1x!ckks.poly<2 * 41 * 13>>
    %211 = "ckks.addcc"(%210, %210) : (tensor<1x!ckks.poly<2 * 41 * 13>>, tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<2 * 41 * 13>>
    %212 = "ckks.mulcc"(%211, %210) : (tensor<1x!ckks.poly<2 * 41 * 13>>, tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<3 * 82 * 13>>
    %213 = "ckks.relinearize"(%212) : (tensor<1x!ckks.poly<3 * 82 * 13>>) -> tensor<1x!ckks.poly<2 * 82 * 13>>
    %214 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.670057866334616 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 13>>
    %215 = "ckks.addcp"(%213, %214) : (tensor<1x!ckks.poly<2 * 82 * 13>>, tensor<1x!ckks.poly<1 * 82 * 13>>) -> tensor<1x!ckks.poly<2 * 82 * 13>>
    %216 = "ckks.rescalec"(%215) : (tensor<1x!ckks.poly<2 * 82 * 13>>) -> tensor<1x!ckks.poly<2 * 51 * 12>>
    %217 = "ckks.modswitchc"(%arg0) <{downFactor = 18 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    %218 = "ckks.modswitchc"(%163) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %219 = "ckks.encode"() <{payload = dense<-0.017732734281918773> : tensor<32768xf64>}> {runtime.scale_log2 = 68.696025852976362 : f64} : () -> tensor<1x!ckks.poly<1 * 69 * 15>>
    %220 = "ckks.mulcp"(%218, %219) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<1 * 69 * 15>>) -> tensor<1x!ckks.poly<2 * 120 * 15>>
    %221 = "ckks.rescalec"(%220) : (tensor<1x!ckks.poly<2 * 120 * 15>>) -> tensor<1x!ckks.poly<2 * 89 * 14>>
    %222 = "ckks.rescalec"(%221) : (tensor<1x!ckks.poly<2 * 89 * 14>>) -> tensor<1x!ckks.poly<2 * 59 * 13>>
    %223 = "ckks.encode"() <{payload = dense<-0.0090840943441134625> : tensor<32768xf64>}> {runtime.scale_log2 = 58.613707920259287 : f64} : () -> tensor<1x!ckks.poly<1 * 59 * 15>>
    %224 = "ckks.mulcp"(%179, %223) : (tensor<1x!ckks.poly<2 * 61 * 15>>, tensor<1x!ckks.poly<1 * 59 * 15>>) -> tensor<1x!ckks.poly<2 * 120 * 15>>
    %225 = "ckks.rescalec"(%224) : (tensor<1x!ckks.poly<2 * 120 * 15>>) -> tensor<1x!ckks.poly<2 * 89 * 14>>
    %226 = "ckks.rescalec"(%225) : (tensor<1x!ckks.poly<2 * 89 * 14>>) -> tensor<1x!ckks.poly<2 * 59 * 13>>
    %227 = "ckks.addcc"(%222, %226) : (tensor<1x!ckks.poly<2 * 59 * 13>>, tensor<1x!ckks.poly<2 * 59 * 13>>) -> tensor<1x!ckks.poly<2 * 59 * 13>>
    %228 = "ckks.mulcc"(%227, %210) : (tensor<1x!ckks.poly<2 * 59 * 13>>, tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<3 * 100 * 13>>
    %229 = "ckks.relinearize"(%228) : (tensor<1x!ckks.poly<3 * 100 * 13>>) -> tensor<1x!ckks.poly<2 * 100 * 13>>
    %230 = "ckks.rescalec"(%229) : (tensor<1x!ckks.poly<2 * 100 * 13>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %231 = "ckks.modswitchc"(%163) <{downFactor = 5 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 13>>
    %232 = "ckks.encode"() <{payload = dense<-0.19732584204806608> : tensor<32768xf64>}> {runtime.scale_log2 = 49.094443440865803 : f64} : () -> tensor<1x!ckks.poly<1 * 50 * 13>>
    %233 = "ckks.mulcp"(%231, %232) : (tensor<1x!ckks.poly<2 * 51 * 13>>, tensor<1x!ckks.poly<1 * 50 * 13>>) -> tensor<1x!ckks.poly<2 * 101 * 13>>
    %234 = "ckks.rescalec"(%233) : (tensor<1x!ckks.poly<2 * 101 * 13>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %235 = "ckks.modswitchc"(%179) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 15>>) -> tensor<1x!ckks.poly<2 * 61 * 14>>
    %236 = "ckks.encode"() <{payload = dense<-0.1411166195583439> : tensor<32768xf64>}> {runtime.scale_log2 = 68.978630961429644 : f64} : () -> tensor<1x!ckks.poly<1 * 69 * 14>>
    %237 = "ckks.mulcp"(%235, %236) : (tensor<1x!ckks.poly<2 * 61 * 14>>, tensor<1x!ckks.poly<1 * 69 * 14>>) -> tensor<1x!ckks.poly<2 * 130 * 14>>
    %238 = "ckks.rescalec"(%237) : (tensor<1x!ckks.poly<2 * 130 * 14>>) -> tensor<1x!ckks.poly<2 * 100 * 13>>
    %239 = "ckks.rescalec"(%238) : (tensor<1x!ckks.poly<2 * 100 * 13>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %240 = "ckks.addcc"(%234, %239) : (tensor<1x!ckks.poly<2 * 70 * 12>>, tensor<1x!ckks.poly<2 * 70 * 12>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %241 = "ckks.encode"() <{payload = dense<-0.052675736911616873> : tensor<32768xf64>}> {runtime.scale_log2 = 58.895234646921097 : f64} : () -> tensor<1x!ckks.poly<1 * 59 * 13>>
    %242 = "ckks.mulcp"(%194, %241) : (tensor<1x!ckks.poly<2 * 41 * 13>>, tensor<1x!ckks.poly<1 * 59 * 13>>) -> tensor<1x!ckks.poly<2 * 100 * 13>>
    %243 = "ckks.rescalec"(%242) : (tensor<1x!ckks.poly<2 * 100 * 13>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %244 = "ckks.addcc"(%240, %243) : (tensor<1x!ckks.poly<2 * 70 * 12>>, tensor<1x!ckks.poly<2 * 70 * 12>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %245 = "ckks.encode"() <{payload = dense<-0.026312988130283786> : tensor<32768xf64>}> {runtime.scale_log2 = 48.812916714204022 : f64} : () -> tensor<1x!ckks.poly<1 * 49 * 13>>
    %246 = "ckks.mulcp"(%203, %245) : (tensor<1x!ckks.poly<2 * 51 * 13>>, tensor<1x!ckks.poly<1 * 49 * 13>>) -> tensor<1x!ckks.poly<2 * 100 * 13>>
    %247 = "ckks.rescalec"(%246) : (tensor<1x!ckks.poly<2 * 100 * 13>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %248 = "ckks.addcc"(%244, %247) : (tensor<1x!ckks.poly<2 * 70 * 12>>, tensor<1x!ckks.poly<2 * 70 * 12>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %249 = "ckks.addcc"(%230, %248) : (tensor<1x!ckks.poly<2 * 70 * 12>>, tensor<1x!ckks.poly<2 * 70 * 12>>) -> tensor<1x!ckks.poly<2 * 70 * 12>>
    %250 = "ckks.mulcc"(%249, %216) : (tensor<1x!ckks.poly<2 * 70 * 12>>, tensor<1x!ckks.poly<2 * 51 * 12>>) -> tensor<1x!ckks.poly<3 * 121 * 12>>
    %251 = "ckks.relinearize"(%250) : (tensor<1x!ckks.poly<3 * 121 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %252 = "ckks.encode"() <{payload = dense<-0.78912296503546786> : tensor<32768xf64>}> {runtime.scale_log2 = 59.463507762372274 : f64} : () -> tensor<1x!ckks.poly<1 * 60 * 13>>
    %253 = "ckks.mulcp"(%231, %252) : (tensor<1x!ckks.poly<2 * 51 * 13>>, tensor<1x!ckks.poly<1 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 111 * 13>>
    %254 = "ckks.modswitchc"(%179) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 15>>) -> tensor<1x!ckks.poly<2 * 61 * 13>>
    %255 = "ckks.encode"() <{payload = dense<-0.72741137353811158> : tensor<32768xf64>}> {runtime.scale_log2 = 49.381189829655199 : f64} : () -> tensor<1x!ckks.poly<1 * 50 * 13>>
    %256 = "ckks.mulcp"(%254, %255) : (tensor<1x!ckks.poly<2 * 61 * 13>>, tensor<1x!ckks.poly<1 * 50 * 13>>) -> tensor<1x!ckks.poly<2 * 111 * 13>>
    %257 = "ckks.addcc"(%253, %256) : (tensor<1x!ckks.poly<2 * 111 * 13>>, tensor<1x!ckks.poly<2 * 111 * 13>>) -> tensor<1x!ckks.poly<2 * 111 * 13>>
    %258 = "ckks.encode"() <{payload = dense<-0.32554529142034655> : tensor<32768xf64>}> {runtime.scale_log2 = 69.264298968427568 : f64} : () -> tensor<1x!ckks.poly<1 * 70 * 13>>
    %259 = "ckks.mulcp"(%194, %258) : (tensor<1x!ckks.poly<2 * 41 * 13>>, tensor<1x!ckks.poly<1 * 70 * 13>>) -> tensor<1x!ckks.poly<2 * 111 * 13>>
    %260 = "ckks.addcc"(%257, %259) : (tensor<1x!ckks.poly<2 * 111 * 13>>, tensor<1x!ckks.poly<2 * 111 * 13>>) -> tensor<1x!ckks.poly<2 * 111 * 13>>
    %261 = "ckks.encode"() <{payload = dense<-0.18601751738208003> : tensor<32768xf64>}> {runtime.scale_log2 = 59.181981035710493 : f64} : () -> tensor<1x!ckks.poly<1 * 60 * 13>>
    %262 = "ckks.mulcp"(%203, %261) : (tensor<1x!ckks.poly<2 * 51 * 13>>, tensor<1x!ckks.poly<1 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 111 * 13>>
    %263 = "ckks.addcc"(%260, %262) : (tensor<1x!ckks.poly<2 * 111 * 13>>, tensor<1x!ckks.poly<2 * 111 * 13>>) -> tensor<1x!ckks.poly<2 * 111 * 13>>
    %264 = "ckks.mulcc"(%263, %210) : (tensor<1x!ckks.poly<2 * 111 * 13>>, tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<3 * 152 * 13>>
    %265 = "ckks.relinearize"(%264) : (tensor<1x!ckks.poly<3 * 152 * 13>>) -> tensor<1x!ckks.poly<2 * 152 * 13>>
    %266 = "ckks.rescalec"(%265) : (tensor<1x!ckks.poly<2 * 152 * 13>>) -> tensor<1x!ckks.poly<2 * 120 * 12>>
    %267 = "ckks.modswitchc"(%163) <{downFactor = 6 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 18>>) -> tensor<1x!ckks.poly<2 * 51 * 12>>
    %268 = "ckks.encode"() <{payload = dense<-0.042857894008498022> : tensor<32768xf64>}> {runtime.scale_log2 = 69.832572083878745 : f64} : () -> tensor<1x!ckks.poly<1 * 70 * 12>>
    %269 = "ckks.mulcp"(%267, %268) : (tensor<1x!ckks.poly<2 * 51 * 12>>, tensor<1x!ckks.poly<1 * 70 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %270 = "ckks.modswitchc"(%179) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 15>>) -> tensor<1x!ckks.poly<2 * 61 * 12>>
    %271 = "ckks.encode"() <{payload = dense<-0.7747926099066248> : tensor<32768xf64>}> {runtime.scale_log2 = 59.75025415116167 : f64} : () -> tensor<1x!ckks.poly<1 * 60 * 12>>
    %272 = "ckks.mulcp"(%270, %271) : (tensor<1x!ckks.poly<2 * 61 * 12>>, tensor<1x!ckks.poly<1 * 60 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %273 = "ckks.addcc"(%269, %272) : (tensor<1x!ckks.poly<2 * 121 * 12>>, tensor<1x!ckks.poly<2 * 121 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %274 = "ckks.modswitchc"(%194) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<2 * 41 * 12>>
    %275 = "ckks.encode"() <{payload = dense<-0.18128174370012556> : tensor<32768xf64>}> {runtime.scale_log2 = 79.633363289934039 : f64} : () -> tensor<1x!ckks.poly<1 * 80 * 12>>
    %276 = "ckks.mulcp"(%274, %275) : (tensor<1x!ckks.poly<2 * 41 * 12>>, tensor<1x!ckks.poly<1 * 80 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %277 = "ckks.addcc"(%273, %276) : (tensor<1x!ckks.poly<2 * 121 * 12>>, tensor<1x!ckks.poly<2 * 121 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %278 = "ckks.modswitchc"(%203) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 13>>) -> tensor<1x!ckks.poly<2 * 51 * 12>>
    %279 = "ckks.encode"() <{payload = dense<-0.24546846913474671> : tensor<32768xf64>}> {runtime.scale_log2 = 69.551045357216964 : f64} : () -> tensor<1x!ckks.poly<1 * 70 * 12>>
    %280 = "ckks.mulcp"(%278, %279) : (tensor<1x!ckks.poly<2 * 51 * 12>>, tensor<1x!ckks.poly<1 * 70 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %281 = "ckks.addcc"(%277, %280) : (tensor<1x!ckks.poly<2 * 121 * 12>>, tensor<1x!ckks.poly<2 * 121 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %282 = "ckks.addcc"(%266, %281) : (tensor<1x!ckks.poly<2 * 120 * 12>>, tensor<1x!ckks.poly<2 * 121 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %283 = "ckks.addcc"(%251, %282) : (tensor<1x!ckks.poly<2 * 121 * 12>>, tensor<1x!ckks.poly<2 * 121 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %284 = "ckks.encode"() <{payload = dense<5.000000e-01> : tensor<32768xf64>}> {runtime.scale_log2 = 119.83257208392283 : f64} : () -> tensor<1x!ckks.poly<1 * 121 * 12>>
    %285 = "ckks.addcp"(%283, %284) : (tensor<1x!ckks.poly<2 * 121 * 12>>, tensor<1x!ckks.poly<1 * 121 * 12>>) -> tensor<1x!ckks.poly<2 * 121 * 12>>
    %286 = "ckks.mulcc"(%217, %285) : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<2 * 121 * 12>>) -> tensor<1x!ckks.poly<3 * 161 * 12>>
    %287 = "ckks.relinearize"(%286) : (tensor<1x!ckks.poly<3 * 161 * 12>>) -> tensor<1x!ckks.poly<2 * 161 * 12>>
    %288 = "ckks.rescalec"(%287) : (tensor<1x!ckks.poly<2 * 161 * 12>>) -> tensor<1x!ckks.poly<2 * 130 * 11>>
    %289 = "ckks.rescalec"(%288) : (tensor<1x!ckks.poly<2 * 130 * 11>>) -> tensor<1x!ckks.poly<2 * 100 * 10>>
    %290 = "ckks.rescalec"(%289) : (tensor<1x!ckks.poly<2 * 100 * 10>>) -> tensor<1x!ckks.poly<2 * 70 * 9>>
    %291 = "ckks.rescalec"(%290) : (tensor<1x!ckks.poly<2 * 70 * 9>>) -> tensor<1x!ckks.poly<2 * 41 * 8>>
    return %291 : tensor<1x!ckks.poly<2 * 41 * 8>>
  }
}
