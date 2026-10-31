module {
  func.func @one_relu(%arg0: tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 41 * 5>> attributes {absorb_coefficient_scale, arg_scale = array<i64: 40>, earth.arg_scales = array<f64: 4.000000e+01>, earth.native_level_anchor = 30 : i64, earth.native_output_level = 30 : i64, earth.native_scale_moduli = array<i64: 4255252481, 4255645697, 4258922497, 4260364289, 4261675009, 4264427521, 4265476097, 4266393601, 4267442177, 1040842753, 1042415617, 1043464193, 1045430273, 1048707073, 1049100289, 1051721729, 1052508161, 1053818881, 1054212097, 1055260673, 1056178177, 1056440321, 1060765697, 1062469633, 1062862849, 1064697857, 1065484289, 1068236801, 1070727169, 1071513601, 1073479681, 2135162881, 4272291841, 4272685057, 1040056321, 2135818241, 4273340417, 2142502913, 4274126849, 4276092929, 2144468993, 2146041857, 2146959361, 4277403649, 4279369729, 2147352577, 4280025089, 4280156161, 4281204737, 1038745601>, earth.region_scale_policy, earth.regional_boot_count = 0 : i64, earth.regional_queries = 58818 : i64, earth.regional_regions = 22 : i64, earth.result_scales = array<f64: 40.000000000049127>, earth.scale_waterline = 4.000000e+01 : f64, init_level = 30 : i64, res_scale = array<i64: 41>} {
    %0 = "ckks.addcc"(%arg0, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    %1 = "ckks.mulcc"(%0, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<3 * 80 * 27>>
    %2 = "ckks.relinearize"(%1) : (tensor<1x!ckks.poly<3 * 80 * 27>>) -> tensor<1x!ckks.poly<2 * 80 * 27>>
    %3 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 8.000000e+01 : f64} : () -> tensor<1x!ckks.poly<1 * 80 * 27>>
    %4 = "ckks.addcp"(%2, %3) : (tensor<1x!ckks.poly<2 * 80 * 27>>, tensor<1x!ckks.poly<1 * 80 * 27>>) -> tensor<1x!ckks.poly<2 * 80 * 27>>
    %5 = "ckks.rescalec"(%4) : (tensor<1x!ckks.poly<2 * 80 * 27>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %6 = "ckks.modswitchc"(%0) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 26>>
    %7 = "ckks.mulcc"(%6, %5) : (tensor<1x!ckks.poly<2 * 40 * 26>>, tensor<1x!ckks.poly<2 * 51 * 26>>) -> tensor<1x!ckks.poly<3 * 91 * 26>>
    %8 = "ckks.relinearize"(%7) : (tensor<1x!ckks.poly<3 * 91 * 26>>) -> tensor<1x!ckks.poly<2 * 91 * 26>>
    %9 = "ckks.negatec"(%arg0) : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    %10 = "ckks.modswitchc"(%9) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 26>>
    %11 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 50.007415654643665 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 26>>
    %12 = "ckks.mulcp"(%10, %11) : (tensor<1x!ckks.poly<2 * 40 * 26>>, tensor<1x!ckks.poly<1 * 51 * 26>>) -> tensor<1x!ckks.poly<2 * 91 * 26>>
    %13 = "ckks.addcc"(%8, %12) : (tensor<1x!ckks.poly<2 * 91 * 26>>, tensor<1x!ckks.poly<2 * 91 * 26>>) -> tensor<1x!ckks.poly<2 * 91 * 26>>
    %14 = "ckks.rescalec"(%13) : (tensor<1x!ckks.poly<2 * 91 * 26>>) -> tensor<1x!ckks.poly<2 * 61 * 25>>
    %15 = "ckks.addcc"(%5, %5) : (tensor<1x!ckks.poly<2 * 51 * 26>>, tensor<1x!ckks.poly<2 * 51 * 26>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %16 = "ckks.mulcc"(%15, %5) : (tensor<1x!ckks.poly<2 * 51 * 26>>, tensor<1x!ckks.poly<2 * 51 * 26>>) -> tensor<1x!ckks.poly<3 * 102 * 26>>
    %17 = "ckks.relinearize"(%16) : (tensor<1x!ckks.poly<3 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 102 * 26>>
    %18 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 100.01483130928732 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 26>>
    %19 = "ckks.addcp"(%17, %18) : (tensor<1x!ckks.poly<2 * 102 * 26>>, tensor<1x!ckks.poly<1 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 102 * 26>>
    %20 = "ckks.rescalec"(%19) : (tensor<1x!ckks.poly<2 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 71 * 25>>
    %21 = "ckks.rescalec"(%20) : (tensor<1x!ckks.poly<2 * 71 * 25>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %22 = "ckks.addcc"(%21, %21) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %23 = "ckks.mulcc"(%22, %21) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<3 * 82 * 24>>
    %24 = "ckks.relinearize"(%23) : (tensor<1x!ckks.poly<3 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 82 * 24>>
    %25 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.076344409504543 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 24>>
    %26 = "ckks.addcp"(%24, %25) : (tensor<1x!ckks.poly<2 * 82 * 24>>, tensor<1x!ckks.poly<1 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 82 * 24>>
    %27 = "ckks.rescalec"(%26) : (tensor<1x!ckks.poly<2 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %28 = "ckks.encode"() <{payload = dense<-4.0270201200636233> : tensor<32768xf64>}> {runtime.scale_log2 = 40.085161121109593 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 27>>
    %29 = "ckks.mulcp"(%arg0, %28) : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 41 * 27>>) -> tensor<1x!ckks.poly<2 * 81 * 27>>
    %30 = "ckks.rescalec"(%29) : (tensor<1x!ckks.poly<2 * 81 * 27>>) -> tensor<1x!ckks.poly<2 * 51 * 26>>
    %31 = "ckks.mulcc"(%30, %5) : (tensor<1x!ckks.poly<2 * 51 * 26>>, tensor<1x!ckks.poly<2 * 51 * 26>>) -> tensor<1x!ckks.poly<3 * 102 * 26>>
    %32 = "ckks.relinearize"(%31) : (tensor<1x!ckks.poly<3 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 102 * 26>>
    %33 = "ckks.rescalec"(%32) : (tensor<1x!ckks.poly<2 * 102 * 26>>) -> tensor<1x!ckks.poly<2 * 71 * 25>>
    %34 = "ckks.rescalec"(%33) : (tensor<1x!ckks.poly<2 * 71 * 25>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %35 = "ckks.modswitchc"(%arg0) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 26>>
    %36 = "ckks.encode"() <{payload = dense<2.243187978526648> : tensor<32768xf64>}> {runtime.scale_log2 = 60.09999243039691 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 26>>
    %37 = "ckks.mulcp"(%35, %36) : (tensor<1x!ckks.poly<2 * 40 * 26>>, tensor<1x!ckks.poly<1 * 61 * 26>>) -> tensor<1x!ckks.poly<2 * 101 * 26>>
    %38 = "ckks.rescalec"(%37) : (tensor<1x!ckks.poly<2 * 101 * 26>>) -> tensor<1x!ckks.poly<2 * 71 * 25>>
    %39 = "ckks.rescalec"(%38) : (tensor<1x!ckks.poly<2 * 71 * 25>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %40 = "ckks.addcc"(%34, %39) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 41 * 24>>
    %41 = "ckks.mulcc"(%40, %21) : (tensor<1x!ckks.poly<2 * 41 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<3 * 82 * 24>>
    %42 = "ckks.relinearize"(%41) : (tensor<1x!ckks.poly<3 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 82 * 24>>
    %43 = "ckks.rescalec"(%42) : (tensor<1x!ckks.poly<2 * 82 * 24>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %44 = "ckks.modswitchc"(%arg0) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 24>>
    %45 = "ckks.encode"() <{payload = dense<1.1583708205212988> : tensor<32768xf64>}> {runtime.scale_log2 = 40.161505530614093 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 24>>
    %46 = "ckks.mulcp"(%44, %45) : (tensor<1x!ckks.poly<2 * 40 * 24>>, tensor<1x!ckks.poly<1 * 41 * 24>>) -> tensor<1x!ckks.poly<2 * 81 * 24>>
    %47 = "ckks.rescalec"(%46) : (tensor<1x!ckks.poly<2 * 81 * 24>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %48 = "ckks.encode"() <{payload = dense<-0.24378614014677616> : tensor<32768xf64>}> {runtime.scale_log2 = 50.130748980505487 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 25>>
    %49 = "ckks.mulcp"(%14, %48) : (tensor<1x!ckks.poly<2 * 61 * 25>>, tensor<1x!ckks.poly<1 * 51 * 25>>) -> tensor<1x!ckks.poly<2 * 112 * 25>>
    %50 = "ckks.rescalec"(%49) : (tensor<1x!ckks.poly<2 * 112 * 25>>) -> tensor<1x!ckks.poly<2 * 81 * 24>>
    %51 = "ckks.rescalec"(%50) : (tensor<1x!ckks.poly<2 * 81 * 24>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %52 = "ckks.addcc"(%47, %51) : (tensor<1x!ckks.poly<2 * 51 * 23>>, tensor<1x!ckks.poly<2 * 51 * 23>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %53 = "ckks.addcc"(%43, %52) : (tensor<1x!ckks.poly<2 * 51 * 23>>, tensor<1x!ckks.poly<2 * 51 * 23>>) -> tensor<1x!ckks.poly<2 * 51 * 23>>
    %54 = "ckks.mulcc"(%53, %27) : (tensor<1x!ckks.poly<2 * 51 * 23>>, tensor<1x!ckks.poly<2 * 51 * 23>>) -> tensor<1x!ckks.poly<3 * 102 * 23>>
    %55 = "ckks.relinearize"(%54) : (tensor<1x!ckks.poly<3 * 102 * 23>>) -> tensor<1x!ckks.poly<2 * 102 * 23>>
    %56 = "ckks.rescalec"(%55) : (tensor<1x!ckks.poly<2 * 102 * 23>>) -> tensor<1x!ckks.poly<2 * 71 * 22>>
    %57 = "ckks.rescalec"(%56) : (tensor<1x!ckks.poly<2 * 71 * 22>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %58 = "ckks.encode"() <{payload = dense<0.38878894178135931> : tensor<32768xf64>}> {runtime.scale_log2 = 50.214369436753117 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 24>>
    %59 = "ckks.mulcp"(%44, %58) : (tensor<1x!ckks.poly<2 * 40 * 24>>, tensor<1x!ckks.poly<1 * 51 * 24>>) -> tensor<1x!ckks.poly<2 * 91 * 24>>
    %60 = "ckks.encode"() <{payload = dense<-0.34103182272202132> : tensor<32768xf64>}> {runtime.scale_log2 = 60.183612886644497 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 25>>
    %61 = "ckks.mulcp"(%14, %60) : (tensor<1x!ckks.poly<2 * 61 * 25>>, tensor<1x!ckks.poly<1 * 61 * 25>>) -> tensor<1x!ckks.poly<2 * 122 * 25>>
    %62 = "ckks.rescalec"(%61) : (tensor<1x!ckks.poly<2 * 122 * 25>>) -> tensor<1x!ckks.poly<2 * 91 * 24>>
    %63 = "ckks.addcc"(%59, %62) : (tensor<1x!ckks.poly<2 * 91 * 24>>, tensor<1x!ckks.poly<2 * 91 * 24>>) -> tensor<1x!ckks.poly<2 * 91 * 24>>
    %64 = "ckks.mulcc"(%63, %21) : (tensor<1x!ckks.poly<2 * 91 * 24>>, tensor<1x!ckks.poly<2 * 41 * 24>>) -> tensor<1x!ckks.poly<3 * 132 * 24>>
    %65 = "ckks.relinearize"(%64) : (tensor<1x!ckks.poly<3 * 132 * 24>>) -> tensor<1x!ckks.poly<2 * 132 * 24>>
    %66 = "ckks.rescalec"(%65) : (tensor<1x!ckks.poly<2 * 132 * 24>>) -> tensor<1x!ckks.poly<2 * 101 * 23>>
    %67 = "ckks.rescalec"(%66) : (tensor<1x!ckks.poly<2 * 101 * 23>>) -> tensor<1x!ckks.poly<2 * 71 * 22>>
    %68 = "ckks.rescalec"(%67) : (tensor<1x!ckks.poly<2 * 71 * 22>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %69 = "ckks.modswitchc"(%arg0) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 23>>
    %70 = "ckks.encode"() <{payload = dense<1.3110156470890746> : tensor<32768xf64>}> {runtime.scale_log2 = 60.267233342892197 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 23>>
    %71 = "ckks.mulcp"(%69, %70) : (tensor<1x!ckks.poly<2 * 40 * 23>>, tensor<1x!ckks.poly<1 * 61 * 23>>) -> tensor<1x!ckks.poly<2 * 101 * 23>>
    %72 = "ckks.rescalec"(%71) : (tensor<1x!ckks.poly<2 * 101 * 23>>) -> tensor<1x!ckks.poly<2 * 71 * 22>>
    %73 = "ckks.rescalec"(%72) : (tensor<1x!ckks.poly<2 * 71 * 22>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %74 = "ckks.modswitchc"(%14) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 25>>) -> tensor<1x!ckks.poly<2 * 61 * 23>>
    %75 = "ckks.encode"() <{payload = dense<-0.46554445822061141> : tensor<32768xf64>}> {runtime.scale_log2 = 40.248679862386012 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 23>>
    %76 = "ckks.mulcp"(%74, %75) : (tensor<1x!ckks.poly<2 * 61 * 23>>, tensor<1x!ckks.poly<1 * 41 * 23>>) -> tensor<1x!ckks.poly<2 * 102 * 23>>
    %77 = "ckks.rescalec"(%76) : (tensor<1x!ckks.poly<2 * 102 * 23>>) -> tensor<1x!ckks.poly<2 * 71 * 22>>
    %78 = "ckks.rescalec"(%77) : (tensor<1x!ckks.poly<2 * 71 * 22>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %79 = "ckks.addcc"(%73, %78) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %80 = "ckks.addcc"(%68, %79) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %81 = "ckks.addcc"(%57, %80) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %82 = "ckks.addcc"(%81, %81) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %83 = "ckks.mulcc"(%82, %81) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<3 * 82 * 21>>
    %84 = "ckks.relinearize"(%83) : (tensor<1x!ckks.poly<3 * 82 * 21>>) -> tensor<1x!ckks.poly<2 * 82 * 21>>
    %85 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.60000000000025 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 21>>
    %86 = "ckks.addcp"(%84, %85) : (tensor<1x!ckks.poly<2 * 82 * 21>>, tensor<1x!ckks.poly<1 * 82 * 21>>) -> tensor<1x!ckks.poly<2 * 82 * 21>>
    %87 = "ckks.rescalec"(%86) : (tensor<1x!ckks.poly<2 * 82 * 21>>) -> tensor<1x!ckks.poly<2 * 51 * 20>>
    %88 = "ckks.modswitchc"(%82) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 20>>
    %89 = "ckks.mulcc"(%88, %87) : (tensor<1x!ckks.poly<2 * 41 * 20>>, tensor<1x!ckks.poly<2 * 51 * 20>>) -> tensor<1x!ckks.poly<3 * 92 * 20>>
    %90 = "ckks.relinearize"(%89) : (tensor<1x!ckks.poly<3 * 92 * 20>>) -> tensor<1x!ckks.poly<2 * 92 * 20>>
    %91 = "ckks.negatec"(%81) : (tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 21>>
    %92 = "ckks.modswitchc"(%91) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 20>>
    %93 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 50.623435875219045 : f64} : () -> tensor<1x!ckks.poly<1 * 51 * 20>>
    %94 = "ckks.mulcp"(%92, %93) : (tensor<1x!ckks.poly<2 * 41 * 20>>, tensor<1x!ckks.poly<1 * 51 * 20>>) -> tensor<1x!ckks.poly<2 * 92 * 20>>
    %95 = "ckks.addcc"(%90, %94) : (tensor<1x!ckks.poly<2 * 92 * 20>>, tensor<1x!ckks.poly<2 * 92 * 20>>) -> tensor<1x!ckks.poly<2 * 92 * 20>>
    %96 = "ckks.rescalec"(%95) : (tensor<1x!ckks.poly<2 * 92 * 20>>) -> tensor<1x!ckks.poly<2 * 61 * 19>>
    %97 = "ckks.addcc"(%87, %87) : (tensor<1x!ckks.poly<2 * 51 * 20>>, tensor<1x!ckks.poly<2 * 51 * 20>>) -> tensor<1x!ckks.poly<2 * 51 * 20>>
    %98 = "ckks.mulcc"(%97, %87) : (tensor<1x!ckks.poly<2 * 51 * 20>>, tensor<1x!ckks.poly<2 * 51 * 20>>) -> tensor<1x!ckks.poly<3 * 102 * 20>>
    %99 = "ckks.relinearize"(%98) : (tensor<1x!ckks.poly<3 * 102 * 20>>) -> tensor<1x!ckks.poly<2 * 102 * 20>>
    %100 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 101.2468717504386 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 20>>
    %101 = "ckks.addcp"(%99, %100) : (tensor<1x!ckks.poly<2 * 102 * 20>>, tensor<1x!ckks.poly<1 * 102 * 20>>) -> tensor<1x!ckks.poly<2 * 102 * 20>>
    %102 = "ckks.rescalec"(%101) : (tensor<1x!ckks.poly<2 * 102 * 20>>) -> tensor<1x!ckks.poly<2 * 72 * 19>>
    %103 = "ckks.rescalec"(%102) : (tensor<1x!ckks.poly<2 * 72 * 19>>) -> tensor<1x!ckks.poly<2 * 42 * 18>>
    %104 = "ckks.addcc"(%103, %103) : (tensor<1x!ckks.poly<2 * 42 * 18>>, tensor<1x!ckks.poly<2 * 42 * 18>>) -> tensor<1x!ckks.poly<2 * 42 * 18>>
    %105 = "ckks.mulcc"(%104, %103) : (tensor<1x!ckks.poly<2 * 42 * 18>>, tensor<1x!ckks.poly<2 * 42 * 18>>) -> tensor<1x!ckks.poly<3 * 84 * 18>>
    %106 = "ckks.relinearize"(%105) : (tensor<1x!ckks.poly<3 * 84 * 18>>) -> tensor<1x!ckks.poly<2 * 84 * 18>>
    %107 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 82.591426768045693 : f64} : () -> tensor<1x!ckks.poly<1 * 84 * 18>>
    %108 = "ckks.addcp"(%106, %107) : (tensor<1x!ckks.poly<2 * 84 * 18>>, tensor<1x!ckks.poly<1 * 84 * 18>>) -> tensor<1x!ckks.poly<2 * 84 * 18>>
    %109 = "ckks.rescalec"(%108) : (tensor<1x!ckks.poly<2 * 84 * 18>>) -> tensor<1x!ckks.poly<2 * 53 * 17>>
    %110 = "ckks.encode"() <{payload = dense<-4.1144251614039149> : tensor<32768xf64>}> {runtime.scale_log2 = 45.008346537265709 : f64} : () -> tensor<1x!ckks.poly<1 * 46 * 21>>
    %111 = "ckks.mulcp"(%81, %110) : (tensor<1x!ckks.poly<2 * 41 * 21>>, tensor<1x!ckks.poly<1 * 46 * 21>>) -> tensor<1x!ckks.poly<2 * 87 * 21>>
    %112 = "ckks.rescalec"(%111) : (tensor<1x!ckks.poly<2 * 87 * 21>>) -> tensor<1x!ckks.poly<2 * 56 * 20>>
    %113 = "ckks.mulcc"(%112, %87) : (tensor<1x!ckks.poly<2 * 56 * 20>>, tensor<1x!ckks.poly<2 * 51 * 20>>) -> tensor<1x!ckks.poly<3 * 107 * 20>>
    %114 = "ckks.relinearize"(%113) : (tensor<1x!ckks.poly<3 * 107 * 20>>) -> tensor<1x!ckks.poly<2 * 107 * 20>>
    %115 = "ckks.rescalec"(%114) : (tensor<1x!ckks.poly<2 * 107 * 20>>) -> tensor<1x!ckks.poly<2 * 76 * 19>>
    %116 = "ckks.rescalec"(%115) : (tensor<1x!ckks.poly<2 * 76 * 19>>) -> tensor<1x!ckks.poly<2 * 47 * 18>>
    %117 = "ckks.modswitchc"(%81) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 20>>
    %118 = "ckks.encode"() <{payload = dense<0.93680785180227221> : tensor<32768xf64>}> {runtime.scale_log2 = 65.655218287703804 : f64} : () -> tensor<1x!ckks.poly<1 * 66 * 20>>
    %119 = "ckks.mulcp"(%117, %118) : (tensor<1x!ckks.poly<2 * 41 * 20>>, tensor<1x!ckks.poly<1 * 66 * 20>>) -> tensor<1x!ckks.poly<2 * 107 * 20>>
    %120 = "ckks.rescalec"(%119) : (tensor<1x!ckks.poly<2 * 107 * 20>>) -> tensor<1x!ckks.poly<2 * 76 * 19>>
    %121 = "ckks.rescalec"(%120) : (tensor<1x!ckks.poly<2 * 76 * 19>>) -> tensor<1x!ckks.poly<2 * 47 * 18>>
    %122 = "ckks.addcc"(%116, %121) : (tensor<1x!ckks.poly<2 * 47 * 18>>, tensor<1x!ckks.poly<2 * 47 * 18>>) -> tensor<1x!ckks.poly<2 * 47 * 18>>
    %123 = "ckks.mulcc"(%122, %103) : (tensor<1x!ckks.poly<2 * 47 * 18>>, tensor<1x!ckks.poly<2 * 42 * 18>>) -> tensor<1x!ckks.poly<3 * 89 * 18>>
    %124 = "ckks.relinearize"(%123) : (tensor<1x!ckks.poly<3 * 89 * 18>>) -> tensor<1x!ckks.poly<2 * 89 * 18>>
    %125 = "ckks.rescalec"(%124) : (tensor<1x!ckks.poly<2 * 89 * 18>>) -> tensor<1x!ckks.poly<2 * 58 * 17>>
    %126 = "ckks.modswitchc"(%81) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 18>>
    %127 = "ckks.encode"() <{payload = dense<0.13312678038183748> : tensor<32768xf64>}> {runtime.scale_log2 = 46.999773305310427 : f64} : () -> tensor<1x!ckks.poly<1 * 47 * 18>>
    %128 = "ckks.mulcp"(%126, %127) : (tensor<1x!ckks.poly<2 * 41 * 18>>, tensor<1x!ckks.poly<1 * 47 * 18>>) -> tensor<1x!ckks.poly<2 * 88 * 18>>
    %129 = "ckks.rescalec"(%128) : (tensor<1x!ckks.poly<2 * 88 * 18>>) -> tensor<1x!ckks.poly<2 * 58 * 17>>
    %130 = "ckks.encode"() <{payload = dense<-0.45988074662238737> : tensor<32768xf64>}> {runtime.scale_log2 = 56.327495796507137 : f64} : () -> tensor<1x!ckks.poly<1 * 57 * 19>>
    %131 = "ckks.mulcp"(%96, %130) : (tensor<1x!ckks.poly<2 * 61 * 19>>, tensor<1x!ckks.poly<1 * 57 * 19>>) -> tensor<1x!ckks.poly<2 * 118 * 19>>
    %132 = "ckks.rescalec"(%131) : (tensor<1x!ckks.poly<2 * 118 * 19>>) -> tensor<1x!ckks.poly<2 * 88 * 18>>
    %133 = "ckks.rescalec"(%132) : (tensor<1x!ckks.poly<2 * 88 * 18>>) -> tensor<1x!ckks.poly<2 * 58 * 17>>
    %134 = "ckks.addcc"(%129, %133) : (tensor<1x!ckks.poly<2 * 58 * 17>>, tensor<1x!ckks.poly<2 * 58 * 17>>) -> tensor<1x!ckks.poly<2 * 58 * 17>>
    %135 = "ckks.addcc"(%125, %134) : (tensor<1x!ckks.poly<2 * 58 * 17>>, tensor<1x!ckks.poly<2 * 58 * 17>>) -> tensor<1x!ckks.poly<2 * 58 * 17>>
    %136 = "ckks.mulcc"(%135, %109) : (tensor<1x!ckks.poly<2 * 58 * 17>>, tensor<1x!ckks.poly<2 * 53 * 17>>) -> tensor<1x!ckks.poly<3 * 111 * 17>>
    %137 = "ckks.relinearize"(%136) : (tensor<1x!ckks.poly<3 * 111 * 17>>) -> tensor<1x!ckks.poly<2 * 111 * 17>>
    %138 = "ckks.rescalec"(%137) : (tensor<1x!ckks.poly<2 * 111 * 17>>) -> tensor<1x!ckks.poly<2 * 80 * 16>>
    %139 = "ckks.rescalec"(%138) : (tensor<1x!ckks.poly<2 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %140 = "ckks.encode"() <{payload = dense<-0.058402193560527074> : tensor<32768xf64>}> {runtime.scale_log2 = 58.321968683260565 : f64} : () -> tensor<1x!ckks.poly<1 * 59 * 18>>
    %141 = "ckks.mulcp"(%126, %140) : (tensor<1x!ckks.poly<2 * 41 * 18>>, tensor<1x!ckks.poly<1 * 59 * 18>>) -> tensor<1x!ckks.poly<2 * 100 * 18>>
    %142 = "ckks.encode"() <{payload = dense<-0.56048531218535647> : tensor<32768xf64>}> {runtime.scale_log2 = 67.649691174457288 : f64} : () -> tensor<1x!ckks.poly<1 * 68 * 19>>
    %143 = "ckks.mulcp"(%96, %142) : (tensor<1x!ckks.poly<2 * 61 * 19>>, tensor<1x!ckks.poly<1 * 68 * 19>>) -> tensor<1x!ckks.poly<2 * 129 * 19>>
    %144 = "ckks.rescalec"(%143) : (tensor<1x!ckks.poly<2 * 129 * 19>>) -> tensor<1x!ckks.poly<2 * 99 * 18>>
    %145 = "ckks.addcc"(%141, %144) : (tensor<1x!ckks.poly<2 * 100 * 18>>, tensor<1x!ckks.poly<2 * 99 * 18>>) -> tensor<1x!ckks.poly<2 * 100 * 18>>
    %146 = "ckks.mulcc"(%145, %103) : (tensor<1x!ckks.poly<2 * 100 * 18>>, tensor<1x!ckks.poly<2 * 42 * 18>>) -> tensor<1x!ckks.poly<3 * 142 * 18>>
    %147 = "ckks.relinearize"(%146) : (tensor<1x!ckks.poly<3 * 142 * 18>>) -> tensor<1x!ckks.poly<2 * 142 * 18>>
    %148 = "ckks.rescalec"(%147) : (tensor<1x!ckks.poly<2 * 142 * 18>>) -> tensor<1x!ckks.poly<2 * 110 * 17>>
    %149 = "ckks.rescalec"(%148) : (tensor<1x!ckks.poly<2 * 110 * 17>>) -> tensor<1x!ckks.poly<2 * 80 * 16>>
    %150 = "ckks.rescalec"(%149) : (tensor<1x!ckks.poly<2 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %151 = "ckks.modswitchc"(%81) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 21>>) -> tensor<1x!ckks.poly<2 * 41 * 17>>
    %152 = "ckks.encode"() <{payload = dense<0.77658111917095896> : tensor<32768xf64>}> {runtime.scale_log2 = 69.644164061210645 : f64} : () -> tensor<1x!ckks.poly<1 * 70 * 17>>
    %153 = "ckks.mulcp"(%151, %152) : (tensor<1x!ckks.poly<2 * 41 * 17>>, tensor<1x!ckks.poly<1 * 70 * 17>>) -> tensor<1x!ckks.poly<2 * 111 * 17>>
    %154 = "ckks.rescalec"(%153) : (tensor<1x!ckks.poly<2 * 111 * 17>>) -> tensor<1x!ckks.poly<2 * 80 * 16>>
    %155 = "ckks.rescalec"(%154) : (tensor<1x!ckks.poly<2 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %156 = "ckks.modswitchc"(%96) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 19>>) -> tensor<1x!ckks.poly<2 * 61 * 17>>
    %157 = "ckks.encode"() <{payload = dense<-0.68184097760015783> : tensor<32768xf64>}> {runtime.scale_log2 = 48.996934277506156 : f64} : () -> tensor<1x!ckks.poly<1 * 49 * 17>>
    %158 = "ckks.mulcp"(%156, %157) : (tensor<1x!ckks.poly<2 * 61 * 17>>, tensor<1x!ckks.poly<1 * 49 * 17>>) -> tensor<1x!ckks.poly<2 * 110 * 17>>
    %159 = "ckks.rescalec"(%158) : (tensor<1x!ckks.poly<2 * 110 * 17>>) -> tensor<1x!ckks.poly<2 * 80 * 16>>
    %160 = "ckks.rescalec"(%159) : (tensor<1x!ckks.poly<2 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %161 = "ckks.addcc"(%155, %160) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %162 = "ckks.addcc"(%150, %161) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %163 = "ckks.addcc"(%139, %162) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %164 = "ckks.addcc"(%163, %163) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %165 = "ckks.mulcc"(%164, %163) : (tensor<1x!ckks.poly<2 * 51 * 15>>, tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<3 * 102 * 15>>
    %166 = "ckks.relinearize"(%165) : (tensor<1x!ckks.poly<3 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 102 * 15>>
    %167 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 100.00000000000364 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 15>>
    %168 = "ckks.addcp"(%166, %167) : (tensor<1x!ckks.poly<2 * 102 * 15>>, tensor<1x!ckks.poly<1 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 102 * 15>>
    %169 = "ckks.rescalec"(%168) : (tensor<1x!ckks.poly<2 * 102 * 15>>) -> tensor<1x!ckks.poly<2 * 71 * 14>>
    %170 = "ckks.rescalec"(%169) : (tensor<1x!ckks.poly<2 * 71 * 14>>) -> tensor<1x!ckks.poly<2 * 41 * 13>>
    %171 = "ckks.negatec"(%163) : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 15>>
    %172 = "ckks.modswitchc"(%164) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 13>>
    %173 = "ckks.mulcc"(%172, %170) : (tensor<1x!ckks.poly<2 * 51 * 13>>, tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<3 * 92 * 13>>
    %174 = "ckks.relinearize"(%173) : (tensor<1x!ckks.poly<3 * 92 * 13>>) -> tensor<1x!ckks.poly<2 * 92 * 13>>
    %175 = "ckks.modswitchc"(%171) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 13>>
    %176 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 40.063388654735149 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 13>>
    %177 = "ckks.mulcp"(%175, %176) : (tensor<1x!ckks.poly<2 * 51 * 13>>, tensor<1x!ckks.poly<1 * 41 * 13>>) -> tensor<1x!ckks.poly<2 * 92 * 13>>
    %178 = "ckks.addcc"(%174, %177) : (tensor<1x!ckks.poly<2 * 92 * 13>>, tensor<1x!ckks.poly<2 * 92 * 13>>) -> tensor<1x!ckks.poly<2 * 92 * 13>>
    %179 = "ckks.rescalec"(%178) : (tensor<1x!ckks.poly<2 * 92 * 13>>) -> tensor<1x!ckks.poly<2 * 61 * 12>>
    %180 = "ckks.addcc"(%170, %170) : (tensor<1x!ckks.poly<2 * 41 * 13>>, tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<2 * 41 * 13>>
    %181 = "ckks.mulcc"(%180, %170) : (tensor<1x!ckks.poly<2 * 41 * 13>>, tensor<1x!ckks.poly<2 * 41 * 13>>) -> tensor<1x!ckks.poly<3 * 82 * 13>>
    %182 = "ckks.relinearize"(%181) : (tensor<1x!ckks.poly<3 * 82 * 13>>) -> tensor<1x!ckks.poly<2 * 82 * 13>>
    %183 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.126777309477575 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 13>>
    %184 = "ckks.addcp"(%182, %183) : (tensor<1x!ckks.poly<2 * 82 * 13>>, tensor<1x!ckks.poly<1 * 82 * 13>>) -> tensor<1x!ckks.poly<2 * 82 * 13>>
    %185 = "ckks.rescalec"(%184) : (tensor<1x!ckks.poly<2 * 82 * 13>>) -> tensor<1x!ckks.poly<2 * 51 * 12>>
    %186 = "ckks.modswitchc"(%164) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 12>>
    %187 = "ckks.mulcc"(%186, %185) : (tensor<1x!ckks.poly<2 * 51 * 12>>, tensor<1x!ckks.poly<2 * 51 * 12>>) -> tensor<1x!ckks.poly<3 * 102 * 12>>
    %188 = "ckks.relinearize"(%187) : (tensor<1x!ckks.poly<3 * 102 * 12>>) -> tensor<1x!ckks.poly<2 * 102 * 12>>
    %189 = "ckks.negatec"(%179) : (tensor<1x!ckks.poly<2 * 61 * 12>>) -> tensor<1x!ckks.poly<2 * 61 * 12>>
    %190 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 40.063388654735157 : f64} : () -> tensor<1x!ckks.poly<1 * 41 * 12>>
    %191 = "ckks.mulcp"(%189, %190) : (tensor<1x!ckks.poly<2 * 61 * 12>>, tensor<1x!ckks.poly<1 * 41 * 12>>) -> tensor<1x!ckks.poly<2 * 102 * 12>>
    %192 = "ckks.addcc"(%188, %191) : (tensor<1x!ckks.poly<2 * 102 * 12>>, tensor<1x!ckks.poly<2 * 102 * 12>>) -> tensor<1x!ckks.poly<2 * 102 * 12>>
    %193 = "ckks.rescalec"(%192) : (tensor<1x!ckks.poly<2 * 102 * 12>>) -> tensor<1x!ckks.poly<2 * 71 * 11>>
    %194 = "ckks.rescalec"(%193) : (tensor<1x!ckks.poly<2 * 71 * 11>>) -> tensor<1x!ckks.poly<2 * 41 * 10>>
    %195 = "ckks.addcc"(%179, %179) : (tensor<1x!ckks.poly<2 * 61 * 12>>, tensor<1x!ckks.poly<2 * 61 * 12>>) -> tensor<1x!ckks.poly<2 * 61 * 12>>
    %196 = "ckks.mulcc"(%195, %185) : (tensor<1x!ckks.poly<2 * 61 * 12>>, tensor<1x!ckks.poly<2 * 51 * 12>>) -> tensor<1x!ckks.poly<3 * 112 * 12>>
    %197 = "ckks.relinearize"(%196) : (tensor<1x!ckks.poly<3 * 112 * 12>>) -> tensor<1x!ckks.poly<2 * 112 * 12>>
    %198 = "ckks.modswitchc"(%171) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 12>>
    %199 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 60.258236740909908 : f64} : () -> tensor<1x!ckks.poly<1 * 61 * 12>>
    %200 = "ckks.mulcp"(%198, %199) : (tensor<1x!ckks.poly<2 * 51 * 12>>, tensor<1x!ckks.poly<1 * 61 * 12>>) -> tensor<1x!ckks.poly<2 * 112 * 12>>
    %201 = "ckks.addcc"(%197, %200) : (tensor<1x!ckks.poly<2 * 112 * 12>>, tensor<1x!ckks.poly<2 * 112 * 12>>) -> tensor<1x!ckks.poly<2 * 112 * 12>>
    %202 = "ckks.rescalec"(%201) : (tensor<1x!ckks.poly<2 * 112 * 12>>) -> tensor<1x!ckks.poly<2 * 81 * 11>>
    %203 = "ckks.rescalec"(%202) : (tensor<1x!ckks.poly<2 * 81 * 11>>) -> tensor<1x!ckks.poly<2 * 51 * 10>>
    %204 = "ckks.addcc"(%185, %185) : (tensor<1x!ckks.poly<2 * 51 * 12>>, tensor<1x!ckks.poly<2 * 51 * 12>>) -> tensor<1x!ckks.poly<2 * 51 * 12>>
    %205 = "ckks.mulcc"(%204, %185) : (tensor<1x!ckks.poly<2 * 51 * 12>>, tensor<1x!ckks.poly<2 * 51 * 12>>) -> tensor<1x!ckks.poly<3 * 102 * 12>>
    %206 = "ckks.relinearize"(%205) : (tensor<1x!ckks.poly<3 * 102 * 12>>) -> tensor<1x!ckks.poly<2 * 102 * 12>>
    %207 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 100.32162539565961 : f64} : () -> tensor<1x!ckks.poly<1 * 102 * 12>>
    %208 = "ckks.addcp"(%206, %207) : (tensor<1x!ckks.poly<2 * 102 * 12>>, tensor<1x!ckks.poly<1 * 102 * 12>>) -> tensor<1x!ckks.poly<2 * 102 * 12>>
    %209 = "ckks.rescalec"(%208) : (tensor<1x!ckks.poly<2 * 102 * 12>>) -> tensor<1x!ckks.poly<2 * 71 * 11>>
    %210 = "ckks.rescalec"(%209) : (tensor<1x!ckks.poly<2 * 71 * 11>>) -> tensor<1x!ckks.poly<2 * 41 * 10>>
    %211 = "ckks.addcc"(%210, %210) : (tensor<1x!ckks.poly<2 * 41 * 10>>, tensor<1x!ckks.poly<2 * 41 * 10>>) -> tensor<1x!ckks.poly<2 * 41 * 10>>
    %212 = "ckks.mulcc"(%211, %210) : (tensor<1x!ckks.poly<2 * 41 * 10>>, tensor<1x!ckks.poly<2 * 41 * 10>>) -> tensor<1x!ckks.poly<3 * 82 * 10>>
    %213 = "ckks.relinearize"(%212) : (tensor<1x!ckks.poly<3 * 82 * 10>>) -> tensor<1x!ckks.poly<2 * 82 * 10>>
    %214 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 80.802883502711751 : f64} : () -> tensor<1x!ckks.poly<1 * 82 * 10>>
    %215 = "ckks.addcp"(%213, %214) : (tensor<1x!ckks.poly<2 * 82 * 10>>, tensor<1x!ckks.poly<1 * 82 * 10>>) -> tensor<1x!ckks.poly<2 * 82 * 10>>
    %216 = "ckks.rescalec"(%215) : (tensor<1x!ckks.poly<2 * 82 * 10>>) -> tensor<1x!ckks.poly<2 * 51 * 9>>
    %217 = "ckks.modswitchc"(%arg0) <{downFactor = 18 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 9>>
    %218 = "ckks.modswitchc"(%163) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 11>>
    %219 = "ckks.encode"() <{payload = dense<-0.017732734281918773> : tensor<32768xf64>}> {runtime.scale_log2 = 44.595234605380682 : f64} : () -> tensor<1x!ckks.poly<1 * 45 * 11>>
    %220 = "ckks.mulcp"(%218, %219) : (tensor<1x!ckks.poly<2 * 51 * 11>>, tensor<1x!ckks.poly<1 * 45 * 11>>) -> tensor<1x!ckks.poly<2 * 96 * 11>>
    %221 = "ckks.rescalec"(%220) : (tensor<1x!ckks.poly<2 * 96 * 11>>) -> tensor<1x!ckks.poly<2 * 65 * 10>>
    %222 = "ckks.encode"() <{payload = dense<-0.0090840943441134625> : tensor<32768xf64>}> {runtime.scale_log2 = 64.459260258071509 : f64} : () -> tensor<1x!ckks.poly<1 * 65 * 12>>
    %223 = "ckks.mulcp"(%179, %222) : (tensor<1x!ckks.poly<2 * 61 * 12>>, tensor<1x!ckks.poly<1 * 65 * 12>>) -> tensor<1x!ckks.poly<2 * 126 * 12>>
    %224 = "ckks.rescalec"(%223) : (tensor<1x!ckks.poly<2 * 126 * 12>>) -> tensor<1x!ckks.poly<2 * 95 * 11>>
    %225 = "ckks.rescalec"(%224) : (tensor<1x!ckks.poly<2 * 95 * 11>>) -> tensor<1x!ckks.poly<2 * 65 * 10>>
    %226 = "ckks.addcc"(%221, %225) : (tensor<1x!ckks.poly<2 * 65 * 10>>, tensor<1x!ckks.poly<2 * 65 * 10>>) -> tensor<1x!ckks.poly<2 * 65 * 10>>
    %227 = "ckks.mulcc"(%226, %210) : (tensor<1x!ckks.poly<2 * 65 * 10>>, tensor<1x!ckks.poly<2 * 41 * 10>>) -> tensor<1x!ckks.poly<3 * 106 * 10>>
    %228 = "ckks.relinearize"(%227) : (tensor<1x!ckks.poly<3 * 106 * 10>>) -> tensor<1x!ckks.poly<2 * 106 * 10>>
    %229 = "ckks.rescalec"(%228) : (tensor<1x!ckks.poly<2 * 106 * 10>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %230 = "ckks.modswitchc"(%163) <{downFactor = 5 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 10>>
    %231 = "ckks.encode"() <{payload = dense<-0.19732584204806608> : tensor<32768xf64>}> {runtime.scale_log2 = 55.037942408196471 : f64} : () -> tensor<1x!ckks.poly<1 * 56 * 10>>
    %232 = "ckks.mulcp"(%230, %231) : (tensor<1x!ckks.poly<2 * 51 * 10>>, tensor<1x!ckks.poly<1 * 56 * 10>>) -> tensor<1x!ckks.poly<2 * 107 * 10>>
    %233 = "ckks.rescalec"(%232) : (tensor<1x!ckks.poly<2 * 107 * 10>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %234 = "ckks.modswitchc"(%179) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 12>>) -> tensor<1x!ckks.poly<2 * 61 * 10>>
    %235 = "ckks.encode"() <{payload = dense<-0.1411166195583439> : tensor<32768xf64>}> {runtime.scale_log2 = 44.940518365109092 : f64} : () -> tensor<1x!ckks.poly<1 * 45 * 10>>
    %236 = "ckks.mulcp"(%234, %235) : (tensor<1x!ckks.poly<2 * 61 * 10>>, tensor<1x!ckks.poly<1 * 45 * 10>>) -> tensor<1x!ckks.poly<2 * 106 * 10>>
    %237 = "ckks.rescalec"(%236) : (tensor<1x!ckks.poly<2 * 106 * 10>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %238 = "ckks.addcc"(%233, %237) : (tensor<1x!ckks.poly<2 * 76 * 9>>, tensor<1x!ckks.poly<2 * 76 * 9>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %239 = "ckks.encode"() <{payload = dense<-0.052675736911616873> : tensor<32768xf64>}> {runtime.scale_log2 = 64.797313354677669 : f64} : () -> tensor<1x!ckks.poly<1 * 65 * 10>>
    %240 = "ckks.mulcp"(%194, %239) : (tensor<1x!ckks.poly<2 * 41 * 10>>, tensor<1x!ckks.poly<1 * 65 * 10>>) -> tensor<1x!ckks.poly<2 * 106 * 10>>
    %241 = "ckks.rescalec"(%240) : (tensor<1x!ckks.poly<2 * 106 * 10>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %242 = "ckks.addcc"(%238, %241) : (tensor<1x!ckks.poly<2 * 76 * 9>>, tensor<1x!ckks.poly<2 * 76 * 9>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %243 = "ckks.encode"() <{payload = dense<-0.026312988130283786> : tensor<32768xf64>}> {runtime.scale_log2 = 54.699889311590297 : f64} : () -> tensor<1x!ckks.poly<1 * 55 * 10>>
    %244 = "ckks.mulcp"(%203, %243) : (tensor<1x!ckks.poly<2 * 51 * 10>>, tensor<1x!ckks.poly<1 * 55 * 10>>) -> tensor<1x!ckks.poly<2 * 106 * 10>>
    %245 = "ckks.rescalec"(%244) : (tensor<1x!ckks.poly<2 * 106 * 10>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %246 = "ckks.addcc"(%242, %245) : (tensor<1x!ckks.poly<2 * 76 * 9>>, tensor<1x!ckks.poly<2 * 76 * 9>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %247 = "ckks.addcc"(%229, %246) : (tensor<1x!ckks.poly<2 * 76 * 9>>, tensor<1x!ckks.poly<2 * 76 * 9>>) -> tensor<1x!ckks.poly<2 * 76 * 9>>
    %248 = "ckks.mulcc"(%247, %216) : (tensor<1x!ckks.poly<2 * 76 * 9>>, tensor<1x!ckks.poly<2 * 51 * 9>>) -> tensor<1x!ckks.poly<3 * 127 * 9>>
    %249 = "ckks.relinearize"(%248) : (tensor<1x!ckks.poly<3 * 127 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %250 = "ckks.encode"() <{payload = dense<-0.78912296503546786> : tensor<32768xf64>}> {runtime.scale_log2 = 65.482100702565404 : f64} : () -> tensor<1x!ckks.poly<1 * 66 * 10>>
    %251 = "ckks.mulcp"(%230, %250) : (tensor<1x!ckks.poly<2 * 51 * 10>>, tensor<1x!ckks.poly<1 * 66 * 10>>) -> tensor<1x!ckks.poly<2 * 117 * 10>>
    %252 = "ckks.encode"() <{payload = dense<-0.72741137353811158> : tensor<32768xf64>}> {runtime.scale_log2 = 55.384676659478025 : f64} : () -> tensor<1x!ckks.poly<1 * 56 * 10>>
    %253 = "ckks.mulcp"(%234, %252) : (tensor<1x!ckks.poly<2 * 61 * 10>>, tensor<1x!ckks.poly<1 * 56 * 10>>) -> tensor<1x!ckks.poly<2 * 117 * 10>>
    %254 = "ckks.addcc"(%251, %253) : (tensor<1x!ckks.poly<2 * 117 * 10>>, tensor<1x!ckks.poly<2 * 117 * 10>>) -> tensor<1x!ckks.poly<2 * 117 * 10>>
    %255 = "ckks.encode"() <{payload = dense<-0.32554529142034655> : tensor<32768xf64>}> {runtime.scale_log2 = 75.241471649046602 : f64} : () -> tensor<1x!ckks.poly<1 * 76 * 10>>
    %256 = "ckks.mulcp"(%194, %255) : (tensor<1x!ckks.poly<2 * 41 * 10>>, tensor<1x!ckks.poly<1 * 76 * 10>>) -> tensor<1x!ckks.poly<2 * 117 * 10>>
    %257 = "ckks.addcc"(%254, %256) : (tensor<1x!ckks.poly<2 * 117 * 10>>, tensor<1x!ckks.poly<2 * 117 * 10>>) -> tensor<1x!ckks.poly<2 * 117 * 10>>
    %258 = "ckks.encode"() <{payload = dense<-0.18601751738208003> : tensor<32768xf64>}> {runtime.scale_log2 = 65.14404760595923 : f64} : () -> tensor<1x!ckks.poly<1 * 66 * 10>>
    %259 = "ckks.mulcp"(%203, %258) : (tensor<1x!ckks.poly<2 * 51 * 10>>, tensor<1x!ckks.poly<1 * 66 * 10>>) -> tensor<1x!ckks.poly<2 * 117 * 10>>
    %260 = "ckks.addcc"(%257, %259) : (tensor<1x!ckks.poly<2 * 117 * 10>>, tensor<1x!ckks.poly<2 * 117 * 10>>) -> tensor<1x!ckks.poly<2 * 117 * 10>>
    %261 = "ckks.mulcc"(%260, %210) : (tensor<1x!ckks.poly<2 * 117 * 10>>, tensor<1x!ckks.poly<2 * 41 * 10>>) -> tensor<1x!ckks.poly<3 * 158 * 10>>
    %262 = "ckks.relinearize"(%261) : (tensor<1x!ckks.poly<3 * 158 * 10>>) -> tensor<1x!ckks.poly<2 * 158 * 10>>
    %263 = "ckks.rescalec"(%262) : (tensor<1x!ckks.poly<2 * 158 * 10>>) -> tensor<1x!ckks.poly<2 * 126 * 9>>
    %264 = "ckks.modswitchc"(%163) <{downFactor = 6 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 15>>) -> tensor<1x!ckks.poly<2 * 51 * 9>>
    %265 = "ckks.encode"() <{payload = dense<-0.042857894008498022> : tensor<32768xf64>}> {runtime.scale_log2 = 75.926258996934308 : f64} : () -> tensor<1x!ckks.poly<1 * 76 * 9>>
    %266 = "ckks.mulcp"(%264, %265) : (tensor<1x!ckks.poly<2 * 51 * 9>>, tensor<1x!ckks.poly<1 * 76 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %267 = "ckks.modswitchc"(%179) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 61 * 12>>) -> tensor<1x!ckks.poly<2 * 61 * 9>>
    %268 = "ckks.encode"() <{payload = dense<-0.7747926099066248> : tensor<32768xf64>}> {runtime.scale_log2 = 65.828834953846922 : f64} : () -> tensor<1x!ckks.poly<1 * 66 * 9>>
    %269 = "ckks.mulcp"(%267, %268) : (tensor<1x!ckks.poly<2 * 61 * 9>>, tensor<1x!ckks.poly<1 * 66 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %270 = "ckks.addcc"(%266, %269) : (tensor<1x!ckks.poly<2 * 127 * 9>>, tensor<1x!ckks.poly<2 * 127 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %271 = "ckks.modswitchc"(%194) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 41 * 10>>) -> tensor<1x!ckks.poly<2 * 41 * 9>>
    %272 = "ckks.encode"() <{payload = dense<-0.18128174370012556> : tensor<32768xf64>}> {runtime.scale_log2 = 85.685629943415506 : f64} : () -> tensor<1x!ckks.poly<1 * 86 * 9>>
    %273 = "ckks.mulcp"(%271, %272) : (tensor<1x!ckks.poly<2 * 41 * 9>>, tensor<1x!ckks.poly<1 * 86 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %274 = "ckks.addcc"(%270, %273) : (tensor<1x!ckks.poly<2 * 127 * 9>>, tensor<1x!ckks.poly<2 * 127 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %275 = "ckks.modswitchc"(%203) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 51 * 10>>) -> tensor<1x!ckks.poly<2 * 51 * 9>>
    %276 = "ckks.encode"() <{payload = dense<-0.24546846913474671> : tensor<32768xf64>}> {runtime.scale_log2 = 75.588205900328134 : f64} : () -> tensor<1x!ckks.poly<1 * 76 * 9>>
    %277 = "ckks.mulcp"(%275, %276) : (tensor<1x!ckks.poly<2 * 51 * 9>>, tensor<1x!ckks.poly<1 * 76 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %278 = "ckks.addcc"(%274, %277) : (tensor<1x!ckks.poly<2 * 127 * 9>>, tensor<1x!ckks.poly<2 * 127 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %279 = "ckks.addcc"(%263, %278) : (tensor<1x!ckks.poly<2 * 126 * 9>>, tensor<1x!ckks.poly<2 * 127 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %280 = "ckks.addcc"(%249, %279) : (tensor<1x!ckks.poly<2 * 127 * 9>>, tensor<1x!ckks.poly<2 * 127 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %281 = "ckks.encode"() <{payload = dense<5.000000e-01> : tensor<32768xf64>}> {runtime.scale_log2 = 125.92625899698344 : f64} : () -> tensor<1x!ckks.poly<1 * 127 * 9>>
    %282 = "ckks.addcp"(%280, %281) : (tensor<1x!ckks.poly<2 * 127 * 9>>, tensor<1x!ckks.poly<1 * 127 * 9>>) -> tensor<1x!ckks.poly<2 * 127 * 9>>
    %283 = "ckks.mulcc"(%217, %282) : (tensor<1x!ckks.poly<2 * 40 * 9>>, tensor<1x!ckks.poly<2 * 127 * 9>>) -> tensor<1x!ckks.poly<3 * 167 * 9>>
    %284 = "ckks.relinearize"(%283) : (tensor<1x!ckks.poly<3 * 167 * 9>>) -> tensor<1x!ckks.poly<2 * 167 * 9>>
    %285 = "ckks.rescalec"(%284) : (tensor<1x!ckks.poly<2 * 167 * 9>>) -> tensor<1x!ckks.poly<2 * 136 * 8>>
    %286 = "ckks.rescalec"(%285) : (tensor<1x!ckks.poly<2 * 136 * 8>>) -> tensor<1x!ckks.poly<2 * 104 * 7>>
    %287 = "ckks.rescalec"(%286) : (tensor<1x!ckks.poly<2 * 104 * 7>>) -> tensor<1x!ckks.poly<2 * 72 * 6>>
    %288 = "ckks.rescalec"(%287) : (tensor<1x!ckks.poly<2 * 72 * 6>>) -> tensor<1x!ckks.poly<2 * 41 * 5>>
    return %288 : tensor<1x!ckks.poly<2 * 41 * 5>>
  }
}

