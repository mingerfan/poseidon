module {
  func.func @relu(%arg0: tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 7>> attributes {absorb_coefficient_scale, arg_scale = array<i64: 40>, btp_schedule = [], btp_target = array<i64>, init_level = 31 : i64, opid0 = 1 : i64, planned_latency = 2.822600e+04 : f64, res_scale = array<i64: 40>, selected_set = 0 : i64, smu0 = 0 : i64, smu_attached = true} {
    %0 = "ckks.encode"() <{payload = dense<-4.0270201200636233> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 31>>
    %1 = "ckks.encode"() <{payload = dense<-0.56048531218535647> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 21>>
    %2 = "ckks.encode"() <{payload = dense<-0.45988074662238737> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 21>>
    %3 = "ckks.encode"() <{payload = dense<-4.1144251614039149> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 25>>
    %4 = "ckks.encode"() <{payload = dense<-0.18601751738208003> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 12>>
    %5 = "ckks.modswitchc"(%arg0) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 30>>
    %6 = "ckks.addcc"(%5, %5) : (tensor<1x!ckks.poly<2 * 40 * 30>>, tensor<1x!ckks.poly<2 * 40 * 30>>) -> tensor<1x!ckks.poly<2 * 40 * 30>>
    %7 = "ckks.addcc"(%arg0, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 31>>, tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 31>>
    %8 = "ckks.mulcc"(%7, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 31>>, tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<3 * 80 * 31>>
    %9 = "ckks.relinearize"(%8) : (tensor<1x!ckks.poly<3 * 80 * 31>>) -> tensor<1x!ckks.poly<2 * 80 * 31>>
    %10 = "ckks.rescalec"(%9) : (tensor<1x!ckks.poly<2 * 80 * 31>>) -> tensor<1x!ckks.poly<2 * 50 * 30>>
    %11 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 30>>
    %12 = "ckks.addcp"(%10, %11) : (tensor<1x!ckks.poly<2 * 50 * 30>>, tensor<1x!ckks.poly<1 * 50 * 30>>) -> tensor<1x!ckks.poly<2 * 50 * 30>>
    %13 = "ckks.mulcc"(%6, %12) : (tensor<1x!ckks.poly<2 * 40 * 30>>, tensor<1x!ckks.poly<2 * 50 * 30>>) -> tensor<1x!ckks.poly<3 * 90 * 30>>
    %14 = "ckks.relinearize"(%13) : (tensor<1x!ckks.poly<3 * 90 * 30>>) -> tensor<1x!ckks.poly<2 * 90 * 30>>
    %15 = "ckks.rescalec"(%14) : (tensor<1x!ckks.poly<2 * 90 * 30>>) -> tensor<1x!ckks.poly<2 * 60 * 29>>
    %16 = "ckks.modswitchc"(%arg0) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 29>>
    %17 = "ckks.negatec"(%16) : (tensor<1x!ckks.poly<2 * 40 * 29>>) -> tensor<1x!ckks.poly<2 * 40 * 29>>
    %18 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 20 * 29>>
    %19 = "ckks.mulcp"(%17, %18) : (tensor<1x!ckks.poly<2 * 40 * 29>>, tensor<1x!ckks.poly<1 * 20 * 29>>) -> tensor<1x!ckks.poly<2 * 60 * 29>>
    %20 = "ckks.addcc"(%15, %19) : (tensor<1x!ckks.poly<2 * 60 * 29>>, tensor<1x!ckks.poly<2 * 60 * 29>>) -> tensor<1x!ckks.poly<2 * 60 * 29>>
    %21 = "ckks.mulcp"(%arg0, %0) : (tensor<1x!ckks.poly<2 * 40 * 31>>, tensor<1x!ckks.poly<1 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 80 * 31>>
    %22 = "ckks.rescalec"(%21) : (tensor<1x!ckks.poly<2 * 80 * 31>>) -> tensor<1x!ckks.poly<2 * 50 * 30>>
    %23 = "ckks.mulcc"(%22, %12) : (tensor<1x!ckks.poly<2 * 50 * 30>>, tensor<1x!ckks.poly<2 * 50 * 30>>) -> tensor<1x!ckks.poly<3 * 100 * 30>>
    %24 = "ckks.relinearize"(%23) : (tensor<1x!ckks.poly<3 * 100 * 30>>) -> tensor<1x!ckks.poly<2 * 100 * 30>>
    %25 = "ckks.rescalec"(%24) : (tensor<1x!ckks.poly<2 * 100 * 30>>) -> tensor<1x!ckks.poly<2 * 70 * 29>>
    %26 = "ckks.rescalec"(%25) : (tensor<1x!ckks.poly<2 * 70 * 29>>) -> tensor<1x!ckks.poly<2 * 40 * 28>>
    %27 = "ckks.encode"() <{payload = dense<2.243187978526648> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 29>>
    %28 = "ckks.mulcp"(%16, %27) : (tensor<1x!ckks.poly<2 * 40 * 29>>, tensor<1x!ckks.poly<1 * 40 * 29>>) -> tensor<1x!ckks.poly<2 * 80 * 29>>
    %29 = "ckks.rescalec"(%28) : (tensor<1x!ckks.poly<2 * 80 * 29>>) -> tensor<1x!ckks.poly<2 * 50 * 28>>
    %30 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 28>>
    %31 = "ckks.mulcp"(%26, %30) : (tensor<1x!ckks.poly<2 * 40 * 28>>, tensor<1x!ckks.poly<1 * 10 * 28>>) -> tensor<1x!ckks.poly<2 * 50 * 28>>
    %32 = "ckks.addcc"(%31, %29) : (tensor<1x!ckks.poly<2 * 50 * 28>>, tensor<1x!ckks.poly<2 * 50 * 28>>) -> tensor<1x!ckks.poly<2 * 50 * 28>>
    %33 = "ckks.addcc"(%12, %12) : (tensor<1x!ckks.poly<2 * 50 * 30>>, tensor<1x!ckks.poly<2 * 50 * 30>>) -> tensor<1x!ckks.poly<2 * 50 * 30>>
    %34 = "ckks.mulcc"(%33, %12) : (tensor<1x!ckks.poly<2 * 50 * 30>>, tensor<1x!ckks.poly<2 * 50 * 30>>) -> tensor<1x!ckks.poly<3 * 100 * 30>>
    %35 = "ckks.relinearize"(%34) : (tensor<1x!ckks.poly<3 * 100 * 30>>) -> tensor<1x!ckks.poly<2 * 100 * 30>>
    %36 = "ckks.rescalec"(%35) : (tensor<1x!ckks.poly<2 * 100 * 30>>) -> tensor<1x!ckks.poly<2 * 70 * 29>>
    %37 = "ckks.rescalec"(%36) : (tensor<1x!ckks.poly<2 * 70 * 29>>) -> tensor<1x!ckks.poly<2 * 40 * 28>>
    %38 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 28>>
    %39 = "ckks.addcp"(%37, %38) : (tensor<1x!ckks.poly<2 * 40 * 28>>, tensor<1x!ckks.poly<1 * 40 * 28>>) -> tensor<1x!ckks.poly<2 * 40 * 28>>
    %40 = "ckks.mulcc"(%32, %39) : (tensor<1x!ckks.poly<2 * 50 * 28>>, tensor<1x!ckks.poly<2 * 40 * 28>>) -> tensor<1x!ckks.poly<3 * 90 * 28>>
    %41 = "ckks.relinearize"(%40) : (tensor<1x!ckks.poly<3 * 90 * 28>>) -> tensor<1x!ckks.poly<2 * 90 * 28>>
    %42 = "ckks.rescalec"(%41) : (tensor<1x!ckks.poly<2 * 90 * 28>>) -> tensor<1x!ckks.poly<2 * 60 * 27>>
    %43 = "ckks.encode"() <{payload = dense<1.1583708205212988> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 28>>
    %44 = "ckks.modswitchc"(%arg0) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 28>>
    %45 = "ckks.mulcp"(%44, %43) : (tensor<1x!ckks.poly<2 * 40 * 28>>, tensor<1x!ckks.poly<1 * 40 * 28>>) -> tensor<1x!ckks.poly<2 * 80 * 28>>
    %46 = "ckks.rescalec"(%45) : (tensor<1x!ckks.poly<2 * 80 * 28>>) -> tensor<1x!ckks.poly<2 * 50 * 27>>
    %47 = "ckks.encode"() <{payload = dense<-0.24378614014677616> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 29>>
    %48 = "ckks.mulcp"(%20, %47) : (tensor<1x!ckks.poly<2 * 60 * 29>>, tensor<1x!ckks.poly<1 * 50 * 29>>) -> tensor<1x!ckks.poly<2 * 110 * 29>>
    %49 = "ckks.rescalec"(%48) : (tensor<1x!ckks.poly<2 * 110 * 29>>) -> tensor<1x!ckks.poly<2 * 80 * 28>>
    %50 = "ckks.rescalec"(%49) : (tensor<1x!ckks.poly<2 * 80 * 28>>) -> tensor<1x!ckks.poly<2 * 50 * 27>>
    %51 = "ckks.addcc"(%46, %50) : (tensor<1x!ckks.poly<2 * 50 * 27>>, tensor<1x!ckks.poly<2 * 50 * 27>>) -> tensor<1x!ckks.poly<2 * 50 * 27>>
    %52 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 27>>
    %53 = "ckks.mulcp"(%51, %52) : (tensor<1x!ckks.poly<2 * 50 * 27>>, tensor<1x!ckks.poly<1 * 10 * 27>>) -> tensor<1x!ckks.poly<2 * 60 * 27>>
    %54 = "ckks.addcc"(%42, %53) : (tensor<1x!ckks.poly<2 * 60 * 27>>, tensor<1x!ckks.poly<2 * 60 * 27>>) -> tensor<1x!ckks.poly<2 * 60 * 27>>
    %55 = "ckks.addcc"(%39, %39) : (tensor<1x!ckks.poly<2 * 40 * 28>>, tensor<1x!ckks.poly<2 * 40 * 28>>) -> tensor<1x!ckks.poly<2 * 40 * 28>>
    %56 = "ckks.mulcc"(%55, %39) : (tensor<1x!ckks.poly<2 * 40 * 28>>, tensor<1x!ckks.poly<2 * 40 * 28>>) -> tensor<1x!ckks.poly<3 * 80 * 28>>
    %57 = "ckks.relinearize"(%56) : (tensor<1x!ckks.poly<3 * 80 * 28>>) -> tensor<1x!ckks.poly<2 * 80 * 28>>
    %58 = "ckks.rescalec"(%57) : (tensor<1x!ckks.poly<2 * 80 * 28>>) -> tensor<1x!ckks.poly<2 * 50 * 27>>
    %59 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 27>>
    %60 = "ckks.addcp"(%58, %59) : (tensor<1x!ckks.poly<2 * 50 * 27>>, tensor<1x!ckks.poly<1 * 50 * 27>>) -> tensor<1x!ckks.poly<2 * 50 * 27>>
    %61 = "ckks.mulcc"(%54, %60) : (tensor<1x!ckks.poly<2 * 60 * 27>>, tensor<1x!ckks.poly<2 * 50 * 27>>) -> tensor<1x!ckks.poly<3 * 110 * 27>>
    %62 = "ckks.relinearize"(%61) : (tensor<1x!ckks.poly<3 * 110 * 27>>) -> tensor<1x!ckks.poly<2 * 110 * 27>>
    %63 = "ckks.rescalec"(%62) : (tensor<1x!ckks.poly<2 * 110 * 27>>) -> tensor<1x!ckks.poly<2 * 80 * 26>>
    %64 = "ckks.rescalec"(%63) : (tensor<1x!ckks.poly<2 * 80 * 26>>) -> tensor<1x!ckks.poly<2 * 50 * 25>>
    %65 = "ckks.encode"() <{payload = dense<0.38878894178135931> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 27>>
    %66 = "ckks.modswitchc"(%arg0) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    %67 = "ckks.mulcp"(%66, %65) : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 80 * 27>>
    %68 = "ckks.rescalec"(%67) : (tensor<1x!ckks.poly<2 * 80 * 27>>) -> tensor<1x!ckks.poly<2 * 50 * 26>>
    %69 = "ckks.modswitchc"(%20) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 29>>) -> tensor<1x!ckks.poly<2 * 60 * 28>>
    %70 = "ckks.encode"() <{payload = dense<-0.34103182272202132> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 28>>
    %71 = "ckks.mulcp"(%69, %70) : (tensor<1x!ckks.poly<2 * 60 * 28>>, tensor<1x!ckks.poly<1 * 50 * 28>>) -> tensor<1x!ckks.poly<2 * 110 * 28>>
    %72 = "ckks.rescalec"(%71) : (tensor<1x!ckks.poly<2 * 110 * 28>>) -> tensor<1x!ckks.poly<2 * 80 * 27>>
    %73 = "ckks.rescalec"(%72) : (tensor<1x!ckks.poly<2 * 80 * 27>>) -> tensor<1x!ckks.poly<2 * 50 * 26>>
    %74 = "ckks.addcc"(%68, %73) : (tensor<1x!ckks.poly<2 * 50 * 26>>, tensor<1x!ckks.poly<2 * 50 * 26>>) -> tensor<1x!ckks.poly<2 * 50 * 26>>
    %75 = "ckks.modswitchc"(%39) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 28>>) -> tensor<1x!ckks.poly<2 * 40 * 26>>
    %76 = "ckks.mulcc"(%74, %75) : (tensor<1x!ckks.poly<2 * 50 * 26>>, tensor<1x!ckks.poly<2 * 40 * 26>>) -> tensor<1x!ckks.poly<3 * 90 * 26>>
    %77 = "ckks.relinearize"(%76) : (tensor<1x!ckks.poly<3 * 90 * 26>>) -> tensor<1x!ckks.poly<2 * 90 * 26>>
    %78 = "ckks.rescalec"(%77) : (tensor<1x!ckks.poly<2 * 90 * 26>>) -> tensor<1x!ckks.poly<2 * 60 * 25>>
    %79 = "ckks.encode"() <{payload = dense<1.3110156470890746> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 26>>
    %80 = "ckks.modswitchc"(%arg0) <{downFactor = 5 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 26>>
    %81 = "ckks.mulcp"(%80, %79) : (tensor<1x!ckks.poly<2 * 40 * 26>>, tensor<1x!ckks.poly<1 * 40 * 26>>) -> tensor<1x!ckks.poly<2 * 80 * 26>>
    %82 = "ckks.rescalec"(%81) : (tensor<1x!ckks.poly<2 * 80 * 26>>) -> tensor<1x!ckks.poly<2 * 50 * 25>>
    %83 = "ckks.modswitchc"(%20) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 29>>) -> tensor<1x!ckks.poly<2 * 60 * 27>>
    %84 = "ckks.encode"() <{payload = dense<-0.46554445822061141> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 27>>
    %85 = "ckks.mulcp"(%83, %84) : (tensor<1x!ckks.poly<2 * 60 * 27>>, tensor<1x!ckks.poly<1 * 50 * 27>>) -> tensor<1x!ckks.poly<2 * 110 * 27>>
    %86 = "ckks.rescalec"(%85) : (tensor<1x!ckks.poly<2 * 110 * 27>>) -> tensor<1x!ckks.poly<2 * 80 * 26>>
    %87 = "ckks.rescalec"(%86) : (tensor<1x!ckks.poly<2 * 80 * 26>>) -> tensor<1x!ckks.poly<2 * 50 * 25>>
    %88 = "ckks.addcc"(%82, %87) : (tensor<1x!ckks.poly<2 * 50 * 25>>, tensor<1x!ckks.poly<2 * 50 * 25>>) -> tensor<1x!ckks.poly<2 * 50 * 25>>
    %89 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 25>>
    %90 = "ckks.mulcp"(%88, %89) : (tensor<1x!ckks.poly<2 * 50 * 25>>, tensor<1x!ckks.poly<1 * 10 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 25>>
    %91 = "ckks.addcc"(%78, %90) : (tensor<1x!ckks.poly<2 * 60 * 25>>, tensor<1x!ckks.poly<2 * 60 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 25>>
    %92 = "ckks.mulcp"(%64, %89) : (tensor<1x!ckks.poly<2 * 50 * 25>>, tensor<1x!ckks.poly<1 * 10 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 25>>
    %93 = "ckks.addcc"(%92, %91) : (tensor<1x!ckks.poly<2 * 60 * 25>>, tensor<1x!ckks.poly<2 * 60 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 25>>
    %94 = "ckks.modswitchc"(%93) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 23>>
    %95 = "ckks.addcc"(%94, %94) : (tensor<1x!ckks.poly<2 * 60 * 23>>, tensor<1x!ckks.poly<2 * 60 * 23>>) -> tensor<1x!ckks.poly<2 * 60 * 23>>
    %96 = "ckks.addcc"(%93, %93) : (tensor<1x!ckks.poly<2 * 60 * 25>>, tensor<1x!ckks.poly<2 * 60 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 25>>
    %97 = "ckks.mulcc"(%96, %93) : (tensor<1x!ckks.poly<2 * 60 * 25>>, tensor<1x!ckks.poly<2 * 60 * 25>>) -> tensor<1x!ckks.poly<3 * 120 * 25>>
    %98 = "ckks.relinearize"(%97) : (tensor<1x!ckks.poly<3 * 120 * 25>>) -> tensor<1x!ckks.poly<2 * 120 * 25>>
    %99 = "ckks.rescalec"(%98) : (tensor<1x!ckks.poly<2 * 120 * 25>>) -> tensor<1x!ckks.poly<2 * 90 * 24>>
    %100 = "ckks.rescalec"(%99) : (tensor<1x!ckks.poly<2 * 90 * 24>>) -> tensor<1x!ckks.poly<2 * 60 * 23>>
    %101 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 23>>
    %102 = "ckks.addcp"(%100, %101) : (tensor<1x!ckks.poly<2 * 60 * 23>>, tensor<1x!ckks.poly<1 * 60 * 23>>) -> tensor<1x!ckks.poly<2 * 60 * 23>>
    %103 = "ckks.mulcc"(%95, %102) : (tensor<1x!ckks.poly<2 * 60 * 23>>, tensor<1x!ckks.poly<2 * 60 * 23>>) -> tensor<1x!ckks.poly<3 * 120 * 23>>
    %104 = "ckks.relinearize"(%103) : (tensor<1x!ckks.poly<3 * 120 * 23>>) -> tensor<1x!ckks.poly<2 * 120 * 23>>
    %105 = "ckks.rescalec"(%104) : (tensor<1x!ckks.poly<2 * 120 * 23>>) -> tensor<1x!ckks.poly<2 * 90 * 22>>
    %106 = "ckks.rescalec"(%105) : (tensor<1x!ckks.poly<2 * 90 * 22>>) -> tensor<1x!ckks.poly<2 * 60 * 21>>
    %107 = "ckks.modswitchc"(%93) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 21>>
    %108 = "ckks.negatec"(%107) : (tensor<1x!ckks.poly<2 * 60 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 21>>
    %109 = "ckks.addcc"(%106, %108) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<2 * 60 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 21>>
    %110 = "ckks.mulcp"(%93, %3) : (tensor<1x!ckks.poly<2 * 60 * 25>>, tensor<1x!ckks.poly<1 * 40 * 25>>) -> tensor<1x!ckks.poly<2 * 100 * 25>>
    %111 = "ckks.rescalec"(%110) : (tensor<1x!ckks.poly<2 * 100 * 25>>) -> tensor<1x!ckks.poly<2 * 70 * 24>>
    %112 = "ckks.rescalec"(%111) : (tensor<1x!ckks.poly<2 * 70 * 24>>) -> tensor<1x!ckks.poly<2 * 40 * 23>>
    %113 = "ckks.mulcc"(%112, %102) : (tensor<1x!ckks.poly<2 * 40 * 23>>, tensor<1x!ckks.poly<2 * 60 * 23>>) -> tensor<1x!ckks.poly<3 * 100 * 23>>
    %114 = "ckks.relinearize"(%113) : (tensor<1x!ckks.poly<3 * 100 * 23>>) -> tensor<1x!ckks.poly<2 * 100 * 23>>
    %115 = "ckks.rescalec"(%114) : (tensor<1x!ckks.poly<2 * 100 * 23>>) -> tensor<1x!ckks.poly<2 * 70 * 22>>
    %116 = "ckks.rescalec"(%115) : (tensor<1x!ckks.poly<2 * 70 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 21>>
    %117 = "ckks.encode"() <{payload = dense<0.93680785180227221> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 23>>
    %118 = "ckks.mulcp"(%94, %117) : (tensor<1x!ckks.poly<2 * 60 * 23>>, tensor<1x!ckks.poly<1 * 40 * 23>>) -> tensor<1x!ckks.poly<2 * 100 * 23>>
    %119 = "ckks.rescalec"(%118) : (tensor<1x!ckks.poly<2 * 100 * 23>>) -> tensor<1x!ckks.poly<2 * 70 * 22>>
    %120 = "ckks.rescalec"(%119) : (tensor<1x!ckks.poly<2 * 70 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 21>>
    %121 = "ckks.addcc"(%116, %120) : (tensor<1x!ckks.poly<2 * 40 * 21>>, tensor<1x!ckks.poly<2 * 40 * 21>>) -> tensor<1x!ckks.poly<2 * 40 * 21>>
    %122 = "ckks.addcc"(%102, %102) : (tensor<1x!ckks.poly<2 * 60 * 23>>, tensor<1x!ckks.poly<2 * 60 * 23>>) -> tensor<1x!ckks.poly<2 * 60 * 23>>
    %123 = "ckks.mulcc"(%122, %102) : (tensor<1x!ckks.poly<2 * 60 * 23>>, tensor<1x!ckks.poly<2 * 60 * 23>>) -> tensor<1x!ckks.poly<3 * 120 * 23>>
    %124 = "ckks.relinearize"(%123) : (tensor<1x!ckks.poly<3 * 120 * 23>>) -> tensor<1x!ckks.poly<2 * 120 * 23>>
    %125 = "ckks.rescalec"(%124) : (tensor<1x!ckks.poly<2 * 120 * 23>>) -> tensor<1x!ckks.poly<2 * 90 * 22>>
    %126 = "ckks.rescalec"(%125) : (tensor<1x!ckks.poly<2 * 90 * 22>>) -> tensor<1x!ckks.poly<2 * 60 * 21>>
    %127 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 21>>
    %128 = "ckks.addcp"(%126, %127) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<1 * 60 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 21>>
    %129 = "ckks.mulcc"(%121, %128) : (tensor<1x!ckks.poly<2 * 40 * 21>>, tensor<1x!ckks.poly<2 * 60 * 21>>) -> tensor<1x!ckks.poly<3 * 100 * 21>>
    %130 = "ckks.relinearize"(%129) : (tensor<1x!ckks.poly<3 * 100 * 21>>) -> tensor<1x!ckks.poly<2 * 100 * 21>>
    %131 = "ckks.rescalec"(%130) : (tensor<1x!ckks.poly<2 * 100 * 21>>) -> tensor<1x!ckks.poly<2 * 70 * 20>>
    %132 = "ckks.rescalec"(%131) : (tensor<1x!ckks.poly<2 * 70 * 20>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %133 = "ckks.encode"() <{payload = dense<0.13312678038183748> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 21>>
    %134 = "ckks.mulcp"(%107, %133) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<1 * 40 * 21>>) -> tensor<1x!ckks.poly<2 * 100 * 21>>
    %135 = "ckks.rescalec"(%134) : (tensor<1x!ckks.poly<2 * 100 * 21>>) -> tensor<1x!ckks.poly<2 * 70 * 20>>
    %136 = "ckks.rescalec"(%135) : (tensor<1x!ckks.poly<2 * 70 * 20>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %137 = "ckks.mulcp"(%109, %2) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<1 * 40 * 21>>) -> tensor<1x!ckks.poly<2 * 100 * 21>>
    %138 = "ckks.rescalec"(%137) : (tensor<1x!ckks.poly<2 * 100 * 21>>) -> tensor<1x!ckks.poly<2 * 70 * 20>>
    %139 = "ckks.rescalec"(%138) : (tensor<1x!ckks.poly<2 * 70 * 20>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %140 = "ckks.addcc"(%136, %139) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %141 = "ckks.addcc"(%132, %140) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %142 = "ckks.addcc"(%128, %128) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<2 * 60 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 21>>
    %143 = "ckks.mulcc"(%142, %128) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<2 * 60 * 21>>) -> tensor<1x!ckks.poly<3 * 120 * 21>>
    %144 = "ckks.relinearize"(%143) : (tensor<1x!ckks.poly<3 * 120 * 21>>) -> tensor<1x!ckks.poly<2 * 120 * 21>>
    %145 = "ckks.rescalec"(%144) : (tensor<1x!ckks.poly<2 * 120 * 21>>) -> tensor<1x!ckks.poly<2 * 90 * 20>>
    %146 = "ckks.rescalec"(%145) : (tensor<1x!ckks.poly<2 * 90 * 20>>) -> tensor<1x!ckks.poly<2 * 60 * 19>>
    %147 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 19>>
    %148 = "ckks.addcp"(%146, %147) : (tensor<1x!ckks.poly<2 * 60 * 19>>, tensor<1x!ckks.poly<1 * 60 * 19>>) -> tensor<1x!ckks.poly<2 * 60 * 19>>
    %149 = "ckks.mulcc"(%141, %148) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<2 * 60 * 19>>) -> tensor<1x!ckks.poly<3 * 100 * 19>>
    %150 = "ckks.relinearize"(%149) : (tensor<1x!ckks.poly<3 * 100 * 19>>) -> tensor<1x!ckks.poly<2 * 100 * 19>>
    %151 = "ckks.rescalec"(%150) : (tensor<1x!ckks.poly<2 * 100 * 19>>) -> tensor<1x!ckks.poly<2 * 70 * 18>>
    %152 = "ckks.rescalec"(%151) : (tensor<1x!ckks.poly<2 * 70 * 18>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %153 = "ckks.encode"() <{payload = dense<-0.058402193560527074> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 21>>
    %154 = "ckks.mulcp"(%107, %153) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<1 * 40 * 21>>) -> tensor<1x!ckks.poly<2 * 100 * 21>>
    %155 = "ckks.rescalec"(%154) : (tensor<1x!ckks.poly<2 * 100 * 21>>) -> tensor<1x!ckks.poly<2 * 70 * 20>>
    %156 = "ckks.rescalec"(%155) : (tensor<1x!ckks.poly<2 * 70 * 20>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %157 = "ckks.mulcp"(%109, %1) : (tensor<1x!ckks.poly<2 * 60 * 21>>, tensor<1x!ckks.poly<1 * 40 * 21>>) -> tensor<1x!ckks.poly<2 * 100 * 21>>
    %158 = "ckks.rescalec"(%157) : (tensor<1x!ckks.poly<2 * 100 * 21>>) -> tensor<1x!ckks.poly<2 * 70 * 20>>
    %159 = "ckks.rescalec"(%158) : (tensor<1x!ckks.poly<2 * 70 * 20>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %160 = "ckks.addcc"(%156, %159) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %161 = "ckks.modswitchc"(%128) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 19>>
    %162 = "ckks.mulcc"(%160, %161) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<2 * 60 * 19>>) -> tensor<1x!ckks.poly<3 * 100 * 19>>
    %163 = "ckks.relinearize"(%162) : (tensor<1x!ckks.poly<3 * 100 * 19>>) -> tensor<1x!ckks.poly<2 * 100 * 19>>
    %164 = "ckks.rescalec"(%163) : (tensor<1x!ckks.poly<2 * 100 * 19>>) -> tensor<1x!ckks.poly<2 * 70 * 18>>
    %165 = "ckks.rescalec"(%164) : (tensor<1x!ckks.poly<2 * 70 * 18>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %166 = "ckks.encode"() <{payload = dense<0.77658111917095896> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 19>>
    %167 = "ckks.modswitchc"(%93) <{downFactor = 6 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 25>>) -> tensor<1x!ckks.poly<2 * 60 * 19>>
    %168 = "ckks.mulcp"(%167, %166) : (tensor<1x!ckks.poly<2 * 60 * 19>>, tensor<1x!ckks.poly<1 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 100 * 19>>
    %169 = "ckks.rescalec"(%168) : (tensor<1x!ckks.poly<2 * 100 * 19>>) -> tensor<1x!ckks.poly<2 * 70 * 18>>
    %170 = "ckks.rescalec"(%169) : (tensor<1x!ckks.poly<2 * 70 * 18>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %171 = "ckks.encode"() <{payload = dense<-0.68184097760015783> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 19>>
    %172 = "ckks.modswitchc"(%109) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 21>>) -> tensor<1x!ckks.poly<2 * 60 * 19>>
    %173 = "ckks.mulcp"(%172, %171) : (tensor<1x!ckks.poly<2 * 60 * 19>>, tensor<1x!ckks.poly<1 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 100 * 19>>
    %174 = "ckks.rescalec"(%173) : (tensor<1x!ckks.poly<2 * 100 * 19>>) -> tensor<1x!ckks.poly<2 * 70 * 18>>
    %175 = "ckks.rescalec"(%174) : (tensor<1x!ckks.poly<2 * 70 * 18>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %176 = "ckks.addcc"(%170, %175) : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %177 = "ckks.addcc"(%165, %176) : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %178 = "ckks.addcc"(%152, %177) : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %179 = "ckks.modswitchc"(%178) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 15>>
    %180 = "ckks.addcc"(%179, %179) : (tensor<1x!ckks.poly<2 * 40 * 15>>, tensor<1x!ckks.poly<2 * 40 * 15>>) -> tensor<1x!ckks.poly<2 * 40 * 15>>
    %181 = "ckks.addcc"(%178, %178) : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %182 = "ckks.mulcc"(%181, %178) : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<3 * 80 * 17>>
    %183 = "ckks.relinearize"(%182) : (tensor<1x!ckks.poly<3 * 80 * 17>>) -> tensor<1x!ckks.poly<2 * 80 * 17>>
    %184 = "ckks.rescalec"(%183) : (tensor<1x!ckks.poly<2 * 80 * 17>>) -> tensor<1x!ckks.poly<2 * 50 * 16>>
    %185 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 16>>
    %186 = "ckks.addcp"(%184, %185) : (tensor<1x!ckks.poly<2 * 50 * 16>>, tensor<1x!ckks.poly<1 * 50 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 16>>
    %187 = "ckks.modswitchc"(%186) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 50 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 15>>
    %188 = "ckks.mulcc"(%180, %187) : (tensor<1x!ckks.poly<2 * 40 * 15>>, tensor<1x!ckks.poly<2 * 50 * 15>>) -> tensor<1x!ckks.poly<3 * 90 * 15>>
    %189 = "ckks.relinearize"(%188) : (tensor<1x!ckks.poly<3 * 90 * 15>>) -> tensor<1x!ckks.poly<2 * 90 * 15>>
    %190 = "ckks.rescalec"(%189) : (tensor<1x!ckks.poly<2 * 90 * 15>>) -> tensor<1x!ckks.poly<2 * 60 * 14>>
    %191 = "ckks.modswitchc"(%178) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 14>>
    %192 = "ckks.negatec"(%191) : (tensor<1x!ckks.poly<2 * 40 * 14>>) -> tensor<1x!ckks.poly<2 * 40 * 14>>
    %193 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 20 * 14>>
    %194 = "ckks.mulcp"(%192, %193) : (tensor<1x!ckks.poly<2 * 40 * 14>>, tensor<1x!ckks.poly<1 * 20 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 14>>
    %195 = "ckks.addcc"(%190, %194) : (tensor<1x!ckks.poly<2 * 60 * 14>>, tensor<1x!ckks.poly<2 * 60 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 14>>
    %196 = "ckks.addcc"(%191, %191) : (tensor<1x!ckks.poly<2 * 40 * 14>>, tensor<1x!ckks.poly<2 * 40 * 14>>) -> tensor<1x!ckks.poly<2 * 40 * 14>>
    %197 = "ckks.addcc"(%186, %186) : (tensor<1x!ckks.poly<2 * 50 * 16>>, tensor<1x!ckks.poly<2 * 50 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 16>>
    %198 = "ckks.mulcc"(%197, %186) : (tensor<1x!ckks.poly<2 * 50 * 16>>, tensor<1x!ckks.poly<2 * 50 * 16>>) -> tensor<1x!ckks.poly<3 * 100 * 16>>
    %199 = "ckks.relinearize"(%198) : (tensor<1x!ckks.poly<3 * 100 * 16>>) -> tensor<1x!ckks.poly<2 * 100 * 16>>
    %200 = "ckks.rescalec"(%199) : (tensor<1x!ckks.poly<2 * 100 * 16>>) -> tensor<1x!ckks.poly<2 * 70 * 15>>
    %201 = "ckks.rescalec"(%200) : (tensor<1x!ckks.poly<2 * 70 * 15>>) -> tensor<1x!ckks.poly<2 * 40 * 14>>
    %202 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 14>>
    %203 = "ckks.addcp"(%201, %202) : (tensor<1x!ckks.poly<2 * 40 * 14>>, tensor<1x!ckks.poly<1 * 40 * 14>>) -> tensor<1x!ckks.poly<2 * 40 * 14>>
    %204 = "ckks.mulcc"(%196, %203) : (tensor<1x!ckks.poly<2 * 40 * 14>>, tensor<1x!ckks.poly<2 * 40 * 14>>) -> tensor<1x!ckks.poly<3 * 80 * 14>>
    %205 = "ckks.relinearize"(%204) : (tensor<1x!ckks.poly<3 * 80 * 14>>) -> tensor<1x!ckks.poly<2 * 80 * 14>>
    %206 = "ckks.rescalec"(%205) : (tensor<1x!ckks.poly<2 * 80 * 14>>) -> tensor<1x!ckks.poly<2 * 50 * 13>>
    %207 = "ckks.addcc"(%195, %195) : (tensor<1x!ckks.poly<2 * 60 * 14>>, tensor<1x!ckks.poly<2 * 60 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 14>>
    %208 = "ckks.mulcc"(%207, %203) : (tensor<1x!ckks.poly<2 * 60 * 14>>, tensor<1x!ckks.poly<2 * 40 * 14>>) -> tensor<1x!ckks.poly<3 * 100 * 14>>
    %209 = "ckks.relinearize"(%208) : (tensor<1x!ckks.poly<3 * 100 * 14>>) -> tensor<1x!ckks.poly<2 * 100 * 14>>
    %210 = "ckks.rescalec"(%209) : (tensor<1x!ckks.poly<2 * 100 * 14>>) -> tensor<1x!ckks.poly<2 * 70 * 13>>
    %211 = "ckks.rescalec"(%210) : (tensor<1x!ckks.poly<2 * 70 * 13>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    %212 = "ckks.modswitchc"(%195) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %213 = "ckks.negatec"(%212) : (tensor<1x!ckks.poly<2 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %214 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 13>>
    %215 = "ckks.mulcp"(%206, %214) : (tensor<1x!ckks.poly<2 * 50 * 13>>, tensor<1x!ckks.poly<1 * 10 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %216 = "ckks.addcc"(%215, %213) : (tensor<1x!ckks.poly<2 * 60 * 13>>, tensor<1x!ckks.poly<2 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %217 = "ckks.modswitchc"(%178) <{downFactor = 5 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    %218 = "ckks.negatec"(%217) : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    %219 = "ckks.addcc"(%211, %218) : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<2 * 40 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    %220 = "ckks.encode"() <{payload = dense<-0.017732734281918773> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 13>>
    %221 = "ckks.modswitchc"(%178) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 13>>
    %222 = "ckks.mulcp"(%221, %220) : (tensor<1x!ckks.poly<2 * 40 * 13>>, tensor<1x!ckks.poly<1 * 40 * 13>>) -> tensor<1x!ckks.poly<2 * 80 * 13>>
    %223 = "ckks.rescalec"(%222) : (tensor<1x!ckks.poly<2 * 80 * 13>>) -> tensor<1x!ckks.poly<2 * 50 * 12>>
    %224 = "ckks.encode"() <{payload = dense<-0.0090840943441134625> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 14>>
    %225 = "ckks.mulcp"(%195, %224) : (tensor<1x!ckks.poly<2 * 60 * 14>>, tensor<1x!ckks.poly<1 * 50 * 14>>) -> tensor<1x!ckks.poly<2 * 110 * 14>>
    %226 = "ckks.rescalec"(%225) : (tensor<1x!ckks.poly<2 * 110 * 14>>) -> tensor<1x!ckks.poly<2 * 80 * 13>>
    %227 = "ckks.rescalec"(%226) : (tensor<1x!ckks.poly<2 * 80 * 13>>) -> tensor<1x!ckks.poly<2 * 50 * 12>>
    %228 = "ckks.addcc"(%223, %227) : (tensor<1x!ckks.poly<2 * 50 * 12>>, tensor<1x!ckks.poly<2 * 50 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 12>>
    %229 = "ckks.modswitchc"(%203) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 14>>) -> tensor<1x!ckks.poly<2 * 40 * 13>>
    %230 = "ckks.addcc"(%229, %229) : (tensor<1x!ckks.poly<2 * 40 * 13>>, tensor<1x!ckks.poly<2 * 40 * 13>>) -> tensor<1x!ckks.poly<2 * 40 * 13>>
    %231 = "ckks.mulcc"(%230, %229) : (tensor<1x!ckks.poly<2 * 40 * 13>>, tensor<1x!ckks.poly<2 * 40 * 13>>) -> tensor<1x!ckks.poly<3 * 80 * 13>>
    %232 = "ckks.relinearize"(%231) : (tensor<1x!ckks.poly<3 * 80 * 13>>) -> tensor<1x!ckks.poly<2 * 80 * 13>>
    %233 = "ckks.rescalec"(%232) : (tensor<1x!ckks.poly<2 * 80 * 13>>) -> tensor<1x!ckks.poly<2 * 50 * 12>>
    %234 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 12>>
    %235 = "ckks.addcp"(%233, %234) : (tensor<1x!ckks.poly<2 * 50 * 12>>, tensor<1x!ckks.poly<1 * 50 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 12>>
    %236 = "ckks.mulcc"(%228, %235) : (tensor<1x!ckks.poly<2 * 50 * 12>>, tensor<1x!ckks.poly<2 * 50 * 12>>) -> tensor<1x!ckks.poly<3 * 100 * 12>>
    %237 = "ckks.relinearize"(%236) : (tensor<1x!ckks.poly<3 * 100 * 12>>) -> tensor<1x!ckks.poly<2 * 100 * 12>>
    %238 = "ckks.rescalec"(%237) : (tensor<1x!ckks.poly<2 * 100 * 12>>) -> tensor<1x!ckks.poly<2 * 70 * 11>>
    %239 = "ckks.rescalec"(%238) : (tensor<1x!ckks.poly<2 * 70 * 11>>) -> tensor<1x!ckks.poly<2 * 40 * 10>>
    %240 = "ckks.encode"() <{payload = dense<-0.19732584204806608> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 11>>
    %241 = "ckks.modswitchc"(%178) <{downFactor = 6 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 11>>
    %242 = "ckks.mulcp"(%241, %240) : (tensor<1x!ckks.poly<2 * 40 * 11>>, tensor<1x!ckks.poly<1 * 40 * 11>>) -> tensor<1x!ckks.poly<2 * 80 * 11>>
    %243 = "ckks.rescalec"(%242) : (tensor<1x!ckks.poly<2 * 80 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %244 = "ckks.modswitchc"(%195) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 12>>
    %245 = "ckks.encode"() <{payload = dense<-0.1411166195583439> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 12>>
    %246 = "ckks.mulcp"(%244, %245) : (tensor<1x!ckks.poly<2 * 60 * 12>>, tensor<1x!ckks.poly<1 * 50 * 12>>) -> tensor<1x!ckks.poly<2 * 110 * 12>>
    %247 = "ckks.rescalec"(%246) : (tensor<1x!ckks.poly<2 * 110 * 12>>) -> tensor<1x!ckks.poly<2 * 80 * 11>>
    %248 = "ckks.rescalec"(%247) : (tensor<1x!ckks.poly<2 * 80 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %249 = "ckks.modswitchc"(%216) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 12>>
    %250 = "ckks.encode"() <{payload = dense<-0.052675736911616873> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 12>>
    %251 = "ckks.mulcp"(%249, %250) : (tensor<1x!ckks.poly<2 * 60 * 12>>, tensor<1x!ckks.poly<1 * 50 * 12>>) -> tensor<1x!ckks.poly<2 * 110 * 12>>
    %252 = "ckks.rescalec"(%251) : (tensor<1x!ckks.poly<2 * 110 * 12>>) -> tensor<1x!ckks.poly<2 * 80 * 11>>
    %253 = "ckks.rescalec"(%252) : (tensor<1x!ckks.poly<2 * 80 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %254 = "ckks.encode"() <{payload = dense<-0.026312988130283786> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 11>>
    %255 = "ckks.modswitchc"(%219) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 11>>
    %256 = "ckks.mulcp"(%255, %254) : (tensor<1x!ckks.poly<2 * 40 * 11>>, tensor<1x!ckks.poly<1 * 40 * 11>>) -> tensor<1x!ckks.poly<2 * 80 * 11>>
    %257 = "ckks.rescalec"(%256) : (tensor<1x!ckks.poly<2 * 80 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %258 = "ckks.addcc"(%243, %248) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<2 * 50 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %259 = "ckks.addcc"(%258, %253) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<2 * 50 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %260 = "ckks.addcc"(%259, %257) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<2 * 50 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %261 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 10>>
    %262 = "ckks.mulcp"(%239, %261) : (tensor<1x!ckks.poly<2 * 40 * 10>>, tensor<1x!ckks.poly<1 * 10 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %263 = "ckks.addcc"(%262, %260) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<2 * 50 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %264 = "ckks.addcc"(%235, %235) : (tensor<1x!ckks.poly<2 * 50 * 12>>, tensor<1x!ckks.poly<2 * 50 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 12>>
    %265 = "ckks.mulcc"(%264, %235) : (tensor<1x!ckks.poly<2 * 50 * 12>>, tensor<1x!ckks.poly<2 * 50 * 12>>) -> tensor<1x!ckks.poly<3 * 100 * 12>>
    %266 = "ckks.relinearize"(%265) : (tensor<1x!ckks.poly<3 * 100 * 12>>) -> tensor<1x!ckks.poly<2 * 100 * 12>>
    %267 = "ckks.rescalec"(%266) : (tensor<1x!ckks.poly<2 * 100 * 12>>) -> tensor<1x!ckks.poly<2 * 70 * 11>>
    %268 = "ckks.rescalec"(%267) : (tensor<1x!ckks.poly<2 * 70 * 11>>) -> tensor<1x!ckks.poly<2 * 40 * 10>>
    %269 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 10>>
    %270 = "ckks.addcp"(%268, %269) : (tensor<1x!ckks.poly<2 * 40 * 10>>, tensor<1x!ckks.poly<1 * 40 * 10>>) -> tensor<1x!ckks.poly<2 * 40 * 10>>
    %271 = "ckks.mulcc"(%263, %270) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<2 * 40 * 10>>) -> tensor<1x!ckks.poly<3 * 90 * 10>>
    %272 = "ckks.relinearize"(%271) : (tensor<1x!ckks.poly<3 * 90 * 10>>) -> tensor<1x!ckks.poly<2 * 90 * 10>>
    %273 = "ckks.rescalec"(%272) : (tensor<1x!ckks.poly<2 * 90 * 10>>) -> tensor<1x!ckks.poly<2 * 60 * 9>>
    %274 = "ckks.encode"() <{payload = dense<-0.78912296503546786> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 12>>
    %275 = "ckks.mulcp"(%217, %274) : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 40 * 12>>) -> tensor<1x!ckks.poly<2 * 80 * 12>>
    %276 = "ckks.rescalec"(%275) : (tensor<1x!ckks.poly<2 * 80 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %277 = "ckks.encode"() <{payload = dense<-0.72741137353811158> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 13>>
    %278 = "ckks.mulcp"(%212, %277) : (tensor<1x!ckks.poly<2 * 60 * 13>>, tensor<1x!ckks.poly<1 * 50 * 13>>) -> tensor<1x!ckks.poly<2 * 110 * 13>>
    %279 = "ckks.rescalec"(%278) : (tensor<1x!ckks.poly<2 * 110 * 13>>) -> tensor<1x!ckks.poly<2 * 80 * 12>>
    %280 = "ckks.rescalec"(%279) : (tensor<1x!ckks.poly<2 * 80 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %281 = "ckks.encode"() <{payload = dense<-0.32554529142034655> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 13>>
    %282 = "ckks.mulcp"(%216, %281) : (tensor<1x!ckks.poly<2 * 60 * 13>>, tensor<1x!ckks.poly<1 * 50 * 13>>) -> tensor<1x!ckks.poly<2 * 110 * 13>>
    %283 = "ckks.rescalec"(%282) : (tensor<1x!ckks.poly<2 * 110 * 13>>) -> tensor<1x!ckks.poly<2 * 80 * 12>>
    %284 = "ckks.rescalec"(%283) : (tensor<1x!ckks.poly<2 * 80 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %285 = "ckks.mulcp"(%219, %4) : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 40 * 12>>) -> tensor<1x!ckks.poly<2 * 80 * 12>>
    %286 = "ckks.rescalec"(%285) : (tensor<1x!ckks.poly<2 * 80 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %287 = "ckks.addcc"(%276, %280) : (tensor<1x!ckks.poly<2 * 50 * 11>>, tensor<1x!ckks.poly<2 * 50 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %288 = "ckks.addcc"(%287, %284) : (tensor<1x!ckks.poly<2 * 50 * 11>>, tensor<1x!ckks.poly<2 * 50 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %289 = "ckks.addcc"(%288, %286) : (tensor<1x!ckks.poly<2 * 50 * 11>>, tensor<1x!ckks.poly<2 * 50 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %290 = "ckks.modswitchc"(%235) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 50 * 12>>) -> tensor<1x!ckks.poly<2 * 50 * 11>>
    %291 = "ckks.mulcc"(%289, %290) : (tensor<1x!ckks.poly<2 * 50 * 11>>, tensor<1x!ckks.poly<2 * 50 * 11>>) -> tensor<1x!ckks.poly<3 * 100 * 11>>
    %292 = "ckks.relinearize"(%291) : (tensor<1x!ckks.poly<3 * 100 * 11>>) -> tensor<1x!ckks.poly<2 * 100 * 11>>
    %293 = "ckks.rescalec"(%292) : (tensor<1x!ckks.poly<2 * 100 * 11>>) -> tensor<1x!ckks.poly<2 * 70 * 10>>
    %294 = "ckks.rescalec"(%293) : (tensor<1x!ckks.poly<2 * 70 * 10>>) -> tensor<1x!ckks.poly<2 * 40 * 9>>
    %295 = "ckks.encode"() <{payload = dense<-0.042857894008498022> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 10>>
    %296 = "ckks.modswitchc"(%178) <{downFactor = 7 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 10>>
    %297 = "ckks.mulcp"(%296, %295) : (tensor<1x!ckks.poly<2 * 40 * 10>>, tensor<1x!ckks.poly<1 * 40 * 10>>) -> tensor<1x!ckks.poly<2 * 80 * 10>>
    %298 = "ckks.rescalec"(%297) : (tensor<1x!ckks.poly<2 * 80 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %299 = "ckks.modswitchc"(%195) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 11>>
    %300 = "ckks.encode"() <{payload = dense<-0.7747926099066248> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 11>>
    %301 = "ckks.mulcp"(%299, %300) : (tensor<1x!ckks.poly<2 * 60 * 11>>, tensor<1x!ckks.poly<1 * 50 * 11>>) -> tensor<1x!ckks.poly<2 * 110 * 11>>
    %302 = "ckks.rescalec"(%301) : (tensor<1x!ckks.poly<2 * 110 * 11>>) -> tensor<1x!ckks.poly<2 * 80 * 10>>
    %303 = "ckks.rescalec"(%302) : (tensor<1x!ckks.poly<2 * 80 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %304 = "ckks.modswitchc"(%216) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 11>>
    %305 = "ckks.encode"() <{payload = dense<-0.18128174370012556> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 11>>
    %306 = "ckks.mulcp"(%304, %305) : (tensor<1x!ckks.poly<2 * 60 * 11>>, tensor<1x!ckks.poly<1 * 50 * 11>>) -> tensor<1x!ckks.poly<2 * 110 * 11>>
    %307 = "ckks.rescalec"(%306) : (tensor<1x!ckks.poly<2 * 110 * 11>>) -> tensor<1x!ckks.poly<2 * 80 * 10>>
    %308 = "ckks.rescalec"(%307) : (tensor<1x!ckks.poly<2 * 80 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %309 = "ckks.encode"() <{payload = dense<-0.24546846913474671> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 10>>
    %310 = "ckks.modswitchc"(%219) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 10>>
    %311 = "ckks.mulcp"(%310, %309) : (tensor<1x!ckks.poly<2 * 40 * 10>>, tensor<1x!ckks.poly<1 * 40 * 10>>) -> tensor<1x!ckks.poly<2 * 80 * 10>>
    %312 = "ckks.rescalec"(%311) : (tensor<1x!ckks.poly<2 * 80 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %313 = "ckks.addcc"(%298, %303) : (tensor<1x!ckks.poly<2 * 50 * 9>>, tensor<1x!ckks.poly<2 * 50 * 9>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %314 = "ckks.addcc"(%313, %308) : (tensor<1x!ckks.poly<2 * 50 * 9>>, tensor<1x!ckks.poly<2 * 50 * 9>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %315 = "ckks.addcc"(%314, %312) : (tensor<1x!ckks.poly<2 * 50 * 9>>, tensor<1x!ckks.poly<2 * 50 * 9>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %316 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 9>>
    %317 = "ckks.mulcp"(%294, %316) : (tensor<1x!ckks.poly<2 * 40 * 9>>, tensor<1x!ckks.poly<1 * 10 * 9>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %318 = "ckks.addcc"(%317, %315) : (tensor<1x!ckks.poly<2 * 50 * 9>>, tensor<1x!ckks.poly<2 * 50 * 9>>) -> tensor<1x!ckks.poly<2 * 50 * 9>>
    %319 = "ckks.mulcp"(%318, %316) : (tensor<1x!ckks.poly<2 * 50 * 9>>, tensor<1x!ckks.poly<1 * 10 * 9>>) -> tensor<1x!ckks.poly<2 * 60 * 9>>
    %320 = "ckks.addcc"(%273, %319) : (tensor<1x!ckks.poly<2 * 60 * 9>>, tensor<1x!ckks.poly<2 * 60 * 9>>) -> tensor<1x!ckks.poly<2 * 60 * 9>>
    %321 = "ckks.encode"() <{payload = dense<5.000000e-01> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 9>>
    %322 = "ckks.addcp"(%320, %321) : (tensor<1x!ckks.poly<2 * 60 * 9>>, tensor<1x!ckks.poly<1 * 60 * 9>>) -> tensor<1x!ckks.poly<2 * 60 * 9>>
    %323 = "ckks.modswitchc"(%arg0) <{downFactor = 22 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 31>>) -> tensor<1x!ckks.poly<2 * 40 * 9>>
    %324 = "ckks.mulcc"(%323, %322) : (tensor<1x!ckks.poly<2 * 40 * 9>>, tensor<1x!ckks.poly<2 * 60 * 9>>) -> tensor<1x!ckks.poly<3 * 100 * 9>>
    %325 = "ckks.relinearize"(%324) : (tensor<1x!ckks.poly<3 * 100 * 9>>) -> tensor<1x!ckks.poly<2 * 100 * 9>>
    %326 = "ckks.rescalec"(%325) : (tensor<1x!ckks.poly<2 * 100 * 9>>) -> tensor<1x!ckks.poly<2 * 70 * 8>>
    %327 = "ckks.rescalec"(%326) : (tensor<1x!ckks.poly<2 * 70 * 8>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    return %327 : tensor<1x!ckks.poly<2 * 40 * 7>>
  }
}

