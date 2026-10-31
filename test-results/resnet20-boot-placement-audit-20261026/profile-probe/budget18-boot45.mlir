module {
  func.func @relu(%arg0: tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 45 * 1>> attributes {absorb_coefficient_scale, arg_scale = array<i64: 40>, btp_schedule = [array<i64: 64, 64>], btp_target = array<i64: 64>, init_level = 19 : i64, opid0 = 1 : i64, planned_latency = 3.691400e+04 : f64, res_scale = array<i64: 45>, selected_set = 1 : i64, smu0 = 0 : i64, smu_attached = true} {
    %0 = "ckks.encode"() <{payload = dense<-4.0270201200636233> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 19>>
    %1 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 5>>
    %2 = "ckks.encode"() <{payload = dense<-4.1144251614039149> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 19>>
    %3 = "ckks.encode"() <{payload = dense<-0.0090840943441134625> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 8>>
    %4 = "ckks.modswitchc"(%arg0) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 18>>
    %5 = "ckks.addcc"(%4, %4) : (tensor<1x!ckks.poly<2 * 40 * 18>>, tensor<1x!ckks.poly<2 * 40 * 18>>) -> tensor<1x!ckks.poly<2 * 40 * 18>>
    %6 = "ckks.addcc"(%arg0, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 19>>
    %7 = "ckks.mulcc"(%6, %arg0) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<3 * 80 * 19>>
    %8 = "ckks.relinearize"(%7) : (tensor<1x!ckks.poly<3 * 80 * 19>>) -> tensor<1x!ckks.poly<2 * 80 * 19>>
    %9 = "ckks.rescalec"(%8) : (tensor<1x!ckks.poly<2 * 80 * 19>>) -> tensor<1x!ckks.poly<2 * 50 * 18>>
    %10 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 18>>
    %11 = "ckks.addcp"(%9, %10) : (tensor<1x!ckks.poly<2 * 50 * 18>>, tensor<1x!ckks.poly<1 * 50 * 18>>) -> tensor<1x!ckks.poly<2 * 50 * 18>>
    %12 = "ckks.mulcc"(%5, %11) : (tensor<1x!ckks.poly<2 * 40 * 18>>, tensor<1x!ckks.poly<2 * 50 * 18>>) -> tensor<1x!ckks.poly<3 * 90 * 18>>
    %13 = "ckks.relinearize"(%12) : (tensor<1x!ckks.poly<3 * 90 * 18>>) -> tensor<1x!ckks.poly<2 * 90 * 18>>
    %14 = "ckks.rescalec"(%13) : (tensor<1x!ckks.poly<2 * 90 * 18>>) -> tensor<1x!ckks.poly<2 * 60 * 17>>
    %15 = "ckks.modswitchc"(%arg0) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %16 = "ckks.negatec"(%15) : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    %17 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 20 * 17>>
    %18 = "ckks.mulcp"(%16, %17) : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 20 * 17>>) -> tensor<1x!ckks.poly<2 * 60 * 17>>
    %19 = "ckks.addcc"(%14, %18) : (tensor<1x!ckks.poly<2 * 60 * 17>>, tensor<1x!ckks.poly<2 * 60 * 17>>) -> tensor<1x!ckks.poly<2 * 60 * 17>>
    %20 = "ckks.mulcp"(%arg0, %0) : (tensor<1x!ckks.poly<2 * 40 * 19>>, tensor<1x!ckks.poly<1 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 80 * 19>>
    %21 = "ckks.rescalec"(%20) : (tensor<1x!ckks.poly<2 * 80 * 19>>) -> tensor<1x!ckks.poly<2 * 50 * 18>>
    %22 = "ckks.mulcc"(%21, %11) : (tensor<1x!ckks.poly<2 * 50 * 18>>, tensor<1x!ckks.poly<2 * 50 * 18>>) -> tensor<1x!ckks.poly<3 * 100 * 18>>
    %23 = "ckks.relinearize"(%22) : (tensor<1x!ckks.poly<3 * 100 * 18>>) -> tensor<1x!ckks.poly<2 * 100 * 18>>
    %24 = "ckks.rescalec"(%23) : (tensor<1x!ckks.poly<2 * 100 * 18>>) -> tensor<1x!ckks.poly<2 * 70 * 17>>
    %25 = "ckks.rescalec"(%24) : (tensor<1x!ckks.poly<2 * 70 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %26 = "ckks.encode"() <{payload = dense<2.243187978526648> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 17>>
    %27 = "ckks.mulcp"(%15, %26) : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 80 * 17>>
    %28 = "ckks.rescalec"(%27) : (tensor<1x!ckks.poly<2 * 80 * 17>>) -> tensor<1x!ckks.poly<2 * 50 * 16>>
    %29 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 16>>
    %30 = "ckks.mulcp"(%25, %29) : (tensor<1x!ckks.poly<2 * 40 * 16>>, tensor<1x!ckks.poly<1 * 10 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 16>>
    %31 = "ckks.addcc"(%30, %28) : (tensor<1x!ckks.poly<2 * 50 * 16>>, tensor<1x!ckks.poly<2 * 50 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 16>>
    %32 = "ckks.addcc"(%11, %11) : (tensor<1x!ckks.poly<2 * 50 * 18>>, tensor<1x!ckks.poly<2 * 50 * 18>>) -> tensor<1x!ckks.poly<2 * 50 * 18>>
    %33 = "ckks.mulcc"(%32, %11) : (tensor<1x!ckks.poly<2 * 50 * 18>>, tensor<1x!ckks.poly<2 * 50 * 18>>) -> tensor<1x!ckks.poly<3 * 100 * 18>>
    %34 = "ckks.relinearize"(%33) : (tensor<1x!ckks.poly<3 * 100 * 18>>) -> tensor<1x!ckks.poly<2 * 100 * 18>>
    %35 = "ckks.rescalec"(%34) : (tensor<1x!ckks.poly<2 * 100 * 18>>) -> tensor<1x!ckks.poly<2 * 70 * 17>>
    %36 = "ckks.rescalec"(%35) : (tensor<1x!ckks.poly<2 * 70 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %37 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 16>>
    %38 = "ckks.addcp"(%36, %37) : (tensor<1x!ckks.poly<2 * 40 * 16>>, tensor<1x!ckks.poly<1 * 40 * 16>>) -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %39 = "ckks.mulcc"(%31, %38) : (tensor<1x!ckks.poly<2 * 50 * 16>>, tensor<1x!ckks.poly<2 * 40 * 16>>) -> tensor<1x!ckks.poly<3 * 90 * 16>>
    %40 = "ckks.relinearize"(%39) : (tensor<1x!ckks.poly<3 * 90 * 16>>) -> tensor<1x!ckks.poly<2 * 90 * 16>>
    %41 = "ckks.rescalec"(%40) : (tensor<1x!ckks.poly<2 * 90 * 16>>) -> tensor<1x!ckks.poly<2 * 60 * 15>>
    %42 = "ckks.encode"() <{payload = dense<1.1583708205212988> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 16>>
    %43 = "ckks.modswitchc"(%arg0) <{downFactor = 3 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %44 = "ckks.mulcp"(%43, %42) : (tensor<1x!ckks.poly<2 * 40 * 16>>, tensor<1x!ckks.poly<1 * 40 * 16>>) -> tensor<1x!ckks.poly<2 * 80 * 16>>
    %45 = "ckks.rescalec"(%44) : (tensor<1x!ckks.poly<2 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 15>>
    %46 = "ckks.encode"() <{payload = dense<-0.24378614014677616> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 17>>
    %47 = "ckks.mulcp"(%19, %46) : (tensor<1x!ckks.poly<2 * 60 * 17>>, tensor<1x!ckks.poly<1 * 50 * 17>>) -> tensor<1x!ckks.poly<2 * 110 * 17>>
    %48 = "ckks.rescalec"(%47) : (tensor<1x!ckks.poly<2 * 110 * 17>>) -> tensor<1x!ckks.poly<2 * 80 * 16>>
    %49 = "ckks.rescalec"(%48) : (tensor<1x!ckks.poly<2 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 15>>
    %50 = "ckks.addcc"(%45, %49) : (tensor<1x!ckks.poly<2 * 50 * 15>>, tensor<1x!ckks.poly<2 * 50 * 15>>) -> tensor<1x!ckks.poly<2 * 50 * 15>>
    %51 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 15>>
    %52 = "ckks.mulcp"(%50, %51) : (tensor<1x!ckks.poly<2 * 50 * 15>>, tensor<1x!ckks.poly<1 * 10 * 15>>) -> tensor<1x!ckks.poly<2 * 60 * 15>>
    %53 = "ckks.addcc"(%41, %52) : (tensor<1x!ckks.poly<2 * 60 * 15>>, tensor<1x!ckks.poly<2 * 60 * 15>>) -> tensor<1x!ckks.poly<2 * 60 * 15>>
    %54 = "ckks.addcc"(%38, %38) : (tensor<1x!ckks.poly<2 * 40 * 16>>, tensor<1x!ckks.poly<2 * 40 * 16>>) -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %55 = "ckks.mulcc"(%54, %38) : (tensor<1x!ckks.poly<2 * 40 * 16>>, tensor<1x!ckks.poly<2 * 40 * 16>>) -> tensor<1x!ckks.poly<3 * 80 * 16>>
    %56 = "ckks.relinearize"(%55) : (tensor<1x!ckks.poly<3 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 80 * 16>>
    %57 = "ckks.rescalec"(%56) : (tensor<1x!ckks.poly<2 * 80 * 16>>) -> tensor<1x!ckks.poly<2 * 50 * 15>>
    %58 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 15>>
    %59 = "ckks.addcp"(%57, %58) : (tensor<1x!ckks.poly<2 * 50 * 15>>, tensor<1x!ckks.poly<1 * 50 * 15>>) -> tensor<1x!ckks.poly<2 * 50 * 15>>
    %60 = "ckks.mulcc"(%53, %59) : (tensor<1x!ckks.poly<2 * 60 * 15>>, tensor<1x!ckks.poly<2 * 50 * 15>>) -> tensor<1x!ckks.poly<3 * 110 * 15>>
    %61 = "ckks.relinearize"(%60) : (tensor<1x!ckks.poly<3 * 110 * 15>>) -> tensor<1x!ckks.poly<2 * 110 * 15>>
    %62 = "ckks.rescalec"(%61) : (tensor<1x!ckks.poly<2 * 110 * 15>>) -> tensor<1x!ckks.poly<2 * 80 * 14>>
    %63 = "ckks.rescalec"(%62) : (tensor<1x!ckks.poly<2 * 80 * 14>>) -> tensor<1x!ckks.poly<2 * 50 * 13>>
    %64 = "ckks.encode"() <{payload = dense<0.38878894178135931> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 15>>
    %65 = "ckks.modswitchc"(%arg0) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 15>>
    %66 = "ckks.mulcp"(%65, %64) : (tensor<1x!ckks.poly<2 * 40 * 15>>, tensor<1x!ckks.poly<1 * 40 * 15>>) -> tensor<1x!ckks.poly<2 * 80 * 15>>
    %67 = "ckks.rescalec"(%66) : (tensor<1x!ckks.poly<2 * 80 * 15>>) -> tensor<1x!ckks.poly<2 * 50 * 14>>
    %68 = "ckks.modswitchc"(%19) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 17>>) -> tensor<1x!ckks.poly<2 * 60 * 16>>
    %69 = "ckks.encode"() <{payload = dense<-0.34103182272202132> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 16>>
    %70 = "ckks.mulcp"(%68, %69) : (tensor<1x!ckks.poly<2 * 60 * 16>>, tensor<1x!ckks.poly<1 * 50 * 16>>) -> tensor<1x!ckks.poly<2 * 110 * 16>>
    %71 = "ckks.rescalec"(%70) : (tensor<1x!ckks.poly<2 * 110 * 16>>) -> tensor<1x!ckks.poly<2 * 80 * 15>>
    %72 = "ckks.rescalec"(%71) : (tensor<1x!ckks.poly<2 * 80 * 15>>) -> tensor<1x!ckks.poly<2 * 50 * 14>>
    %73 = "ckks.addcc"(%67, %72) : (tensor<1x!ckks.poly<2 * 50 * 14>>, tensor<1x!ckks.poly<2 * 50 * 14>>) -> tensor<1x!ckks.poly<2 * 50 * 14>>
    %74 = "ckks.modswitchc"(%38) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 16>>) -> tensor<1x!ckks.poly<2 * 40 * 14>>
    %75 = "ckks.mulcc"(%73, %74) : (tensor<1x!ckks.poly<2 * 50 * 14>>, tensor<1x!ckks.poly<2 * 40 * 14>>) -> tensor<1x!ckks.poly<3 * 90 * 14>>
    %76 = "ckks.relinearize"(%75) : (tensor<1x!ckks.poly<3 * 90 * 14>>) -> tensor<1x!ckks.poly<2 * 90 * 14>>
    %77 = "ckks.rescalec"(%76) : (tensor<1x!ckks.poly<2 * 90 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %78 = "ckks.encode"() <{payload = dense<1.3110156470890746> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 14>>
    %79 = "ckks.modswitchc"(%arg0) <{downFactor = 5 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 14>>
    %80 = "ckks.mulcp"(%79, %78) : (tensor<1x!ckks.poly<2 * 40 * 14>>, tensor<1x!ckks.poly<1 * 40 * 14>>) -> tensor<1x!ckks.poly<2 * 80 * 14>>
    %81 = "ckks.rescalec"(%80) : (tensor<1x!ckks.poly<2 * 80 * 14>>) -> tensor<1x!ckks.poly<2 * 50 * 13>>
    %82 = "ckks.modswitchc"(%19) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 17>>) -> tensor<1x!ckks.poly<2 * 60 * 15>>
    %83 = "ckks.encode"() <{payload = dense<-0.46554445822061141> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 15>>
    %84 = "ckks.mulcp"(%82, %83) : (tensor<1x!ckks.poly<2 * 60 * 15>>, tensor<1x!ckks.poly<1 * 50 * 15>>) -> tensor<1x!ckks.poly<2 * 110 * 15>>
    %85 = "ckks.rescalec"(%84) : (tensor<1x!ckks.poly<2 * 110 * 15>>) -> tensor<1x!ckks.poly<2 * 80 * 14>>
    %86 = "ckks.rescalec"(%85) : (tensor<1x!ckks.poly<2 * 80 * 14>>) -> tensor<1x!ckks.poly<2 * 50 * 13>>
    %87 = "ckks.addcc"(%81, %86) : (tensor<1x!ckks.poly<2 * 50 * 13>>, tensor<1x!ckks.poly<2 * 50 * 13>>) -> tensor<1x!ckks.poly<2 * 50 * 13>>
    %88 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 13>>
    %89 = "ckks.mulcp"(%87, %88) : (tensor<1x!ckks.poly<2 * 50 * 13>>, tensor<1x!ckks.poly<1 * 10 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %90 = "ckks.addcc"(%77, %89) : (tensor<1x!ckks.poly<2 * 60 * 13>>, tensor<1x!ckks.poly<2 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %91 = "ckks.mulcp"(%63, %88) : (tensor<1x!ckks.poly<2 * 50 * 13>>, tensor<1x!ckks.poly<1 * 10 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %92 = "ckks.addcc"(%91, %90) : (tensor<1x!ckks.poly<2 * 60 * 13>>, tensor<1x!ckks.poly<2 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 60 * 13>>
    %93 = "ckks.bootstrapc"(%92) : (tensor<1x!ckks.poly<2 * 60 * 13>>) -> tensor<1x!ckks.poly<2 * 45 * 19>>
    %94 = "ckks.modswitchc"(%93) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 45 * 19>>) -> tensor<1x!ckks.poly<2 * 45 * 17>>
    %95 = "ckks.addcc"(%94, %94) : (tensor<1x!ckks.poly<2 * 45 * 17>>, tensor<1x!ckks.poly<2 * 45 * 17>>) -> tensor<1x!ckks.poly<2 * 45 * 17>>
    %96 = "ckks.addcc"(%93, %93) : (tensor<1x!ckks.poly<2 * 45 * 19>>, tensor<1x!ckks.poly<2 * 45 * 19>>) -> tensor<1x!ckks.poly<2 * 45 * 19>>
    %97 = "ckks.mulcc"(%96, %93) : (tensor<1x!ckks.poly<2 * 45 * 19>>, tensor<1x!ckks.poly<2 * 45 * 19>>) -> tensor<1x!ckks.poly<3 * 90 * 19>>
    %98 = "ckks.relinearize"(%97) : (tensor<1x!ckks.poly<3 * 90 * 19>>) -> tensor<1x!ckks.poly<2 * 90 * 19>>
    %99 = "ckks.rescalec"(%98) : (tensor<1x!ckks.poly<2 * 90 * 19>>) -> tensor<1x!ckks.poly<2 * 60 * 18>>
    %100 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 18>>
    %101 = "ckks.addcp"(%99, %100) : (tensor<1x!ckks.poly<2 * 60 * 18>>, tensor<1x!ckks.poly<1 * 60 * 18>>) -> tensor<1x!ckks.poly<2 * 60 * 18>>
    %102 = "ckks.modswitchc"(%101) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 18>>) -> tensor<1x!ckks.poly<2 * 60 * 17>>
    %103 = "ckks.mulcc"(%95, %102) : (tensor<1x!ckks.poly<2 * 45 * 17>>, tensor<1x!ckks.poly<2 * 60 * 17>>) -> tensor<1x!ckks.poly<3 * 105 * 17>>
    %104 = "ckks.relinearize"(%103) : (tensor<1x!ckks.poly<3 * 105 * 17>>) -> tensor<1x!ckks.poly<2 * 105 * 17>>
    %105 = "ckks.rescalec"(%104) : (tensor<1x!ckks.poly<2 * 105 * 17>>) -> tensor<1x!ckks.poly<2 * 75 * 16>>
    %106 = "ckks.rescalec"(%105) : (tensor<1x!ckks.poly<2 * 75 * 16>>) -> tensor<1x!ckks.poly<2 * 45 * 15>>
    %107 = "ckks.modswitchc"(%93) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 45 * 19>>) -> tensor<1x!ckks.poly<2 * 45 * 15>>
    %108 = "ckks.negatec"(%107) : (tensor<1x!ckks.poly<2 * 45 * 15>>) -> tensor<1x!ckks.poly<2 * 45 * 15>>
    %109 = "ckks.addcc"(%106, %108) : (tensor<1x!ckks.poly<2 * 45 * 15>>, tensor<1x!ckks.poly<2 * 45 * 15>>) -> tensor<1x!ckks.poly<2 * 45 * 15>>
    %110 = "ckks.mulcp"(%93, %2) : (tensor<1x!ckks.poly<2 * 45 * 19>>, tensor<1x!ckks.poly<1 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 85 * 19>>
    %111 = "ckks.rescalec"(%110) : (tensor<1x!ckks.poly<2 * 85 * 19>>) -> tensor<1x!ckks.poly<2 * 55 * 18>>
    %112 = "ckks.mulcc"(%111, %101) : (tensor<1x!ckks.poly<2 * 55 * 18>>, tensor<1x!ckks.poly<2 * 60 * 18>>) -> tensor<1x!ckks.poly<3 * 115 * 18>>
    %113 = "ckks.relinearize"(%112) : (tensor<1x!ckks.poly<3 * 115 * 18>>) -> tensor<1x!ckks.poly<2 * 115 * 18>>
    %114 = "ckks.rescalec"(%113) : (tensor<1x!ckks.poly<2 * 115 * 18>>) -> tensor<1x!ckks.poly<2 * 85 * 17>>
    %115 = "ckks.rescalec"(%114) : (tensor<1x!ckks.poly<2 * 85 * 17>>) -> tensor<1x!ckks.poly<2 * 55 * 16>>
    %116 = "ckks.encode"() <{payload = dense<0.93680785180227221> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 17>>
    %117 = "ckks.mulcp"(%94, %116) : (tensor<1x!ckks.poly<2 * 45 * 17>>, tensor<1x!ckks.poly<1 * 40 * 17>>) -> tensor<1x!ckks.poly<2 * 85 * 17>>
    %118 = "ckks.rescalec"(%117) : (tensor<1x!ckks.poly<2 * 85 * 17>>) -> tensor<1x!ckks.poly<2 * 55 * 16>>
    %119 = "ckks.addcc"(%115, %118) : (tensor<1x!ckks.poly<2 * 55 * 16>>, tensor<1x!ckks.poly<2 * 55 * 16>>) -> tensor<1x!ckks.poly<2 * 55 * 16>>
    %120 = "ckks.addcc"(%101, %101) : (tensor<1x!ckks.poly<2 * 60 * 18>>, tensor<1x!ckks.poly<2 * 60 * 18>>) -> tensor<1x!ckks.poly<2 * 60 * 18>>
    %121 = "ckks.mulcc"(%120, %101) : (tensor<1x!ckks.poly<2 * 60 * 18>>, tensor<1x!ckks.poly<2 * 60 * 18>>) -> tensor<1x!ckks.poly<3 * 120 * 18>>
    %122 = "ckks.relinearize"(%121) : (tensor<1x!ckks.poly<3 * 120 * 18>>) -> tensor<1x!ckks.poly<2 * 120 * 18>>
    %123 = "ckks.rescalec"(%122) : (tensor<1x!ckks.poly<2 * 120 * 18>>) -> tensor<1x!ckks.poly<2 * 90 * 17>>
    %124 = "ckks.rescalec"(%123) : (tensor<1x!ckks.poly<2 * 90 * 17>>) -> tensor<1x!ckks.poly<2 * 60 * 16>>
    %125 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 16>>
    %126 = "ckks.addcp"(%124, %125) : (tensor<1x!ckks.poly<2 * 60 * 16>>, tensor<1x!ckks.poly<1 * 60 * 16>>) -> tensor<1x!ckks.poly<2 * 60 * 16>>
    %127 = "ckks.mulcc"(%119, %126) : (tensor<1x!ckks.poly<2 * 55 * 16>>, tensor<1x!ckks.poly<2 * 60 * 16>>) -> tensor<1x!ckks.poly<3 * 115 * 16>>
    %128 = "ckks.relinearize"(%127) : (tensor<1x!ckks.poly<3 * 115 * 16>>) -> tensor<1x!ckks.poly<2 * 115 * 16>>
    %129 = "ckks.rescalec"(%128) : (tensor<1x!ckks.poly<2 * 115 * 16>>) -> tensor<1x!ckks.poly<2 * 85 * 15>>
    %130 = "ckks.rescalec"(%129) : (tensor<1x!ckks.poly<2 * 85 * 15>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %131 = "ckks.encode"() <{payload = dense<0.13312678038183748> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 15>>
    %132 = "ckks.mulcp"(%107, %131) : (tensor<1x!ckks.poly<2 * 45 * 15>>, tensor<1x!ckks.poly<1 * 40 * 15>>) -> tensor<1x!ckks.poly<2 * 85 * 15>>
    %133 = "ckks.rescalec"(%132) : (tensor<1x!ckks.poly<2 * 85 * 15>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %134 = "ckks.encode"() <{payload = dense<-0.45988074662238737> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 15>>
    %135 = "ckks.mulcp"(%109, %134) : (tensor<1x!ckks.poly<2 * 45 * 15>>, tensor<1x!ckks.poly<1 * 40 * 15>>) -> tensor<1x!ckks.poly<2 * 85 * 15>>
    %136 = "ckks.rescalec"(%135) : (tensor<1x!ckks.poly<2 * 85 * 15>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %137 = "ckks.addcc"(%133, %136) : (tensor<1x!ckks.poly<2 * 55 * 14>>, tensor<1x!ckks.poly<2 * 55 * 14>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %138 = "ckks.addcc"(%130, %137) : (tensor<1x!ckks.poly<2 * 55 * 14>>, tensor<1x!ckks.poly<2 * 55 * 14>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %139 = "ckks.addcc"(%126, %126) : (tensor<1x!ckks.poly<2 * 60 * 16>>, tensor<1x!ckks.poly<2 * 60 * 16>>) -> tensor<1x!ckks.poly<2 * 60 * 16>>
    %140 = "ckks.mulcc"(%139, %126) : (tensor<1x!ckks.poly<2 * 60 * 16>>, tensor<1x!ckks.poly<2 * 60 * 16>>) -> tensor<1x!ckks.poly<3 * 120 * 16>>
    %141 = "ckks.relinearize"(%140) : (tensor<1x!ckks.poly<3 * 120 * 16>>) -> tensor<1x!ckks.poly<2 * 120 * 16>>
    %142 = "ckks.rescalec"(%141) : (tensor<1x!ckks.poly<2 * 120 * 16>>) -> tensor<1x!ckks.poly<2 * 90 * 15>>
    %143 = "ckks.rescalec"(%142) : (tensor<1x!ckks.poly<2 * 90 * 15>>) -> tensor<1x!ckks.poly<2 * 60 * 14>>
    %144 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 14>>
    %145 = "ckks.addcp"(%143, %144) : (tensor<1x!ckks.poly<2 * 60 * 14>>, tensor<1x!ckks.poly<1 * 60 * 14>>) -> tensor<1x!ckks.poly<2 * 60 * 14>>
    %146 = "ckks.mulcc"(%138, %145) : (tensor<1x!ckks.poly<2 * 55 * 14>>, tensor<1x!ckks.poly<2 * 60 * 14>>) -> tensor<1x!ckks.poly<3 * 115 * 14>>
    %147 = "ckks.relinearize"(%146) : (tensor<1x!ckks.poly<3 * 115 * 14>>) -> tensor<1x!ckks.poly<2 * 115 * 14>>
    %148 = "ckks.rescalec"(%147) : (tensor<1x!ckks.poly<2 * 115 * 14>>) -> tensor<1x!ckks.poly<2 * 85 * 13>>
    %149 = "ckks.rescalec"(%148) : (tensor<1x!ckks.poly<2 * 85 * 13>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %150 = "ckks.encode"() <{payload = dense<-0.058402193560527074> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 15>>
    %151 = "ckks.mulcp"(%107, %150) : (tensor<1x!ckks.poly<2 * 45 * 15>>, tensor<1x!ckks.poly<1 * 40 * 15>>) -> tensor<1x!ckks.poly<2 * 85 * 15>>
    %152 = "ckks.rescalec"(%151) : (tensor<1x!ckks.poly<2 * 85 * 15>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %153 = "ckks.encode"() <{payload = dense<-0.56048531218535647> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 15>>
    %154 = "ckks.mulcp"(%109, %153) : (tensor<1x!ckks.poly<2 * 45 * 15>>, tensor<1x!ckks.poly<1 * 40 * 15>>) -> tensor<1x!ckks.poly<2 * 85 * 15>>
    %155 = "ckks.rescalec"(%154) : (tensor<1x!ckks.poly<2 * 85 * 15>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %156 = "ckks.addcc"(%152, %155) : (tensor<1x!ckks.poly<2 * 55 * 14>>, tensor<1x!ckks.poly<2 * 55 * 14>>) -> tensor<1x!ckks.poly<2 * 55 * 14>>
    %157 = "ckks.modswitchc"(%126) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 60 * 16>>) -> tensor<1x!ckks.poly<2 * 60 * 14>>
    %158 = "ckks.mulcc"(%156, %157) : (tensor<1x!ckks.poly<2 * 55 * 14>>, tensor<1x!ckks.poly<2 * 60 * 14>>) -> tensor<1x!ckks.poly<3 * 115 * 14>>
    %159 = "ckks.relinearize"(%158) : (tensor<1x!ckks.poly<3 * 115 * 14>>) -> tensor<1x!ckks.poly<2 * 115 * 14>>
    %160 = "ckks.rescalec"(%159) : (tensor<1x!ckks.poly<2 * 115 * 14>>) -> tensor<1x!ckks.poly<2 * 85 * 13>>
    %161 = "ckks.rescalec"(%160) : (tensor<1x!ckks.poly<2 * 85 * 13>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %162 = "ckks.encode"() <{payload = dense<0.77658111917095896> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 13>>
    %163 = "ckks.modswitchc"(%93) <{downFactor = 6 : i64}> : (tensor<1x!ckks.poly<2 * 45 * 19>>) -> tensor<1x!ckks.poly<2 * 45 * 13>>
    %164 = "ckks.mulcp"(%163, %162) : (tensor<1x!ckks.poly<2 * 45 * 13>>, tensor<1x!ckks.poly<1 * 40 * 13>>) -> tensor<1x!ckks.poly<2 * 85 * 13>>
    %165 = "ckks.rescalec"(%164) : (tensor<1x!ckks.poly<2 * 85 * 13>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %166 = "ckks.encode"() <{payload = dense<-0.68184097760015783> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 13>>
    %167 = "ckks.modswitchc"(%109) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 45 * 15>>) -> tensor<1x!ckks.poly<2 * 45 * 13>>
    %168 = "ckks.mulcp"(%167, %166) : (tensor<1x!ckks.poly<2 * 45 * 13>>, tensor<1x!ckks.poly<1 * 40 * 13>>) -> tensor<1x!ckks.poly<2 * 85 * 13>>
    %169 = "ckks.rescalec"(%168) : (tensor<1x!ckks.poly<2 * 85 * 13>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %170 = "ckks.addcc"(%165, %169) : (tensor<1x!ckks.poly<2 * 55 * 12>>, tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %171 = "ckks.addcc"(%161, %170) : (tensor<1x!ckks.poly<2 * 55 * 12>>, tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %172 = "ckks.addcc"(%149, %171) : (tensor<1x!ckks.poly<2 * 55 * 12>>, tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %173 = "ckks.modswitchc"(%172) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 10>>
    %174 = "ckks.addcc"(%173, %173) : (tensor<1x!ckks.poly<2 * 55 * 10>>, tensor<1x!ckks.poly<2 * 55 * 10>>) -> tensor<1x!ckks.poly<2 * 55 * 10>>
    %175 = "ckks.addcc"(%172, %172) : (tensor<1x!ckks.poly<2 * 55 * 12>>, tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 12>>
    %176 = "ckks.mulcc"(%175, %172) : (tensor<1x!ckks.poly<2 * 55 * 12>>, tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<3 * 110 * 12>>
    %177 = "ckks.relinearize"(%176) : (tensor<1x!ckks.poly<3 * 110 * 12>>) -> tensor<1x!ckks.poly<2 * 110 * 12>>
    %178 = "ckks.rescalec"(%177) : (tensor<1x!ckks.poly<2 * 110 * 12>>) -> tensor<1x!ckks.poly<2 * 80 * 11>>
    %179 = "ckks.rescalec"(%178) : (tensor<1x!ckks.poly<2 * 80 * 11>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %180 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 10>>
    %181 = "ckks.addcp"(%179, %180) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<1 * 50 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %182 = "ckks.mulcc"(%174, %181) : (tensor<1x!ckks.poly<2 * 55 * 10>>, tensor<1x!ckks.poly<2 * 50 * 10>>) -> tensor<1x!ckks.poly<3 * 105 * 10>>
    %183 = "ckks.relinearize"(%182) : (tensor<1x!ckks.poly<3 * 105 * 10>>) -> tensor<1x!ckks.poly<2 * 105 * 10>>
    %184 = "ckks.rescalec"(%183) : (tensor<1x!ckks.poly<2 * 105 * 10>>) -> tensor<1x!ckks.poly<2 * 75 * 9>>
    %185 = "ckks.rescalec"(%184) : (tensor<1x!ckks.poly<2 * 75 * 9>>) -> tensor<1x!ckks.poly<2 * 45 * 8>>
    %186 = "ckks.modswitchc"(%172) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 8>>
    %187 = "ckks.negatec"(%186) : (tensor<1x!ckks.poly<2 * 55 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 8>>
    %188 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 8>>
    %189 = "ckks.mulcp"(%185, %188) : (tensor<1x!ckks.poly<2 * 45 * 8>>, tensor<1x!ckks.poly<1 * 10 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 8>>
    %190 = "ckks.addcc"(%189, %187) : (tensor<1x!ckks.poly<2 * 55 * 8>>, tensor<1x!ckks.poly<2 * 55 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 8>>
    %191 = "ckks.addcc"(%186, %186) : (tensor<1x!ckks.poly<2 * 55 * 8>>, tensor<1x!ckks.poly<2 * 55 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 8>>
    %192 = "ckks.addcc"(%181, %181) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<2 * 50 * 10>>) -> tensor<1x!ckks.poly<2 * 50 * 10>>
    %193 = "ckks.mulcc"(%192, %181) : (tensor<1x!ckks.poly<2 * 50 * 10>>, tensor<1x!ckks.poly<2 * 50 * 10>>) -> tensor<1x!ckks.poly<3 * 100 * 10>>
    %194 = "ckks.relinearize"(%193) : (tensor<1x!ckks.poly<3 * 100 * 10>>) -> tensor<1x!ckks.poly<2 * 100 * 10>>
    %195 = "ckks.rescalec"(%194) : (tensor<1x!ckks.poly<2 * 100 * 10>>) -> tensor<1x!ckks.poly<2 * 70 * 9>>
    %196 = "ckks.rescalec"(%195) : (tensor<1x!ckks.poly<2 * 70 * 9>>) -> tensor<1x!ckks.poly<2 * 40 * 8>>
    %197 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 8>>
    %198 = "ckks.addcp"(%196, %197) : (tensor<1x!ckks.poly<2 * 40 * 8>>, tensor<1x!ckks.poly<1 * 40 * 8>>) -> tensor<1x!ckks.poly<2 * 40 * 8>>
    %199 = "ckks.mulcc"(%191, %198) : (tensor<1x!ckks.poly<2 * 55 * 8>>, tensor<1x!ckks.poly<2 * 40 * 8>>) -> tensor<1x!ckks.poly<3 * 95 * 8>>
    %200 = "ckks.relinearize"(%199) : (tensor<1x!ckks.poly<3 * 95 * 8>>) -> tensor<1x!ckks.poly<2 * 95 * 8>>
    %201 = "ckks.rescalec"(%200) : (tensor<1x!ckks.poly<2 * 95 * 8>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %202 = "ckks.addcc"(%190, %190) : (tensor<1x!ckks.poly<2 * 55 * 8>>, tensor<1x!ckks.poly<2 * 55 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 8>>
    %203 = "ckks.mulcc"(%202, %198) : (tensor<1x!ckks.poly<2 * 55 * 8>>, tensor<1x!ckks.poly<2 * 40 * 8>>) -> tensor<1x!ckks.poly<3 * 95 * 8>>
    %204 = "ckks.relinearize"(%203) : (tensor<1x!ckks.poly<3 * 95 * 8>>) -> tensor<1x!ckks.poly<2 * 95 * 8>>
    %205 = "ckks.rescalec"(%204) : (tensor<1x!ckks.poly<2 * 95 * 8>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %206 = "ckks.modswitchc"(%190) <{downFactor = 1 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 7>>
    %207 = "ckks.negatec"(%206) : (tensor<1x!ckks.poly<2 * 55 * 7>>) -> tensor<1x!ckks.poly<2 * 55 * 7>>
    %208 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 7>>
    %209 = "ckks.mulcp"(%207, %208) : (tensor<1x!ckks.poly<2 * 55 * 7>>, tensor<1x!ckks.poly<1 * 10 * 7>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %210 = "ckks.addcc"(%201, %209) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<2 * 65 * 7>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %211 = "ckks.modswitchc"(%172) <{downFactor = 5 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 7>>
    %212 = "ckks.negatec"(%211) : (tensor<1x!ckks.poly<2 * 55 * 7>>) -> tensor<1x!ckks.poly<2 * 55 * 7>>
    %213 = "ckks.mulcp"(%212, %208) : (tensor<1x!ckks.poly<2 * 55 * 7>>, tensor<1x!ckks.poly<1 * 10 * 7>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %214 = "ckks.addcc"(%205, %213) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<2 * 65 * 7>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %215 = "ckks.encode"() <{payload = dense<-0.017732734281918773> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 8>>
    %216 = "ckks.mulcp"(%186, %215) : (tensor<1x!ckks.poly<2 * 55 * 8>>, tensor<1x!ckks.poly<1 * 40 * 8>>) -> tensor<1x!ckks.poly<2 * 95 * 8>>
    %217 = "ckks.rescalec"(%216) : (tensor<1x!ckks.poly<2 * 95 * 8>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %218 = "ckks.mulcp"(%190, %3) : (tensor<1x!ckks.poly<2 * 55 * 8>>, tensor<1x!ckks.poly<1 * 40 * 8>>) -> tensor<1x!ckks.poly<2 * 95 * 8>>
    %219 = "ckks.rescalec"(%218) : (tensor<1x!ckks.poly<2 * 95 * 8>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %220 = "ckks.addcc"(%217, %219) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<2 * 65 * 7>>) -> tensor<1x!ckks.poly<2 * 65 * 7>>
    %221 = "ckks.addcc"(%198, %198) : (tensor<1x!ckks.poly<2 * 40 * 8>>, tensor<1x!ckks.poly<2 * 40 * 8>>) -> tensor<1x!ckks.poly<2 * 40 * 8>>
    %222 = "ckks.mulcc"(%221, %198) : (tensor<1x!ckks.poly<2 * 40 * 8>>, tensor<1x!ckks.poly<2 * 40 * 8>>) -> tensor<1x!ckks.poly<3 * 80 * 8>>
    %223 = "ckks.relinearize"(%222) : (tensor<1x!ckks.poly<3 * 80 * 8>>) -> tensor<1x!ckks.poly<2 * 80 * 8>>
    %224 = "ckks.rescalec"(%223) : (tensor<1x!ckks.poly<2 * 80 * 8>>) -> tensor<1x!ckks.poly<2 * 50 * 7>>
    %225 = "ckks.encode"() <{payload = dense<-1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 50 * 7>>
    %226 = "ckks.addcp"(%224, %225) : (tensor<1x!ckks.poly<2 * 50 * 7>>, tensor<1x!ckks.poly<1 * 50 * 7>>) -> tensor<1x!ckks.poly<2 * 50 * 7>>
    %227 = "ckks.mulcc"(%220, %226) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<2 * 50 * 7>>) -> tensor<1x!ckks.poly<3 * 115 * 7>>
    %228 = "ckks.relinearize"(%227) : (tensor<1x!ckks.poly<3 * 115 * 7>>) -> tensor<1x!ckks.poly<2 * 115 * 7>>
    %229 = "ckks.rescalec"(%228) : (tensor<1x!ckks.poly<2 * 115 * 7>>) -> tensor<1x!ckks.poly<2 * 85 * 6>>
    %230 = "ckks.rescalec"(%229) : (tensor<1x!ckks.poly<2 * 85 * 6>>) -> tensor<1x!ckks.poly<2 * 55 * 5>>
    %231 = "ckks.encode"() <{payload = dense<-0.19732584204806608> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 6>>
    %232 = "ckks.modswitchc"(%172) <{downFactor = 6 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 6>>
    %233 = "ckks.mulcp"(%232, %231) : (tensor<1x!ckks.poly<2 * 55 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %234 = "ckks.rescalec"(%233) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %235 = "ckks.encode"() <{payload = dense<-0.1411166195583439> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 6>>
    %236 = "ckks.modswitchc"(%190) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 6>>
    %237 = "ckks.mulcp"(%236, %235) : (tensor<1x!ckks.poly<2 * 55 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %238 = "ckks.rescalec"(%237) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %239 = "ckks.encode"() <{payload = dense<-0.052675736911616873> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 7>>
    %240 = "ckks.mulcp"(%210, %239) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<1 * 60 * 7>>) -> tensor<1x!ckks.poly<2 * 125 * 7>>
    %241 = "ckks.rescalec"(%240) : (tensor<1x!ckks.poly<2 * 125 * 7>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %242 = "ckks.rescalec"(%241) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %243 = "ckks.encode"() <{payload = dense<-0.026312988130283786> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 7>>
    %244 = "ckks.mulcp"(%214, %243) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<1 * 60 * 7>>) -> tensor<1x!ckks.poly<2 * 125 * 7>>
    %245 = "ckks.rescalec"(%244) : (tensor<1x!ckks.poly<2 * 125 * 7>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %246 = "ckks.rescalec"(%245) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %247 = "ckks.addcc"(%234, %238) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 65 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %248 = "ckks.addcc"(%247, %242) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 65 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %249 = "ckks.addcc"(%248, %246) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 65 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %250 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 5>>
    %251 = "ckks.mulcp"(%230, %250) : (tensor<1x!ckks.poly<2 * 55 * 5>>, tensor<1x!ckks.poly<1 * 10 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %252 = "ckks.addcc"(%251, %249) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 65 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %253 = "ckks.addcc"(%226, %226) : (tensor<1x!ckks.poly<2 * 50 * 7>>, tensor<1x!ckks.poly<2 * 50 * 7>>) -> tensor<1x!ckks.poly<2 * 50 * 7>>
    %254 = "ckks.mulcc"(%253, %226) : (tensor<1x!ckks.poly<2 * 50 * 7>>, tensor<1x!ckks.poly<2 * 50 * 7>>) -> tensor<1x!ckks.poly<3 * 100 * 7>>
    %255 = "ckks.relinearize"(%254) : (tensor<1x!ckks.poly<3 * 100 * 7>>) -> tensor<1x!ckks.poly<2 * 100 * 7>>
    %256 = "ckks.rescalec"(%255) : (tensor<1x!ckks.poly<2 * 100 * 7>>) -> tensor<1x!ckks.poly<2 * 70 * 6>>
    %257 = "ckks.rescalec"(%256) : (tensor<1x!ckks.poly<2 * 70 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %258 = "ckks.addcp"(%257, %1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<1 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %259 = "ckks.mulcc"(%252, %258) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<3 * 105 * 5>>
    %260 = "ckks.relinearize"(%259) : (tensor<1x!ckks.poly<3 * 105 * 5>>) -> tensor<1x!ckks.poly<2 * 105 * 5>>
    %261 = "ckks.rescalec"(%260) : (tensor<1x!ckks.poly<2 * 105 * 5>>) -> tensor<1x!ckks.poly<2 * 75 * 4>>
    %262 = "ckks.rescalec"(%261) : (tensor<1x!ckks.poly<2 * 75 * 4>>) -> tensor<1x!ckks.poly<2 * 45 * 3>>
    %263 = "ckks.encode"() <{payload = dense<-0.78912296503546786> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 6>>
    %264 = "ckks.mulcp"(%232, %263) : (tensor<1x!ckks.poly<2 * 55 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %265 = "ckks.rescalec"(%264) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %266 = "ckks.encode"() <{payload = dense<-0.72741137353811158> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 6>>
    %267 = "ckks.mulcp"(%236, %266) : (tensor<1x!ckks.poly<2 * 55 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %268 = "ckks.rescalec"(%267) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %269 = "ckks.encode"() <{payload = dense<-0.32554529142034655> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 7>>
    %270 = "ckks.mulcp"(%210, %269) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<1 * 60 * 7>>) -> tensor<1x!ckks.poly<2 * 125 * 7>>
    %271 = "ckks.rescalec"(%270) : (tensor<1x!ckks.poly<2 * 125 * 7>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %272 = "ckks.rescalec"(%271) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %273 = "ckks.encode"() <{payload = dense<-0.18601751738208003> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 7>>
    %274 = "ckks.mulcp"(%214, %273) : (tensor<1x!ckks.poly<2 * 65 * 7>>, tensor<1x!ckks.poly<1 * 60 * 7>>) -> tensor<1x!ckks.poly<2 * 125 * 7>>
    %275 = "ckks.rescalec"(%274) : (tensor<1x!ckks.poly<2 * 125 * 7>>) -> tensor<1x!ckks.poly<2 * 95 * 6>>
    %276 = "ckks.rescalec"(%275) : (tensor<1x!ckks.poly<2 * 95 * 6>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %277 = "ckks.addcc"(%265, %268) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 65 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %278 = "ckks.addcc"(%277, %272) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 65 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %279 = "ckks.addcc"(%278, %276) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 65 * 5>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %280 = "ckks.modswitchc"(%226) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 50 * 7>>) -> tensor<1x!ckks.poly<2 * 50 * 5>>
    %281 = "ckks.mulcc"(%279, %280) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<2 * 50 * 5>>) -> tensor<1x!ckks.poly<3 * 115 * 5>>
    %282 = "ckks.relinearize"(%281) : (tensor<1x!ckks.poly<3 * 115 * 5>>) -> tensor<1x!ckks.poly<2 * 115 * 5>>
    %283 = "ckks.rescalec"(%282) : (tensor<1x!ckks.poly<2 * 115 * 5>>) -> tensor<1x!ckks.poly<2 * 85 * 4>>
    %284 = "ckks.rescalec"(%283) : (tensor<1x!ckks.poly<2 * 85 * 4>>) -> tensor<1x!ckks.poly<2 * 55 * 3>>
    %285 = "ckks.encode"() <{payload = dense<-0.042857894008498022> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 4>>
    %286 = "ckks.modswitchc"(%172) <{downFactor = 8 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 12>>) -> tensor<1x!ckks.poly<2 * 55 * 4>>
    %287 = "ckks.mulcp"(%286, %285) : (tensor<1x!ckks.poly<2 * 55 * 4>>, tensor<1x!ckks.poly<1 * 40 * 4>>) -> tensor<1x!ckks.poly<2 * 95 * 4>>
    %288 = "ckks.rescalec"(%287) : (tensor<1x!ckks.poly<2 * 95 * 4>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %289 = "ckks.encode"() <{payload = dense<-0.7747926099066248> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 4>>
    %290 = "ckks.modswitchc"(%190) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 55 * 8>>) -> tensor<1x!ckks.poly<2 * 55 * 4>>
    %291 = "ckks.mulcp"(%290, %289) : (tensor<1x!ckks.poly<2 * 55 * 4>>, tensor<1x!ckks.poly<1 * 40 * 4>>) -> tensor<1x!ckks.poly<2 * 95 * 4>>
    %292 = "ckks.rescalec"(%291) : (tensor<1x!ckks.poly<2 * 95 * 4>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %293 = "ckks.modswitchc"(%210) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 65 * 7>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %294 = "ckks.encode"() <{payload = dense<-0.18128174370012556> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 5>>
    %295 = "ckks.mulcp"(%293, %294) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<1 * 60 * 5>>) -> tensor<1x!ckks.poly<2 * 125 * 5>>
    %296 = "ckks.rescalec"(%295) : (tensor<1x!ckks.poly<2 * 125 * 5>>) -> tensor<1x!ckks.poly<2 * 95 * 4>>
    %297 = "ckks.rescalec"(%296) : (tensor<1x!ckks.poly<2 * 95 * 4>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %298 = "ckks.modswitchc"(%214) <{downFactor = 2 : i64}> : (tensor<1x!ckks.poly<2 * 65 * 7>>) -> tensor<1x!ckks.poly<2 * 65 * 5>>
    %299 = "ckks.encode"() <{payload = dense<-0.24546846913474671> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 60 * 5>>
    %300 = "ckks.mulcp"(%298, %299) : (tensor<1x!ckks.poly<2 * 65 * 5>>, tensor<1x!ckks.poly<1 * 60 * 5>>) -> tensor<1x!ckks.poly<2 * 125 * 5>>
    %301 = "ckks.rescalec"(%300) : (tensor<1x!ckks.poly<2 * 125 * 5>>) -> tensor<1x!ckks.poly<2 * 95 * 4>>
    %302 = "ckks.rescalec"(%301) : (tensor<1x!ckks.poly<2 * 95 * 4>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %303 = "ckks.addcc"(%288, %292) : (tensor<1x!ckks.poly<2 * 65 * 3>>, tensor<1x!ckks.poly<2 * 65 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %304 = "ckks.addcc"(%303, %297) : (tensor<1x!ckks.poly<2 * 65 * 3>>, tensor<1x!ckks.poly<2 * 65 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %305 = "ckks.addcc"(%304, %302) : (tensor<1x!ckks.poly<2 * 65 * 3>>, tensor<1x!ckks.poly<2 * 65 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %306 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 10 * 3>>
    %307 = "ckks.mulcp"(%284, %306) : (tensor<1x!ckks.poly<2 * 55 * 3>>, tensor<1x!ckks.poly<1 * 10 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %308 = "ckks.addcc"(%307, %305) : (tensor<1x!ckks.poly<2 * 65 * 3>>, tensor<1x!ckks.poly<2 * 65 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %309 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 20 * 3>>
    %310 = "ckks.mulcp"(%262, %309) : (tensor<1x!ckks.poly<2 * 45 * 3>>, tensor<1x!ckks.poly<1 * 20 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %311 = "ckks.addcc"(%310, %308) : (tensor<1x!ckks.poly<2 * 65 * 3>>, tensor<1x!ckks.poly<2 * 65 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %312 = "ckks.encode"() <{payload = dense<5.000000e-01> : tensor<32768xf64>}> : () -> tensor<1x!ckks.poly<1 * 65 * 3>>
    %313 = "ckks.addcp"(%311, %312) : (tensor<1x!ckks.poly<2 * 65 * 3>>, tensor<1x!ckks.poly<1 * 65 * 3>>) -> tensor<1x!ckks.poly<2 * 65 * 3>>
    %314 = "ckks.modswitchc"(%arg0) <{downFactor = 16 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 19>>) -> tensor<1x!ckks.poly<2 * 40 * 3>>
    %315 = "ckks.mulcc"(%314, %313) : (tensor<1x!ckks.poly<2 * 40 * 3>>, tensor<1x!ckks.poly<2 * 65 * 3>>) -> tensor<1x!ckks.poly<3 * 105 * 3>>
    %316 = "ckks.relinearize"(%315) : (tensor<1x!ckks.poly<3 * 105 * 3>>) -> tensor<1x!ckks.poly<2 * 105 * 3>>
    %317 = "ckks.rescalec"(%316) : (tensor<1x!ckks.poly<2 * 105 * 3>>) -> tensor<1x!ckks.poly<2 * 75 * 2>>
    %318 = "ckks.rescalec"(%317) : (tensor<1x!ckks.poly<2 * 75 * 2>>) -> tensor<1x!ckks.poly<2 * 45 * 1>>
    return %318 : tensor<1x!ckks.poly<2 * 45 * 1>>
  }
}

