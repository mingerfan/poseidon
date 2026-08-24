module {
  func.func @mpi_gpu_fanout(
      %arg0: tensor<1x!ckks.poly<2 * 40 * 16>>)
      -> tensor<1x!ckks.poly<2 * 40 * 16>> {
    %n0 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n1 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n2 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n3 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n4 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n5 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n6 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n7 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n8 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %n9 = "ckks.negatec"(%arg0) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>

    %a0 = "ckks.addcc"(%n0, %n1) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %a1 = "ckks.addcc"(%n2, %n3) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %a2 = "ckks.addcc"(%n4, %n5) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %a3 = "ckks.addcc"(%n6, %n7) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %a4 = "ckks.addcc"(%n8, %n9) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>

    %b0 = "ckks.addcc"(%a0, %a1) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %b1 = "ckks.addcc"(%a2, %a3) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %c0 = "ckks.addcc"(%b0, %b1) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    %out = "ckks.addcc"(%c0, %a4) :
        (tensor<1x!ckks.poly<2 * 40 * 16>>,
         tensor<1x!ckks.poly<2 * 40 * 16>>)
        -> tensor<1x!ckks.poly<2 * 40 * 16>>
    return %out : tensor<1x!ckks.poly<2 * 40 * 16>>
  }
}
