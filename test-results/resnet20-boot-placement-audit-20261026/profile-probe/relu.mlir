module { func.func @relu(%x: tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>> {
%p0 = "earth.constant"() {value = dense<-1.82077307001662858e-27> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p1 = "earth.constant"() {value = dense<1.31101564708907459e+00> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p2 = "earth.constant"() {value = dense<-1.47147232252902143e-26> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p3 = "earth.constant"() {value = dense<-4.65544458220611412e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p4 = "earth.constant"() {value = dense<-7.72265086164312448e-27> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p5 = "earth.constant"() {value = dense<3.88788941781359310e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p6 = "earth.constant"() {value = dense<1.15972634077791380e-26> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p7 = "earth.constant"() {value = dense<-3.41031822722021316e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p8 = "earth.constant"() {value = dense<6.52333109285148246e-28> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p9 = "earth.constant"() {value = dense<1.15837082052129881e+00> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p10 = "earth.constant"() {value = dense<-2.60793313686974664e-26> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p11 = "earth.constant"() {value = dense<-2.43786140146776159e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p12 = "earth.constant"() {value = dense<3.66961977847983118e-27> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p13 = "earth.constant"() {value = dense<2.24318797852664797e+00> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p14 = "earth.constant"() {value = dense<2.91961270406305950e-26> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p15 = "earth.constant"() {value = dense<-4.02702012006362331e+00> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p16 = "earth.constant"() {value = dense<2.00000000000000000e+00> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p17 = "earth.constant"() {value = dense<-1.00000000000000000e+00> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p18 = "earth.constant"() {value = dense<-4.85029309884266865e-28> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p19 = "earth.constant"() {value = dense<7.76581119170958956e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p20 = "earth.constant"() {value = dense<3.56319962570844687e-28> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p21 = "earth.constant"() {value = dense<-6.81840977600157827e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p22 = "earth.constant"() {value = dense<-9.70147497432007666e-29> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p23 = "earth.constant"() {value = dense<-5.84021935605270742e-02> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p24 = "earth.constant"() {value = dense<-8.80028842543061656e-28> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p25 = "earth.constant"() {value = dense<-5.60485312185356466e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p26 = "earth.constant"() {value = dense<-9.09119858554532056e-28> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p27 = "earth.constant"() {value = dense<1.33126780381837478e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p28 = "earth.constant"() {value = dense<7.09743090221869181e-28> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p29 = "earth.constant"() {value = dense<-4.59880746622387371e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p30 = "earth.constant"() {value = dense<-1.43997125156290869e-27> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p31 = "earth.constant"() {value = dense<9.36807851802272218e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p32 = "earth.constant"() {value = dense<-3.30247344556268846e-27> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p33 = "earth.constant"() {value = dense<-4.11442516140391490e+00> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p34 = "earth.constant"() {value = dense<-9.69936050144544457e-32> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p35 = "earth.constant"() {value = dense<-4.28578940084980223e-02> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p36 = "earth.constant"() {value = dense<-1.58254108252753645e-31> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p37 = "earth.constant"() {value = dense<-7.74792609906624796e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p38 = "earth.constant"() {value = dense<-1.11928276880738053e-31> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p39 = "earth.constant"() {value = dense<-1.81281743700125564e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p40 = "earth.constant"() {value = dense<-5.82415298844679404e-32> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p41 = "earth.constant"() {value = dense<-2.45468469134746708e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p42 = "earth.constant"() {value = dense<-1.24497055066859506e-31> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p43 = "earth.constant"() {value = dense<-7.89122965035467860e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p44 = "earth.constant"() {value = dense<-1.88674582525058680e-31> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p45 = "earth.constant"() {value = dense<-7.27411373538111583e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p46 = "earth.constant"() {value = dense<-1.25075742379997299e-31> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p47 = "earth.constant"() {value = dense<-3.25545291420346550e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p48 = "earth.constant"() {value = dense<-6.24818705275061301e-32> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p49 = "earth.constant"() {value = dense<-1.86017517382080028e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p50 = "earth.constant"() {value = dense<-3.10710833208513908e-32> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p51 = "earth.constant"() {value = dense<-1.97325842048066080e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p52 = "earth.constant"() {value = dense<-3.62916286281001505e-32> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p53 = "earth.constant"() {value = dense<-1.41116619558343898e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p54 = "earth.constant"() {value = dense<-1.88004724918971328e-32> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p55 = "earth.constant"() {value = dense<-5.26757369116168730e-02> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p56 = "earth.constant"() {value = dense<-7.72303060271554293e-33> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p57 = "earth.constant"() {value = dense<-2.63129881302837862e-02> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p58 = "earth.constant"() {value = dense<-3.34096028461656420e-33> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p59 = "earth.constant"() {value = dense<-1.77327342819187730e-02> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p60 = "earth.constant"() {value = dense<-2.06890324902290744e-33> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%p61 = "earth.constant"() {value = dense<-9.08409434411346257e-03> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%v62 = "earth.mul"(%p16,%x) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v63 = "earth.mul"(%p16,%x) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v64 = "earth.mul"(%v63,%x) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v65 = "earth.add"(%v64,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v66 = "earth.mul"(%v62,%v65) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v67 = "earth.negate"(%x) : (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v68 = "earth.add"(%v66,%v67) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v69 = "earth.mul"(%p15,%x) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v70 = "earth.mul"(%v69,%v65) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v71 = "earth.mul"(%p13,%x) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v72 = "earth.add"(%v70,%v71) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v73 = "earth.mul"(%p16,%v65) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v74 = "earth.mul"(%v73,%v65) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v75 = "earth.add"(%v74,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v76 = "earth.mul"(%v72,%v75) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v77 = "earth.mul"(%p9,%x) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v78 = "earth.mul"(%p11,%v68) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v79 = "earth.add"(%v77,%v78) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v80 = "earth.add"(%v76,%v79) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v81 = "earth.mul"(%p16,%v75) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v82 = "earth.mul"(%v81,%v75) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v83 = "earth.add"(%v82,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v84 = "earth.mul"(%v80,%v83) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v85 = "earth.mul"(%p5,%x) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v86 = "earth.mul"(%p7,%v68) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v87 = "earth.add"(%v85,%v86) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v88 = "earth.mul"(%v87,%v75) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v89 = "earth.mul"(%p1,%x) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v90 = "earth.mul"(%p3,%v68) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v91 = "earth.add"(%v89,%v90) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v92 = "earth.add"(%v88,%v91) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v93 = "earth.add"(%v84,%v92) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v94 = "earth.mul"(%p16,%v93) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v95 = "earth.mul"(%p16,%v93) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v96 = "earth.mul"(%v95,%v93) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v97 = "earth.add"(%v96,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v98 = "earth.mul"(%v94,%v97) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v99 = "earth.negate"(%v93) : (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v100 = "earth.add"(%v98,%v99) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v101 = "earth.mul"(%p33,%v93) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v102 = "earth.mul"(%v101,%v97) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v103 = "earth.mul"(%p31,%v93) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v104 = "earth.add"(%v102,%v103) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v105 = "earth.mul"(%p16,%v97) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v106 = "earth.mul"(%v105,%v97) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v107 = "earth.add"(%v106,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v108 = "earth.mul"(%v104,%v107) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v109 = "earth.mul"(%p27,%v93) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v110 = "earth.mul"(%p29,%v100) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v111 = "earth.add"(%v109,%v110) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v112 = "earth.add"(%v108,%v111) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v113 = "earth.mul"(%p16,%v107) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v114 = "earth.mul"(%v113,%v107) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v115 = "earth.add"(%v114,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v116 = "earth.mul"(%v112,%v115) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v117 = "earth.mul"(%p23,%v93) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v118 = "earth.mul"(%p25,%v100) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v119 = "earth.add"(%v117,%v118) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v120 = "earth.mul"(%v119,%v107) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v121 = "earth.mul"(%p19,%v93) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v122 = "earth.mul"(%p21,%v100) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v123 = "earth.add"(%v121,%v122) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v124 = "earth.add"(%v120,%v123) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v125 = "earth.add"(%v116,%v124) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v126 = "earth.mul"(%p16,%v125) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v127 = "earth.mul"(%p16,%v125) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v128 = "earth.mul"(%v127,%v125) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v129 = "earth.add"(%v128,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v130 = "earth.mul"(%v126,%v129) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v131 = "earth.negate"(%v125) : (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v132 = "earth.add"(%v130,%v131) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v133 = "earth.mul"(%p16,%v125) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v134 = "earth.mul"(%p16,%v129) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v135 = "earth.mul"(%v134,%v129) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v136 = "earth.add"(%v135,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v137 = "earth.mul"(%v133,%v136) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v138 = "earth.mul"(%p16,%v132) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v139 = "earth.mul"(%v138,%v136) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v140 = "earth.negate"(%v132) : (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v141 = "earth.add"(%v137,%v140) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v142 = "earth.negate"(%v125) : (tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v143 = "earth.add"(%v139,%v142) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v144 = "earth.mul"(%p59,%v125) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v145 = "earth.mul"(%p61,%v132) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v146 = "earth.add"(%v144,%v145) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v147 = "earth.mul"(%p16,%v136) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v148 = "earth.mul"(%v147,%v136) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v149 = "earth.add"(%v148,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v150 = "earth.mul"(%v146,%v149) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v151 = "earth.mul"(%p51,%v125) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v152 = "earth.mul"(%p53,%v132) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v153 = "earth.mul"(%p55,%v141) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v154 = "earth.mul"(%p57,%v143) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v155 = "earth.add"(%v151,%v152) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v156 = "earth.add"(%v155,%v153) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v157 = "earth.add"(%v156,%v154) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v158 = "earth.add"(%v150,%v157) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v159 = "earth.mul"(%p16,%v149) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v160 = "earth.mul"(%v159,%v149) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v161 = "earth.add"(%v160,%p17) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v162 = "earth.mul"(%v158,%v161) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v163 = "earth.mul"(%p43,%v125) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v164 = "earth.mul"(%p45,%v132) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v165 = "earth.mul"(%p47,%v141) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v166 = "earth.mul"(%p49,%v143) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v167 = "earth.add"(%v163,%v164) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v168 = "earth.add"(%v167,%v165) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v169 = "earth.add"(%v168,%v166) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v170 = "earth.mul"(%v169,%v149) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v171 = "earth.mul"(%p35,%v125) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v172 = "earth.mul"(%p37,%v132) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v173 = "earth.mul"(%p39,%v141) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v174 = "earth.mul"(%p41,%v143) : (tensor<1x!earth.pl<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v175 = "earth.add"(%v171,%v172) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v176 = "earth.add"(%v175,%v173) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v177 = "earth.add"(%v176,%v174) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v178 = "earth.add"(%v170,%v177) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v179 = "earth.add"(%v162,%v178) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%p62 = "earth.constant"() {value = dense<5.00000000000000000e-01> : tensor<32768xf64>, rms_var = 1.0 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%v181 = "earth.add"(%v179,%p62) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v182 = "earth.mul"(%x,%v181) : (tensor<1x!earth.ci<0 * 0>>,tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %v182 : tensor<1x!earth.ci<0 * 0>>
}}
