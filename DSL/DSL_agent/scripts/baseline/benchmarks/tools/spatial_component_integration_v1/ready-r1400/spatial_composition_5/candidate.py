@hc.func('c,c,c,c')
def golden(x0, x1, x2, zero_ct):
    v238 = x0 * pv0 + x0.rotate(2) * pv1 + x0.rotate(1).rotate(2) * pv2 + x1 * pv24 + x1.rotate(1) * pv25 + x1.rotate(2) * pv26 + x2.rotate(1) * pv48 + pv72
    v239 = x0 * pv3 + x0.rotate(1) * pv4 + x0.rotate(2) * pv5 + x1 * pv27 + x1.rotate(1) * pv28 + x1.rotate(1).rotate(2) * pv29 + x2 * pv49 + x2.rotate(2) * pv50 + x2.rotate(1).rotate(2) * pv51 + pv72
    v240 = x0 * pv6 + x0.rotate(1) * pv7 + x0.rotate(1).rotate(2) * pv8 + x1 * pv30 + x1.rotate(2) * pv31 + x1.rotate(1).rotate(2) * pv32 + x2 * pv50 + x2.rotate(1) * pv52 + x2.rotate(2) * pv53 + x2.rotate(1).rotate(2) * pv54 + pv72
    v241 = x0.rotate(2) * pv9 + x0.rotate(1).rotate(2) * pv10 + x1.rotate(1) * pv1 + x1.rotate(2) * pv33 + x1.rotate(1).rotate(2) * pv34 + x2 * pv55 + x2.rotate(1) * pv26 + x2.rotate(1).rotate(2) * pv24 + pv72
    v242 = x0 * pv8 + x0.rotate(1) * pv11 + x1 * pv9 + x1.rotate(1) * pv35 + x2.rotate(2) * pv56 + x2.rotate(1).rotate(2) * pv57 + pv72
    v243 = x1 * pv7 + x1.rotate(2) * pv8 + x1.rotate(1).rotate(2) * pv6 + x2 * pv29 + x2.rotate(1) * pv58 + x2.rotate(2) * pv59 + pv72
    v244 = x0 * pv12 + x0.rotate(2) * pv13 + x0.rotate(1).rotate(2) * pv14 + x1 * pv36 + x1.rotate(1) * pv37 + x1.rotate(2) * pv38 + x2.rotate(1) * pv60 + pv73
    v245 = x0 * pv15 + x0.rotate(1) * pv16 + x0.rotate(2) * pv17 + x1 * pv39 + x1.rotate(1) * pv40 + x1.rotate(1).rotate(2) * pv41 + x2 * pv61 + x2.rotate(2) * pv62 + x2.rotate(1).rotate(2) * pv63 + pv73
    v246 = x0 * pv18 + x0.rotate(1) * pv19 + x0.rotate(1).rotate(2) * pv20 + x1 * pv42 + x1.rotate(2) * pv43 + x1.rotate(1).rotate(2) * pv44 + x2 * pv62 + x2.rotate(1) * pv64 + x2.rotate(2) * pv65 + x2.rotate(1).rotate(2) * pv66 + pv73
    v247 = x0.rotate(2) * pv21 + x0.rotate(1).rotate(2) * pv22 + x1.rotate(1) * pv13 + x1.rotate(2) * pv45 + x1.rotate(1).rotate(2) * pv46 + x2 * pv67 + x2.rotate(1) * pv38 + x2.rotate(1).rotate(2) * pv36 + pv73
    v248 = x0 * pv20 + x0.rotate(1) * pv23 + x1 * pv21 + x1.rotate(1) * pv47 + x2.rotate(2) * pv68 + x2.rotate(1).rotate(2) * pv69 + pv73
    v249 = x1 * pv19 + x1.rotate(2) * pv20 + x1.rotate(1).rotate(2) * pv18 + x2 * pv41 + x2.rotate(1) * pv70 + x2.rotate(2) * pv71 + pv73
    v251 = v238 + v238.rotate(1)
    v256 = v239 + v239.rotate(1)
    v262 = v240 + v240.rotate(1)
    v268 = v241 + v241.rotate(1)
    v274 = v242 + v242.rotate(1)
    v280 = v243 + v243.rotate(1)
    v286 = v244 + v244.rotate(1)
    v292 = v245 + v245.rotate(1)
    v298 = v246 + v246.rotate(1)
    v304 = v247 + v247.rotate(1)
    v310 = v248 + v248.rotate(1)
    v316 = v249 + v249.rotate(1)
    return [v238, v239, v240, v241, v242, v243, v244, v245, v246, v247, v248, v249, (v251 + v251.rotate(2)) * pv74 + (v256 + v256.rotate(2)) * pv74 + (v262 + v262.rotate(2)) * pv74 + (v268 + v268.rotate(2)) * pv74 + (v274 + v274.rotate(2)) * pv74 + (v280 + v280.rotate(2)) * pv74 + (v286 + v286.rotate(2)) * pv75 + (v292 + v292.rotate(2)) * pv75 + (v298 + v298.rotate(2)) * pv75 + (v304 + v304.rotate(2)) * pv75 + (v310 + v310.rotate(2)) * pv75 + (v316 + v316.rotate(2)) * pv75]
