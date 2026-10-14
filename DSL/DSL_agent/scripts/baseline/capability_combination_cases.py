"""Nine manual contexts for the explicitly opened BN/native composition."""
from benchmark_suite import Builder
from component_contract import prepare_task
from upstream_bn_candidate_cases import bn
from capability_combinations import NAME, HELPER_PROFILE, CONSTRUCTIONS
def cases():
    rows=[]
    for construction in CONSTRUCTIONS:
        for variant in range(3):
            b=Builder([(1,2)])
            h=bn(b,"input0")
            if variant==1:h=b.node("square",[h])
            if variant==2:h=b.node("add",[h,"input0"])
            if construction=="unified-scalar":
                h=b.node("add",[h,b.const([.125])])
            model=b.finish(h)
            prefix='@hc.func("c")\ndef invoke(v):\n    return HE_BN0(v)\n' if construction=="unified-native-call-scalar_cipher" else ''
            source=prefix+'@hc.func("c,c")\ndef golden(x,zero_ct):\n'
            if construction=="unified-native-call-scalar_cipher":source+='    h=invoke(x)\n'
            elif construction=="unified-native-copy":source+='    a=np.array([HE_BN0(x)],dtype=object)\n    b=a.copy()\n    h=b[0]\n'
            else:source+='    h=HE_BN0(x)\n'
            if variant==1:source+='    h=h*h\n'
            if variant==2:source+='    h=h+x\n'
            if construction=="unified-scalar":source+='    h+=0.125\n'
            source+='    return [h]\n'
            request=prepare_task(model,dict(construction=construction,helper_profile=HELPER_PROFILE,
                helper_exercise=["HE_BN0"],capability_composition=NAME))
            rows.append(dict(name=construction+"-"+str(variant),construction=construction,variant=variant,
                model=model,request=request,candidate=dict(schema=1,request_id=request["request_id"],hecate_source=source),
                manual_not_agent=True))
    return rows
