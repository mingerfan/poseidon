"""Independent replay of all exported level/scale transitions, not noise simulation."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'resnet20_scale_chain_20260911'))
from search import TREES, prime_test

def equal(a,b,label='scale'):
    assert abs(a-b)<1e-7,(label,a,b)

def verify_folding(data,raw_scale,min_coefficient):
    """P'=rP, c1'=r²c1, c2'=r⁴c2 => DA2(P')=r⁴ DA2(P)."""
    r=2**data['seed_log']
    factor=r**4
    equal(4*data['seed_log'],data['value_factor_log'],'fold exponent')
    equal(raw_scale+data['value_factor_log'],data['application_scale'],'fold interface')
    equal(data['application_scale'],40)
    for original,folded in zip(data['original_coefficients'],data['folded_coefficients'],strict=True):
        for x,y in zip(original,folded,strict=True):
            assert abs(y-x*r)<=1e-12*max(abs(x*r),1e-300),'folded coefficient changed'
    constants=data['original_constants'];adjusted=data['folded_constants']
    assert len(constants)==len(adjusted)==2
    for i,(x,y) in enumerate(zip(constants,adjusted,strict=True)):
        expected=x*r**(2**(i+1))
        assert abs(y-expected)<=1e-12*max(abs(expected),1e-300),'double-angle constant not folded'
    # Exercise the algebra independently of the metadata/count planner.
    for y0 in (-.4,-.1,0,.1,.4):
        a,b=y0,r*y0
        for x,y in zip(constants,adjusted,strict=True):
            a=2*a*a-x;b=2*b*b-y
        assert abs(b-factor*a)<=1e-12*max(abs(factor*a),1e-30),'DA normalization identity'
    effective=min_coefficient+data['seed_log']
    assert effective>=40-1e-7,('fold reduces coefficient precision proxy below 40',effective)
    return effective

class Replay:
    def __init__(self,q):
        self.q=q
        self.logs=[math.log2(v) for v in q]
        self.bits=[0]
        for b in self.logs:self.bits.append(self.bits[-1]+b)
        self.events=[]
        self.minimum_headroom=1e9
        self.maximum_scale=0

    def track(self,n,s,label):
        assert 1<=n<=len(self.q),(label,'q out of range',n)
        assert math.isfinite(s),(label,'nonfinite scale')
        headroom=self.bits[n]-s
        assert headroom>0,(label,'scale exceeds active Q',n,s,headroom)
        self.minimum_headroom=min(self.minimum_headroom,headroom)
        self.maximum_scale=max(self.maximum_scale,s)
        self.events.append(dict(label=label,q=n,scale=s,headroom_bits=headroom))
        return n,s

    def rescale(self,state,count,label):
        n,s=state
        assert 0<=count<n,(label,'invalid rescale',n,count)
        out=(n-count,s-(self.bits[n]-self.bits[n-count]))
        return self.track(*out,label)

    def linear(self,trace,start,label):
        n,s=start
        for i,t in enumerate(trace):
            assert t['q_in']==n,(label,i,'input Q discontinuity')
            equal(t['s_in'],s,label)
            equal(t['pre'],s+t['plain'],label)
            assert t['plain']>=40-1e-7,(label,'matrix encoding below bound')
            self.track(n,t['pre'],f'{label}.{i}.product')
            n,s=self.rescale((n,t['pre']),t['drop'],f'{label}.{i}.rescale')
            assert n==t['q_out']
            equal(s,t['s_out'],label)
            assert s>=44-1e-7
            assert n==1 or s-self.logs[n-1]<44+1e-7
        return n,s

    def preparation(self,data,start):
        n,s=self.linear(data['s2c'],start,'S2C')
        for t in data['guard']:
            assert s>54 and n==t['q_in']
            n,s=self.rescale((n,s),1,'prepare.guard')
            assert n==t['q_out'];equal(s,t['s_out'])
        assert s<=54 and n>=2
        target=self.bits[2]-5
        equal(target,data['target_scale'])
        k=max(1,math.floor(2**(target-s)+0.5))
        assert k==data['multiplier'] and k<=2**32-1
        equal(s+math.log2(k),data['prepared_scale'])
        # Physical input remains Q2; multiplying integer and scale together
        # preserves decoded values, and source's C2S calibration is explicit.
        self.track(2,data['prepared_scale'],'prepare.integer_and_moddrop')
        logical=data['prepared_scale']+50-round(self.bits[2])
        equal(logical,data['c2s_in'])
        return 2,logical

    def boot(self,data,logical_scale):
        n,s=self.linear(data['c2s'],(len(self.q),logical_scale),'C2S')
        assert n==data['eval_q_in'];equal(s,data['eval_s_in'])
        bases={1:(n,s)}
        for st in data['basis']:
            a,sa=bases[st['left']];b,sb=bases[st['right']]
            work=min(a,b)
            equal(sa+sb,st['pre'],'eval basis product')
            self.track(work,st['pre'],f"Eval.T{st['degree']}.product")
            if st['diff']:
                d,sd=bases[st['diff']]
                assert d>=work and st['correction']['q']>=work
                equal(sd+st['correction']['scale'],st['pre'],'basis correction')
            else:
                equal(st['correction']['scale'],st['pre'],'basis constant')
            out=self.rescale((work,st['pre']),st['drop'],f"Eval.T{st['degree']}")
            equal(out[1],st['scale']);assert out[1]>=44-1e-7
            bases[st['degree']]=out
        nodes={}
        minimum_coefficient=1e9
        for idx,leaf in enumerate(data['leaves']):
            assert leaf['drop']==0,'dynamic leaf expected no separate rescale'
            for term in leaf['terms']:
                pl=term['plain'];assert pl['q']>=leaf['q']
                if term['degree']:
                    b,bs=bases[term['degree']]
                    assert b>=leaf['q']
                    equal(bs+pl['scale'],leaf['scale'],'leaf MAC')
                    assert pl['scale']>=45-1e-7
                    minimum_coefficient=min(minimum_coefficient,pl['scale'])
                else:equal(pl['scale'],leaf['scale'],'leaf constant')
            nodes[idx]=self.track(leaf['q'],leaf['scale'],f'Eval.leaf{idx}')
        for st in data['combines']:
            quot=self.rescale(nodes[st['quotient']],st['quotient_drop'],'Eval.quotient')
            equal(quot[1],st['quotient_scale'],'quotient output')
            b,bs=bases[st['basis']]
            work=min(quot[0],b)
            assert work==st['product_q']
            equal(quot[1]+bs,st['product_scale'],'combine multiply')
            self.track(work,st['product_scale'],'Eval.combine_product')
            rem=self.rescale(nodes[st['remainder']],st['remainder_drop'],'Eval.remainder')
            # Accept only exact-scale branches. The source can encode rounded
            # integer corrections; those must not silently pass this audit.
            assert st['product_plain']['q']==0 and st['remainder_plain']['q']==0, (
                'rounded branch correction requires separate accuracy analysis')
            equal(st['product_scale'],rem[1],'combine add alignment')
            equal(st['scale'],st['product_scale'])
            equal(st['product_aligned'],st['scale'])
            equal(st['remainder_aligned'],st['scale'])
            assert st['q']==min(work,rem[0])
            nodes[st['node']]=self.track(st['q'],st['scale'],f"Eval.combine{st['node']}")
        poly_q=data['eval_q_out']+sum(data['double_angle_drops'])
        root=nodes[data['root_node']]
        state=self.rescale(root,root[0]-poly_q,'Eval.final_polynomial_rescale')
        equal(state[1],data['polynomial_scale'])
        for i,drop in enumerate(data['double_angle_drops']):
            pre=self.track(state[0],2*state[1],f'DA{i}.square')
            state=self.rescale(pre,drop,f'DA{i}.rescale')
        assert state[0]==data['eval_q_out'];equal(state[1],data['eval_s_out'])
        return state,minimum_coefficient

    def relu(self,plan,start):
        qin,sin=start
        # ReLU indices are offsets from its actual input Q, not full-chain Q.
        previous_end,previous_scale=0,sin
        minplain=1e9
        def rescale(work,end,pre,post,label):
            assert 0<=work<=end<qin
            self.track(qin-work,pre,label+'.product')
            out=self.rescale((qin-work,pre),end-work,label)
            equal(out[1],post,label)
            assert pre>=40-1e-7 and post>=40-1e-7
        for idx,stage in enumerate(plan['stages']):
            assert stage['start']==previous_end
            equal(stage['input_scale'],previous_scale,'ReLU input interface')
            bases={1:(stage['start'],stage['input_scale'])}
            for e in stage['basis']:
                d=e['degree']
                if d&(d-1)==0:a=b=d//2;diff=0
                else:a=1<<(d.bit_length()-1);b=d-a;diff=2*a-d
                ka,sa=bases[a];kb,sb=bases[b]
                assert max(ka,kb)<=e['work']
                equal(sa+sb,e['pre'])
                if diff:
                    kd,sd=bases[diff];assert kd<=e['work']
                    assert e['pre']-sd>=40-1e-7
                    minplain=min(minplain,e['pre']-sd)
                rescale(e['work'],e['output'],e['pre'],e['scale'],f'ReLU{idx}.T{d}')
                bases[d]=(e['output'],e['scale'])
            def visit(node,tree,degree):
                nonlocal minplain
                work,end,pre,post=node['work'],node['output'],node['pre'],node['scale']
                rescale(work,end,pre,post,f'ReLU{idx}.node')
                if tree is None:
                    assert node['degree']==degree and node['type']=='leaf'
                    for d,enc in zip(range(1,degree+1,2),node['encoding_scales'],strict=True):
                        k,s=bases[d];assert k<=work
                        equal(s+enc,pre);assert enc>=40-1e-7
                        minplain=min(minplain,enc)
                else:
                    split,left,right=tree
                    assert node['split']==split
                    k,s=bases[split];assert k<=work
                    quo,rem=node['quotient'],node['remainder']
                    assert quo['output']==rem['output']==work
                    equal(quo['scale']+s,pre);equal(rem['scale'],pre)
                    visit(quo,right,degree-split);visit(rem,left,split-1)
            visit(stage['tree'],TREES[stage['degree']][0],stage['degree'])
            assert stage['tree']['output']==stage['end']
            equal(stage['tree']['scale'],stage['output_scale'])
            previous_end,previous_scale=stage['end'],stage['output_scale']
        assert previous_end<=plan['tail_work']
        assert plan['tail_work']+plan['tail_drop']==plan['total']
        rescale(plan['tail_work'],plan['total'],sin+previous_scale,40,'ReLU.multiply_original_x')
        equal(self.bits[qin]-self.bits[qin-plan['total']],plan['dropped_bits'])
        return (qin-plan['total'],40),minplain

    def conv(self,trace,start):
        state=start
        assert len(trace)==6
        original=state
        for t in trace:
            assert state[0]==t['q_in'];equal(state[1],t['s_in'])
            if 'plain' in t:
                offset=1 if t['name']=='support_mac' else 0
                equal(t['plain'],self.logs[original[0]-1-offset] if t['name']!='selector_mac'
                      else self.logs[state[0]-1],'Conv actual-modulus encoding')
                assert t['q_out']==state[0]
                equal(state[1]+t['plain'],t['s_out'])
                state=self.track(t['q_out'],t['s_out'],'Conv.'+t['name'])
            else:
                state=self.rescale(state,1,'Conv.'+t['name'])
                assert state[0]==t['q_out'];equal(state[1],t['s_out'])
        assert state[0]==original[0]-3;equal(state[1],original[1])
        return state

def verify(report):
    q,p=report['q_bottom_first'],report['p']
    assert len(set(q+p))==len(q+p)
    assert all(0<v<2**32 and v%131072==1 and prime_test(v) for v in q+p)
    rep=Replay(q)
    _,logical=rep.preparation(report['preparation'],(6,40))
    boot,minboot=rep.boot(report['bootstrap'],logical)
    folding=report['bootstrap'].get('fold',{})
    effective_boot_coefficient=minboot
    if folding.get('enabled'):
        effective_boot_coefficient=verify_folding(folding,boot[1],minboot)
        boot=rep.track(boot[0],folding['application_scale'],'Eval.coefficient_compensated_output_scale')
    if report.get('normalization'):
        norm=report['normalization']
        assert norm['q_in']==boot[0];equal(norm['s_in'],boot[1])
        k=norm['integer'];assert k>=1 and int(k)==k and k<2**32
        multiplied=rep.track(boot[0],boot[1]+math.log2(k),'output_integer_scaling')
        boot=rep.rescale(multiplied,1,'output_normalization')
        assert boot[0]==norm['q_out'];equal(boot[1],norm['s_out'])
    relu,minrelu=rep.relu(report['relu'],boot)
    conv=rep.conv(report['conv'],relu)
    assert conv[0]==6,('loop input not Q6',conv)
    equal(conv[1],40,'loop input scale')
    end=rep.preparation(report['next_preparation'],conv)
    assert end[0]==2;equal(end[1],logical,'bootstrap loop closure')
    assert not report['bootstrap']['keys_generated'] and not report['bootstrap']['gpu_executed']
    return dict(status='PASS_SCALE_LEDGER',full_q=len(q),bootstrap_out=report['bootstrap']['eval_q_out'],
      relu_in_q=boot[0],relu_in_logscale=boot[1],relu_drop=report['relu']['total'],
      relu_out=relu[0],conv_out=conv[0],s2c_out=2,
      min_relu_plain_logscale=minrelu,min_eval_coefficient_logscale=minboot,
      min_effective_eval_coefficient_logscale=effective_boot_coefficient,
      maximum_intermediate_logscale=rep.maximum_scale,minimum_capacity_headroom_bits=rep.minimum_headroom,
      checked_states=len(rep.events),events=rep.events)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('report');ap.add_argument('--output')
    args=ap.parse_args();result=verify(json.loads(Path(args.report).read_text()))
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
