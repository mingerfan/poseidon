"""A2-only initial16 / native3->16 planner; existing placement modules stay intact.

Native Boot returns level16. This private planner conservatively reserves
one level and rescale[16] cost after each Boot, exposing effective level15.
These are frontend planning reserves; an extra physical PARS operation is
not asserted. Scores are conservative frontend proxies, with no claim of
global native optimality. Final costs use actual compiled RuntimePlans.
Source arithmetic stays unchanged. The stage dynamic program keeps its
complete live frontier because initial inputs16 differ from reserved15.
"""
from dataclasses import dataclass
import math
import time
from types import FunctionType

from ..placement import Placement
from .. import stage_plan


class NativePlanError(ValueError):
    pass


@dataclass(frozen=True)
class Model:
    table: dict
    input_level: int = 16
    boot_in_level: int = 3
    raw_boot_out_level: int = 16
    normalization_levels: int = 1

    @property
    def boot_out_level(self):
        return self.raw_boot_out_level-self.normalization_levels

    @property
    def boot_us(self):
        return self.at('bootstrap_single',self.boot_in_level)+self.at('rescale_single',self.raw_boot_out_level)

    def at(self,key,level):
        row=self.table.get(key)
        if type(level)is not int or not isinstance(row,list)or not 0<=level<len(row):
            raise NativePlanError('E_TABLE_LEVEL: missing original '+key+' at level'+str(level))
        return row[level]

    def op_latency(self,kind,ciphers,level):
        if kind in ('mul_cc','mul_cp'):
            return self.at('mul_double'if kind=='mul_cc'else'mul_single',level)+self.at('rescale_single',level)
        key='rotate_single'if kind=='rot'else'negate_single'if kind=='neg'else'add_double'if ciphers==2 else'add_single'
        return self.at(key,level)


def model_for(config):
    expected=dict(runtime='HEAAN-HEVM',polynomialDegree=131072,rescalingFactor=51,
        levelLowerBound=1,levelUpperBound=29,bootstrapLevelLowerBound=3,bootstrapLevelUpperBound=16)
    if any(config.get(k)!=v for k,v in expected.items()):raise NativePlanError('A2 requires the original native profile unchanged')
    table={key.removeprefix('earth.'):value for key,value in config.get('latencyTable',{}).items()}
    required=('add_double','add_single','bootstrap_single','modswitch_single','negate_single','mul_double','mul_single','rescale_single','rotate_single')
    if any(not isinstance(table.get(k),list)or not table[k]
        or any(type(v)not in(int,float)or not math.isfinite(v)or v<0 for v in table[k])for k in required):
        raise NativePlanError('A2 requires finite nonempty original latency rows')
    return Model(table)


def simulate(dag,refreshed,model):
    n=len(dag.kind);levels=[0]*n;latency=[0.]*n
    for nid,(kind,args)in enumerate(zip(dag.kind,dag.args)):
        if kind=='input':levels[nid]=model.input_level;continue
        if not args:raise NativePlanError('A2 requires ordinary operand-bearing finalized arithmetic')
        if any(nid in refreshed.get(arg,())and levels[arg]<model.boot_in_level for arg in args):return None
        current=min(model.boot_out_level if nid in refreshed.get(arg,())else levels[arg]for arg in args)
        result=current-(kind in stage_plan.MUL)
        if result<model.boot_in_level:return None
        levels[nid]=result;latency[nid]=model.op_latency(kind,len(args),current)
    for arg,users in refreshed.items():
        if type(arg)is not int or not 0<=arg<n or levels[arg]<model.boot_in_level:return None
        if any(type(user)is not int or user<0 or user>n for user in users):return None
    return levels,latency


def as_placement(dag,refreshed,model,*,cuts,max_depth):
    result=simulate(dag,refreshed,model)
    if result is None:raise NativePlanError('Native placement violates the effective post-normalization level budget')
    _,latency=result;node_latency={nid:value for nid,value in enumerate(latency)if value}
    for nid in refreshed:node_latency[nid]=node_latency.get(nid,0)+model.boot_us
    return Placement(list(cuts),len(cuts),len(refreshed),sum(latency)+len(refreshed)*model.boot_us,
        sorted(refreshed),node_latency,max_depth,{nid:sorted(users)for nid,users in refreshed.items()})


def _depths(dag):
    n=len(dag.kind);users=stage_plan._users(dag);depth=[]
    for kind,args in zip(dag.kind,dag.args):depth.append(max((depth[arg]for arg in args),default=0)+(kind in stage_plan.MUL))
    maximum=max(depth,default=0);depth.append(maximum+1)
    for nid in range(n-1,-1,-1):
        if dag.kind[nid]in('rot','neg')and users[nid]:depth[nid]=min(depth[user]for user in users[nid])
    return depth,users,maximum


def depth_candidate(dag,model):
    """Original depth-boundary objective with13 fresh and12 normalized depths."""
    n=len(dag.kind);depth,users,maximum=_depths(dag);crossing=[0]*(maximum+2)
    for nid,uses in enumerate(users):
        if uses:
            start=0 if dag.kind[nid]=='input'else depth[nid];end=max(start,max(depth[user]for user in uses))
            crossing[start]+=1;crossing[end]-=1
    for index in range(1,maximum+1):crossing[index]+=crossing[index-1]
    initial=model.input_level-model.boot_in_level;segment=model.boot_out_level-model.boot_in_level
    best=[math.inf]*(maximum+1);previous=[-1]*(maximum+1);best[0]=0
    for end in range(1,maximum+1):
        if end<=initial:best[end]=0
        for start in range(end-1,max(0,end-segment)-1,-1):
            cost=best[start]+crossing[start]
            if cost<best[end]:best[end],previous[end]=cost,start
    if not math.isfinite(best[maximum]):raise NativePlanError('No native depth placement')
    cuts=[];end=maximum
    while previous[end]>=0:end=previous[end];cuts.append(end)
    cuts.reverse();levels={nid:model.input_level for nid,k in enumerate(dag.kind)if k=='input'}
    remaining=[len(uses)for uses in users];live={nid for nid,k in enumerate(dag.kind)if k=='input'and users[nid]}
    refreshed={};booted=set();next_cut=0
    for nid in sorted(range(n+1),key=depth.__getitem__):
        if nid!=n and dag.kind[nid]=='input':continue
        if next_cut<len(cuts)and depth[nid]>cuts[next_cut]:
            end=cuts[next_cut+1]if next_cut+1<len(cuts)else maximum
            minimum=model.boot_in_level+end-cuts[next_cut]
            for value in sorted(live):
                if levels[value]<model.boot_in_level:raise NativePlanError('Depth frontier below native bootstrap input')
                if levels[value]<minimum:
                    levels[value]=model.boot_out_level;booted.add(value);refreshed.setdefault(value,set())
            next_cut+=1
        if nid==n:
            for value in dag._outputs or []:
                if value.nid in booted:refreshed.setdefault(value.nid,set()).add(n)
            continue
        args=dag.args[nid]
        for arg in args:
            if arg in booted:refreshed.setdefault(arg,set()).add(nid)
        current=min(levels[arg]for arg in args);levels[nid]=current-(dag.kind[nid]in stage_plan.MUL)
        if levels[nid]<model.boot_in_level:raise NativePlanError('Depth candidate exceeds native initial/post-Boot budget')
        for arg in args:
            remaining[arg]-=1
            if remaining[arg]==0:live.discard(arg)
        if remaining[nid]>0:live.add(nid)
    return as_placement(dag,refreshed,model,cuts=cuts,max_depth=maximum)


def _stage_structure(dag,node_labels):
    n=len(dag.kind);distinct=sorted({node_labels[nid]for nid,k in enumerate(dag.kind)if k!='input'})
    index={label:i for i,label in enumerate(distinct)};m=len(distinct)
    li=[-1 if k=='input'else index[node_labels[nid]]for nid,k in enumerate(dag.kind)]
    users=stage_plan._users(dag);last=[max((m if u==n else li[u]for u in uses),default=-2)for uses in users]
    groups=[[]for _ in distinct]
    for nid,args in enumerate(dag.args):
        if any(li[arg]>li[nid]for arg in args):raise NativePlanError('A2 node reads a later stage')
        if li[nid]>=0:groups[li[nid]].append(nid)
    return distinct,li,users,last,groups


def _advance(dag,group,state,last,label,model):
    current=dict(state);cost=0.
    for nid in group:
        args=dag.args[nid];level=min(current[arg][0]for arg in args)
        multiplied=dag.kind[nid]in stage_plan.MUL;result=level-multiplied
        if result<model.boot_in_level:return None
        cost+=model.op_latency(dag.kind[nid],len(args),level)
        current[nid]=(result,multiplied or any(current[arg][1]for arg in args))
    return {nid:value for nid,value in current.items()if last[nid]>label},cost


def _cut(state,model):
    touched=[nid for nid,(level,dirty)in state.items()if dirty]
    after={nid:(model.boot_out_level,False)if dirty else(level,False)for nid,(level,dirty)in state.items()}
    return after,len(touched)*model.boot_us,touched


def stage_candidate(dag,node_labels,model,*,time_limit=600.,state_limit=4096):
    """Exact label-boundary DP with level/dirty frontier state and explicit limits."""
    started=time.monotonic();distinct,li,users,last,groups=_stage_structure(dag,node_labels)
    initial={nid:(model.input_level,False)for nid,k in enumerate(dag.kind)if k=='input'and last[nid]>=0}
    states=[(initial,0.,())];maximum=max(_depths(dag)[0],default=0)-1;peak=1;work=0
    for label,group in enumerate(groups):
        found={}
        for state,cost,cuts in states:
            result=_advance(dag,group,state,last,label,model);work+=len(group)
            if result is None:continue
            next_state,delta=result;options=[(next_state,cost+delta,cuts)]
            if label+1<len(groups):
                cut_state,cut_cost,touched=_cut(next_state,model)
                if touched:options.append((cut_state,cost+delta+cut_cost,cuts+(label,)))
            for next_state,next_cost,next_cuts in options:
                key=tuple((nid,*value)for nid,value in sorted(next_state.items()))
                old=found.get(key)
                if old is None or(next_cost,next_cuts)<(old[1],old[2]):found[key]=(next_state,next_cost,next_cuts)
            if time.monotonic()-started>time_limit:raise TimeoutError('A2 stage frontier placement time allowance exhausted')
            if len(found)>state_limit:raise NativePlanError('A2 stage frontier state allowance exhausted')
        if not found:raise NativePlanError('No feasible native stage-cut frontier')
        states=list(found.values());peak=max(peak,len(states))
    if not states:raise NativePlanError('Empty native stage DAG')
    _,objective,cuts=min(states,key=lambda row:(row[1],row[2]));state=initial;refreshed={}
    for label,group in enumerate(groups):
        result=_advance(dag,group,state,last,label,model)
        if result is None:raise NativePlanError('Chosen stage path replay became infeasible')
        state,_=result
        if label in cuts:
            state,_,touched=_cut(state,model)
            for nid in touched:refreshed[nid]={u for u in users[nid]if(u==len(dag.kind)or li[u]>label)}
    result=as_placement(dag,refreshed,model,cuts=[distinct[i]for i in cuts],max_depth=maximum)
    if not math.isclose(result.latency_us,objective,rel_tol=1e-12,abs_tol=1e-9):raise NativePlanError('Native stage frontier objective differs from independent replay')
    return result,dict(states_peak=peak,node_steps=work,stage_seconds=time.monotonic()-started)


def prune(dag,placement,model):
    """Original E5b prune body, privately bound to the native exact simulator."""
    namespace=dict(vars(stage_plan),simulate=simulate,as_placement=as_placement)
    function=FunctionType(stage_plan.prune.__code__,namespace,'prune',stage_plan.prune.__defaults__)
    function.__kwdefaults__=dict(stage_plan.prune.__kwdefaults__)
    return function(dag,placement,model)


def candidates(dag,recorder,model,*,time_limit=600.,state_limit=4096):
    found={};started=time.monotonic();base=depth_candidate(dag,model)
    found['depth-dp']=dict(placement=base,seconds=time.monotonic()-started)
    pruned,stats=prune(dag,base,model);found['depth-dp+prune']=dict(placement=pruned,seconds=stats['prune_s'])
    cut,stats=stage_candidate(dag,stage_plan.labels(dag,recorder),model,time_limit=time_limit,state_limit=state_limit)
    found['stage-cut']=dict(placement=cut,seconds=stats['stage_seconds'],states_peak=stats['states_peak'])
    pruned,stats=prune(dag,cut,model);found['stage-cut+prune']=dict(placement=pruned,seconds=stats['prune_s'])
    return found
