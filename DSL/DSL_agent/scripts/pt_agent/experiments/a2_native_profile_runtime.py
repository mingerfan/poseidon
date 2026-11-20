"""Trusted A2 recording and exact callback Count/Emit checks for native profiles."""
from dataclasses import replace
from types import FunctionType

from ..backends import DagBackend,CountBackend,EmitBackend,RefreshBackend
from ..backends.refresh import RValue
from .. import plan as original_plan
from . import a2_native_profile_planner as planner


class NativeRefresh(RefreshBackend):
    """Keep raw Boot16, then conservatively reserve one logical level to15."""
    def __init__(self,inner,model,plan):
        self.native_model=model;self.normalization_events=0
        super().__init__(inner,input_level=model.input_level,boot_in_level=model.boot_in_level,
            boot_out_level=model.raw_boot_out_level,policy='plan',plan=plan)

    def _planned(self,x,node):
        fresh=node not in self._planned_cache
        value=super()._planned(x,node)
        if fresh:
            value=RValue(self,value.value,self.native_model.boot_out_level)
            self._planned_cache[node]=value;self.normalization_events+=1
            self._record('rescale_single',self.native_model.raw_boot_out_level)
        return value


def record(run,slots,model,*,frontend_normalize=False):
    from ..backends.cost_dag import CostDag
    dag=CostDag(slots)if frontend_normalize else DagBackend(slots);recorder=RefreshBackend(dag,input_level=model.input_level,
        boot_in_level=model.boot_in_level,boot_out_level=model.raw_boot_out_level,policy='record')
    run(recorder)
    if frontend_normalize:dag.finalize()
    return dag,recorder


def express(dag,recorder,placement,model,name):
    namespace=dict(vars(original_plan),levels=lambda profile:dict(input_level=model.input_level,
        boot_in_level=model.boot_in_level,boot_out_level=model.raw_boot_out_level))
    function=FunctionType(original_plan.express.__code__,namespace,'express',original_plan.express.__defaults__)
    function.__kwdefaults__=dict(original_plan.express.__kwdefaults__)
    result=function(dag,recorder,placement,profile='a2-native')
    merged=set(result['merged'])
    if merged:
        latency=dict(placement.node_latency)
        for nid in merged:latency[nid]-=model.boot_us
        placement=replace(placement,boots=placement.boots-len(merged),
            latency_us=placement.latency_us-len(merged)*model.boot_us,
            boot_nodes=[nid for nid in placement.boot_nodes if nid not in merged],node_latency=latency)
    result.update(placement=name,latency_us=placement.latency_us,source='pt_agent.lower',
        normalization=dict(levels_consumed=1,raw_boot_out_level=model.raw_boot_out_level,
            effective_boot_out_level=model.boot_out_level,rescalingFactor=51,waterline=40))
    return result,placement


def run_checked(run,slots,model,plan,inner):
    from .qwen_component_cost import _dead_count_scope
    active=NativeRefresh(inner,model,plan)
    with _dead_count_scope(active):run(active)
    stats=inner.stats();logical=active.stats()
    if stats['counts']['boot']!=plan['boots']or logical['boots']!=plan['boots']:
        raise planner.NativePlanError('A2 callback bootstrap inventory differs from the planned coverage')
    if active.normalization_events!=plan['boots']:
        raise planner.NativePlanError('A2 normalized Boot event count differs')
    return stats,dict(planned_bootstraps=plan['boots'],callback_bootstraps=stats['counts']['boot'],
        normalizations=active.normalization_events,normalization_levels_consumed=1,
        raw_boot_out_level=model.raw_boot_out_level,effective_boot_out_level=model.boot_out_level,
        logical_min_level=logical['min_level'],calls=logical['calls'])


def validate_all(run,slots,model,plans,*,frontend=None):
    """Each candidate executes trusted lowering once for Count and once for Emit."""
    proofs={}
    for name,plan in plans.items():
        count=CountBackend(slots,input_level=model.input_level,boot_out_level=model.raw_boot_out_level)
        count_stats,count_proof=run_checked(run,slots,model,plan,count)
        emitter=EmitBackend(slots,**({}if frontend is None else{'frontend':frontend}))
        emit_stats,emit_proof=run_checked(run,slots,model,plan,emitter)
        if count_stats['counts']!=emit_stats['counts']or count_proof!=emit_proof:
            raise planner.NativePlanError('A2 complete Count/Emit requests or logical contracts differ')
        proofs[name]=dict(status='ok',count=count_proof,emit=emit_proof,
            counts=count_stats['counts'],ops_total=count_stats['ops_total'],
            scope='Raw callback arithmetic and Boot counts; level15 is a conservative frontend reserve and its rescale16 cost is a planning proxy. No extra physical PARS operation or global native optimality is asserted; actual RuntimePlan sums are authoritative.')
    return proofs


def plan(run,slots,model,*,time_limit=600.,state_limit=4096):
    dag,recorder=record(run,slots,model)
    found=planner.candidates(dag,recorder,model,time_limit=time_limit,state_limit=state_limit)
    plans={};summary={}
    for name,row in found.items():
        item,placement=express(dag,recorder,row['placement'],model,name);plans[name]=item
        summary[name]=dict(boots=placement.boots,latency_us=placement.latency_us,
            seconds=row['seconds'],states_peak=row.get('states_peak'))
    selected=min(summary,key=lambda name:summary[name]['latency_us'])
    return plans,summary,selected
