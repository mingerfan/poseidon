"""One actual tiny layer DAG/plan/Count protocol; no real model or full graph."""
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from pt_agent.experiments import qwen_component_cost as entry

TEMPLATE=['{hecate_opt}','{input}','--pars','--ckks-config={package}/config/compiler-config.json',
    '--waterline=43','--output-val=10','--runtime-plan-context-id=qwen-bootstrap48',
    '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
    '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,'-o','{output_dir}/lowered.mlir']


class ActualTinyMeasureChecks(unittest.TestCase):
    def test_head_own_interface_does_not_redefine_original_layer_connector_contract(self):
        from pt_agent.tests.test_qwen_component_cost import row
        first=row('layer0',2,index=0);head=row('head',1)
        head['caller_connector_blocks']=1
        result=entry.aggregate([first,head],boot_us=10.)
        self.assertEqual(result['full_generation_projection']['external_connector_boot_requests'],3)
        self.assertEqual(head['caller_connector_blocks'],1)

    def test_original_measure_real_plan_verify_bound_Count_and_immediate_cleanup(self):
        from pt_agent.experiments.r29_local_compile_matrix import native_metadata_commands
        with native_metadata_commands(TEMPLATE):
            import numpy as np
            from pt_agent.dsl.parser import parse
            from pt_agent.experiments import qwen_layered_structure as layered
            from pt_agent.experiments import qwen_agent_cost_lifetime as lifetime
            from pt_agent.backends import streaming_count
            text=('def qwen_layer(x, am, w):\n    y=bias(x,b=W[w],axis=-1)\n'
                '    out=add(y,x)\n    return out\n'
                '@fhe.program(slots=16384,refresh="agent",placement="program-plan",l_max=5,boot_out_level=5)\n'
                'def qwen2(inputs_embeds: Cipher[1,32,896],attention_mask: Mask[1,32],sel: Mask[1,32]):\n'
                '    x=pack(inputs_embeds,layout="token")\n    am=pack(attention_mask,layout="flat")\n'
                '    sm=pack(sel,layout="flat")\n')
            shapes={};current='x'
            for index in range(24):
                name=f'original.model.layers.{index}.fixture.weight';shapes[name]=(896,)
                text+=f'    h{index}=qwen_layer({current},am,"{name}")\n';current=f'h{index}'
            text+=(f'    n=rmsnorm({current},w=W["original.model.norm.weight"],eps=0.001,'
                'rsqrt_method="newton",rsqrt_domain=(0.01,0.2),rsqrt_deg0=7,rsqrt_iters=2)\n'
                '    selected=select(n,sm)\n    logits=linear(selected,w=W["original.model.embed_tokens.weight"])\n'
                '    return logits\n')
            shapes['original.model.norm.weight']=(896,);shapes['original.model.embed_tokens.weight']=(151936,896)
            program=parse(text);partition=layered.split_program(program,shapes)
            self.assertEqual(len(partition.head.program.params),3)
            self.assertEqual([p.shape for p in partition.head.program.params],[(1,32,896),(1,32),(1,32)])
            weights={name:SimpleNamespace(shape=shape)for name,shape in shapes.items()}
            vector=np.arange(896,dtype=np.float64)*2.**-20
            weights['original.model.layers.0.fixture.weight']=vector
            callback=layered.callbacks(partition,weights)[0][0]
            references=[];lifetimes=[];base_life=lifetime.Lifetime;base_count=streaming_count.StreamingCountBackend
            class ObservedLifetime(base_life):
                def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);lifetimes.append(self)
                def track_backend(self,backend):references.append(weakref.ref(backend));return super().track_backend(backend)
            class ObservedCount(base_count):
                def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);references.append(weakref.ref(self))
            enabled=gc.isenabled();gc.disable()
            try:
                with tempfile.TemporaryDirectory()as temp,patch.object(lifetime,'Lifetime',ObservedLifetime), \
                        patch.object(streaming_count,'StreamingCountBackend',ObservedCount):
                    api=entry.OriginalComponents('qwen48',Path(temp),lifetime.Diagnostics())
                    row=api.measure(callback)
                    plan=json.loads(Path(row['plan_artifact']).read_text())
                    self.assertEqual(plan['format'],'pt-agent-refresh-plan-v1')
                    self.assertEqual(row['boundary_boots'],32)
                    self.assertEqual(row['complete_count']['counts']['boot'],plan['boots']+32)
                    self.assertEqual(row['internal']['refresh_plan']['replay']['boots'],plan['boots'])
                    self.assertEqual(row['output_levels'],[5]*32)
                    self.assertEqual(row['caller_connector_blocks'],3)
                    self.assertEqual(row['source'],layered.source_binding(partition.layers[0]))
                    self.assertEqual(row['internal']['placement_model'],'program-plan')
                    json.dumps({k:v for k,v in row.items()if k!='constants'},allow_nan=False)
                    self.assertTrue(references)
                    self.assertTrue(all(reference()is None for reference in references))
                    self.assertTrue(all(not item.records and not item.active for item in lifetimes))
                    self.assertIs(weights['original.model.layers.0.fixture.weight'],vector)
            finally:
                if enabled:gc.enable()


if __name__=='__main__':unittest.main()
