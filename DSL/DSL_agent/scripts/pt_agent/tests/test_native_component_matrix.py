"""Small metadata-only checks; no artifact/model/native dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pt_agent.experiments import native_component_matrix as m


def fixture():
    def value(name, level):
        return dict(id=name,kind='ciphertext',level=level,scale_log2=43,
                    components=2,context='fixture')
    values=[value('input',5),value('a',1),value('b',5),value('c',1),value('d',1)]
    operations=[dict(kind='compute',ordinal=10,op='mod_switch',inputs=['input'],output='a',attrs=dict(target_level=1)),
                dict(kind='compute',ordinal=11,op='boot',inputs=['a'],output='b',attrs=dict(implementation='native',target_level=5,target_scale_log2=43,target_components=2)),
                dict(kind='compute',ordinal=12,op='mod_switch',inputs=['b'],output='c',attrs=dict(target_level=1)),
                dict(kind='compute',ordinal=13,op='add_cc',inputs=['a','c'],output='d')]
    plan=dict(values=values,initialization=[],execution=operations)
    index={x['id']:x for x in values};lookup={}
    def meta(name):
        return dict(kind='ciphertext',level=index[name]['level'],scale_log2=50,components=2)
    for op in operations:
        attrs=dict(op.get('attrs',{}))
        if 'target_scale_log2' in attrs:attrs['target_scale_log2']=50
        cell=dict(op=op['op'],inputs=[meta(x) for x in op['inputs']],output=meta(op['output']),attrs=attrs)
        lookup[m.cell_key(cell)]=dict(core_ns=100)
    return plan,lookup


class NativeComponentMatrixChecks(unittest.TestCase):
    def test_wrong_compiled_directory_is_rejected(self):
        record=dict(model='resnet20_silu',phases=dict(compile=dict(status='ok',runtime_plan='/tmp/other/plan.json')))
        with self.assertRaisesRegex(ValueError,'saved successful'):
            m.check_sources(Path('/tmp/case'),Path('/tmp/case/plan.json'),record,Path('/tmp/quality.json'),{},'resnet20_silu')

    def test_fresh_quality_program_binding_is_required(self):
        record=dict(model='resnet20_silu',parameter_group='original qwen48',phases=dict(
            compile=dict(status='ok',runtime_plan='/tmp/case/plan.json',process=dict(exit_code=0)),
            agent=dict(agent_result=dict(selected_program='/tmp/case/program.ptdsl'))),
            processes=[dict(phase='emit',argv=['--program','/tmp/case/program.ptdsl','--post-dir','/tmp/case/post-freeze'])])
        quality=dict(status='quality_accepted',formal_frozen=True,quality_acceptance_passed=True,
                     strict_math_passed=True,model='resnet20_silu',selected_program='/tmp/foreign/program.ptdsl')
        with self.assertRaisesRegex(ValueError,'quality/program'):
            m.check_sources(Path('/tmp/case'),Path('/tmp/case/plan.json'),record,Path('/tmp/quality.json'),quality,'resnet20_silu')
        quality['selected_program']='/tmp/case/program.ptdsl'
        self.assertTrue(m.check_sources(Path('/tmp/case'),Path('/tmp/case/plan.json'),record,Path('/tmp/quality.json'),quality,'resnet20_silu')['same_model_quality_program_source'])

    def test_fresh_worker_record_path_is_resolved_as_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory);worker=case/'worker-result.json';program=case/'program.ptdsl'
            worker.write_text(json.dumps(dict(selected_program=str(program))))
            record=dict(model='resnet20_silu',parameter_group='original qwen48',phases=dict(
                compile=dict(status='ok',runtime_plan=str(case/'plan.json'),process=dict(exit_code=0)),
                agent=dict(agent_result=str(worker))),processes=[dict(phase='emit',argv=[
                    '--program',str(program),'--post-dir',str(case/'post-freeze')])])
            quality=dict(status='quality_accepted',formal_frozen=True,quality_acceptance_passed=True,
                         strict_math_passed=True,model='resnet20_silu',selected_program=str(program))
            self.assertTrue(m.check_sources(case,case/'plan.json',record,case/'quality.json',quality,'resnet20_silu')['same_model_quality_program_source'])
            record['processes'][0]['argv'][1]=str(case/'foreign.ptdsl')
            with self.assertRaisesRegex(ValueError,'Emit source differs'):
                m.check_sources(case,case/'plan.json',record,case/'quality.json',quality,'resnet20_silu')

    def test_control_compile_copy_requires_same_program_AST(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory);original=case/'original.ptdsl';copied=case/'copied.ptdsl';quality_path=case/'quality.json'
            original.write_text('def model(x):\n return x\n');copied.write_text('def model(x):\n    return x\n')
            quality=dict(status='ok',model='pt_agent.experiments.r13_common_models:resnet20_silu',
                         metrics=dict(heldout_acceptance=True),worker=dict(artifacts=dict(program=str(original))))
            record=dict(model=quality['model'],system='dacapo-depth-dp',parameter_group='original qwen48',
                        source=dict(source_result=str(quality_path),program=str(original)),
                        phases=dict(compile=dict(status='ok',runtime_plan=str(case/'plan.json'),
                                                 source_program=str(copied),process=dict(exit_code=0))))
            self.assertTrue(m.check_sources(case,case/'plan.json',record,quality_path,quality,'resnet20_silu')['same_model_quality_program_source'])
            copied.write_text('def model(x):\n return different\n')
            with self.assertRaisesRegex(ValueError,'quality/program'):
                m.check_sources(case,case/'plan.json',record,quality_path,quality,'resnet20_silu')

    def test_static_lru_keeps_constructor_order_and_counts_reuse(self):
        order,cold=m.lru([1,2,3],2)
        self.assertEqual((order,cold['misses']),([2,3],3))
        order,warm=m.lru([2,4,3],2,order)
        self.assertEqual((order,warm),([4,3],dict(requests=3,hits=1,misses=2)))

    def test_projection_scale_mapping_rejects_unmapped_metadata(self):
        self.assertEqual((m.scale(43),m.scale(86)),(50,100))
        with self.assertRaises(ValueError):m.scale(51)

    def test_stream_decoder_reads_small_json_without_payload_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json'
            path.write_text(json.dumps(dict(padding='a'*70000,values=[dict(id='a'),dict(id='b')])))
            self.assertEqual(list(m.array_items(path,'values')),[dict(id='a'),dict(id='b')])

    def test_actual_plan_metadata_counts_neighborhood_and_edges(self):
        plan,lookup=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text(json.dumps(plan))
            record=m.inventory(path,lookup)
        self.assertEqual((record['source_operations'],record['bootstrap_count'],record['unique_cells']),(4,1,3))
        self.assertEqual(sum(x['component_ns'] for x in record['weighted_cells']),400)
        self.assertEqual(record['bootstrap_neighborhoods'][0]['actual_SSA_edges'],[
            dict(producer_index=0,consumer_index=1,value='a'),
            dict(producer_index=1,consumer_index=2,value='b'),
            dict(producer_index=0,consumer_index=3,value='a'),
            dict(producer_index=2,consumer_index=3,value='c')])

    def test_unmeasured_cell_is_rejected(self):
        plan,lookup=fixture();lookup.pop(next(iter(lookup)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'does not cover'):m.inventory(path,lookup)

    def test_plan_without_bootstrap_has_no_invented_neighborhood(self):
        plan,lookup=fixture();plan['execution']=[]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'bootstrapped'):m.inventory(path,lookup)

    def test_svg_is_editable_arial_and_only_actual_edges(self):
        plan,lookup=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text(json.dumps(plan));record=m.inventory(path,lookup)
        rows=[]
        for identity,model,label,boots,*_ in m.CATALOG:
            if model=='AlexNet':continue
            rows.append(dict(record,id=identity,model=model,variant_label=label))
        svg=m.render_svg(rows);tree=ET.fromstring(svg)
        ns={'s':'http://www.w3.org/2000/svg'}
        self.assertIn('font-family:Arial',svg)
        self.assertIn('tuner disabled',svg)
        self.assertEqual(len(tree.findall('.//s:path[@class="edge"]',ns)),48)
        self.assertNotIn('<image',svg)


if __name__=='__main__':unittest.main()
