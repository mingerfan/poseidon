"""Small source/session/orchestration fixtures; no original model, text or API."""
import json
import io
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_agent_continuation as worker


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class QwenContinuationCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        root = self.root; model = root/'model'; model.mkdir()
        info = root/'info'; (info/'graph').mkdir(parents=True)
        args = SimpleNamespace(
            model_info=info, model=model, inputs=root/'samples.jsonl', source_rows=root/'source_rows.json',
            metadata_template=root/'template.json', frontend=root/'expr.py', profile=root/'profile.json',
            linked_result=root/'linked.json', coefficients=root/'coefficients.json',
            intake_python=root/'python', ledger=root/'ledger.jsonl', allowance=root/'allowance.json',
            out=root/'actual', calibration=root/'actual/calibration.json', session=root/'actual/session',
            codegen=root/'actual/codegen', memory_gib=12., rss_gib=14., case_gib=32,
            seconds=21600, through='metadata', server_mode=False)
        split = dict(calibration=list(range(23)), heldout=list(range(23,43)),
                     confirm=list(range(43,63)), report=list(range(63,223)),
                     sample_ids=['samples.jsonl:'+str(i).zfill(8) for i in range(223)])
        record = dict(task='causal_lm', metric='lm_next_token',
            threshold=dict(agreement_pct=99., logit_rel_err=.01),
            source=dict(local_path=str(model), config=dict(num_hidden_layers=24,hidden_size=896,vocab_size=151936)),
            input_handling=dict(single_lm_head=True), sequence=dict(length=32), split=split, graph='graph')
        put(info/'model_info.json', record)
        put(info/'graph/graph.json',dict(nodes=[[str(i),kind,[],[],{},None]
            for i,kind in enumerate(['rmsnorm']*49+['softmax']*24+['act']*24)]))
        (info/'graph/weights.npz').write_text('fixture, no weights')
        args.inputs.write_text('fixture, no sentence')
        put(args.source_rows,dict(source=str(root/'train-00000-of-00001.parquet'),column='sentence',
            row_number_base=0,source_rows=list(range(223))))
        put(args.inputs.parent/'worker_result.json',dict(status='ok',phase='prepare-text',sample_count=223,
            inputs_jsonl=str(args.inputs),source_rows=str(args.source_rows)))
        put(args.linked_result,dict(status='ok',linked_packed_math_verified=True,
            same_backend_embedding_handle_consumed=True,candidate_frozen=False,
            model_source=dict(model_info=str(info)),source=dict(coefficients=str(args.coefficients))))
        for path in (args.metadata_template,args.frontend,args.profile,args.coefficients,args.intake_python):
            path.write_text('{}')
        return args, record

    def state(self, phase='done', selected=None, failed=None):
        directory = self.root/'session'; directory.mkdir(exist_ok=True)
        state = dict(phase=phase,prompt_version='r8-v11',max_iters=10,placement='program-plan',
            confirm_rule='c2',calls=2,iteration=3,failed=failed)
        put(directory/'SESSION.json',state)
        status = 'error' if failed else 'confirm_failed' if selected is None else 'feasible'
        confirm = dict(selected=selected,evals=[])
        if selected is not None:
            confirm['evals']=[dict(iteration=selected['iteration'],origin=selected['origin'],passed=True,tail=dict(passed=True))]
            program=directory/'program.ptdsl';program.write_text(selected['source'])
            put(directory/'FROZEN.json',dict(program=str(program)))
        put(directory/'AGENT.json',dict(status=status,confirm=confirm,error=failed))
        return directory

    def test_source_retains_original_split_and_does_not_read_payload(self):
        args,_=self.fixture(); actual=worker.source_gate(args)
        self.assertEqual(actual['split_counts'],dict(calibration=23,heldout=20,confirm=20,report=160))
        self.assertFalse(actual['weights_archive']['values_read'])
        self.assertFalse(actual['report_reader_available'])

    def test_source_rejects_partition_overlap(self):
        args,record=self.fixture();record['split']['confirm'][0]=0;put(args.model_info/'model_info.json',record)
        with self.assertRaisesRegex(ValueError,'disjoint'):worker.source_gate(args)

    def test_source_rejects_dropped_layer(self):
        args,_=self.fixture();path=args.model_info/'graph/graph.json';record=json.loads(path.read_text())
        record['nodes'].pop();put(path,record)
        with self.assertRaisesRegex(ValueError,'24'):worker.source_gate(args)

    def test_source_rejects_different_model(self):
        args,_=self.fixture();args.model=self.root/'other'
        with self.assertRaisesRegex(ValueError,'original full'):worker.source_gate(args)

    def test_report_access_rejected_before_numeric_import(self):
        splits=worker.TokenSplits(None,self.root/'not-read',None,[])
        with self.assertRaisesRegex(ValueError,'report'):splits.get('report')

    def test_done_without_candidate_is_normal_quality_rejection(self):
        self.assertEqual(worker.session_outcome(self.state())['status'],'quality_rejected')

    def test_done_execution_error_stays_error(self):
        result=worker.session_outcome(self.state(failed='resource failure'))
        self.assertEqual(result['status'],'error');self.assertFalse(result['frozen'])

    def test_frozen_requires_passing_c2_and_same_program(self):
        selected=dict(source='def f(x):\n return x\n',iteration=1,origin='llm')
        directory=self.state(selected=selected)
        self.assertTrue(worker.session_outcome(directory)['frozen'])
        record=json.loads((directory/'AGENT.json').read_text());record['confirm']['evals'][0]['tail']['passed']=False
        put(directory/'AGENT.json',record)
        with self.assertRaisesRegex(ValueError,'C2'):worker.session_outcome(directory)

    def test_rejected_session_cannot_retain_frozen_marker(self):
        directory=self.state();put(directory/'FROZEN.json',{})
        with self.assertRaisesRegex(ValueError,'inconsistent'):worker.session_outcome(directory)

    def test_unknown_phase_rejected(self):
        with self.assertRaisesRegex(ValueError,'phase'):worker.session_outcome(self.state(phase='report'))

    def test_handle_binding_captures_outputs_and_restores_methods_on_error(self):
        class Backend:
            def input(self,index):return 'original'+str(index)
            def output(self,values):raise AssertionError('No intermediate output declaration')
        backend=Backend();handle=object()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with worker._existing_inputs(backend,[handle]) as captured:
                self.assertIs(backend.input(0),handle);backend.output([handle]);self.assertIs(captured[0],handle)
                raise RuntimeError('fixture')
        self.assertNotIn('input',backend.__dict__);self.assertNotIn('output',backend.__dict__)
        self.assertEqual(backend.input(0),'original0')

    def test_phase_command_passes_explicit_paths_without_report(self):
        args,_=self.fixture();command=worker.phase_command(args,'step',self.root/'phase')
        for name in ('--model-info','--inputs','--calibration','--session','--ledger','--allowance'):
            self.assertIn(name,command)
        self.assertNotIn('--report',command);self.assertNotIn('pt_agent.experiments.qwen_server_job',command)
        self.assertEqual(command[command.index('--memory-gib')+1],'12.0')

    def test_server_package_is_finite_and_never_executes(self):
        args,_=self.fixture();target=self.root/'server_jobs/qwen'
        with patch.object(worker.subprocess,'run',side_effect=AssertionError('No process')):
            worker.write_server_job(args,target)
        record=json.loads((target/'job.json').read_text())
        self.assertEqual(record['status'],'prepared_not_executed')
        self.assertIn('--server-mode',record['argv']);self.assertEqual(record['report_samples'],0)
        self.assertEqual(record['required_hardware']['physical_ram_gib'],128)
        self.assertTrue((target/'run.sh').is_file())

    def test_registry_has_exact_eight_fields(self):
        path=self.root/'registry.jsonl'
        row=dict(run_id='fixture',round=worker.ROUND,model=worker.MODEL,mode=worker.MODE,
            git_rev='fixture',metrics={},artifacts_dir=str(self.root),ts='fixture')
        worker._append_registry(path,row)
        actual=json.loads(path.read_text());self.assertEqual(set(actual),set(row));self.assertEqual(len(actual),8)

    def test_metadata_endpoint_persists_completion_after_unique_registry(self):
        args,_=self.fixture()
        from pt_agent import paths
        with patch.object(paths,'RESULTS_DIR',self.root), \
                patch.object(worker.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='fixture')), \
                patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=100*worker.GIB)):
            self.assertEqual(worker.execute(args),0)
        result=json.loads((args.out/'result.json').read_text())
        complete=json.loads((args.out/'completion.json').read_text())
        registry=[json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(registry),1);self.assertEqual(len(registry[0]),8)
        self.assertEqual(result['status'],'prepared');self.assertEqual(result['api_calls'],0)
        self.assertTrue(complete['registry_persisted'])
        self.assertEqual(complete['case_artifact_bytes'],worker.artifact_bytes(args.out))
        self.assertFalse(result['generation_protocol']['actual_two_token_execution_completed'])

    def test_first_phase_failure_stops_before_prompt_api_or_emit(self):
        args,_=self.fixture();args.through='compile'
        from pt_agent import paths
        process=dict(exit_code=1,timed_out=False,signal=None,remaining_pids=[])
        with patch.object(paths,'RESULTS_DIR',self.root), \
                patch.object(worker.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='fixture')), \
                patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=100*worker.GIB)), \
                patch.object(worker,'run_process',return_value=process) as run:
            self.assertEqual(worker.execute(args),1)
        result=json.loads((args.out/'result.json').read_text())
        self.assertEqual(run.call_count,1)
        self.assertEqual(result['attempted']['calibration'],1)
        self.assertEqual(result['attempted']['step'],0);self.assertEqual(result['attempted']['emit'],0)
        self.assertEqual(result['api_calls'],0);self.assertTrue((args.out/'completion.json').is_file())

    def test_api_registration_is_qwen_and_deduplicates_explicit_ledger(self):
        from pt_agent import autonomous_budget
        registry=self.root/'registry.jsonl';ledger=self.root/'other-ledger.jsonl'
        calls=[dict(ts='fixture',purpose='test',model='provider-model')]
        with patch.object(autonomous_budget,'call_records',return_value=calls):
            first=worker.register_api_calls(ledger,registry,self.root/'case/phases/step',before=0,after=1,git_rev='fixture')
            second=worker.register_api_calls(ledger,registry,self.root/'case/phases/step',before=0,after=1,git_rev='fixture')
        rows=[json.loads(line) for line in registry.read_text().splitlines()]
        self.assertEqual(first['registered'],[1]);self.assertEqual(second['retained'],[1])
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['model'],worker.MODEL)
        self.assertEqual(rows[0]['round'],worker.ROUND);self.assertEqual(len(rows[0]),8)

    def test_two_forwards_keep_actual_embedding_handle_and_one_terminal_output(self):
        # Run the existing generate_two control flow on symbolic tiny handles.
        # No slot arithmetic, tokenizer, model or library backend is instantiated.
        name='pt_agent.experiments._qwen_generation_flow_fixture'
        path=Path(worker.__file__).parents[1]/'kernels/encrypted_generation.py'
        spec=importlib.util.spec_from_file_location(name,path);generation=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{name:generation}):spec.loader.exec_module(generation)
        class Value:
            def __init__(self,backend,label):self.backend,self.label=backend,label
            def __mul__(self,other):return Value(self.backend,'product')
            def __add__(self,other):return Value(self.backend,'sum')
        class Backend:
            slots=16384
            def __init__(self):self.inputs=[];self.outputs=None
            def input(self,index):
                self.assertion(index==len(self.inputs));value=Value(self,'input'+str(index));self.inputs.append(value);return value
            @staticmethod
            def assertion(value):
                if not value:raise AssertionError('consecutive initial inputs')
            def const(self,value):return object()
            def output(self,values):
                if self.outputs is not None:raise AssertionError('only final output')
                self.outputs=list(values)
        class Prefix:
            def __init__(self,backend,tokens,length,hidden,capacity):
                self.backend,self.token_blocks,self.prompt_length,self.selections=backend,tokens,length,[]
            @property
            def length(self):return self.prompt_length+len(self.selections)
            def public_masks(self):return dict(attention_mask=[[0.]*32],sel=[[0.]*32])
            def append(self,embedding,selection):
                self.token_blocks[self.length]=embedding;self.selections.append(selection)
        class Refresh:
            def __init__(self,inner,**kwargs):self.inner=inner
            def stats(self):return dict(boots=0)
        selectors=[];embeddings=[];forward_inputs=[]
        def select(backend,logits,vocab,**kwargs):
            value=Value(backend,'selection'+str(len(selectors)));selectors.append(value)
            return dict(one_hot=[value],metadata=dict(vocabulary=vocab))
        def embed(backend,selection,weights):
            value=Value(backend,'embedding'+str(len(embeddings)));embeddings.append(value)
            return [value],None
        generation.EncryptedPrefix=Prefix;generation.select_maximum=select;generation.embedding_from_selection=embed
        def lower(program,weights,active,**kwargs):
            row=[active.inner.input(i) for i in range(34)];forward_inputs.append(row)
            output=[Value(active.inner,'logits'+str(len(forward_inputs)))];active.inner.output(output)
            return output,[object()]
        fake_numpy=SimpleNamespace(float64=object(),zeros=lambda *args,**kwargs:[0.]*16384)
        modules={'pt_agent.backends':SimpleNamespace(RefreshBackend=Refresh),
            'pt_agent.kernels.encrypted_generation':generation,
            'pt_agent.kernels.compact_linear':SimpleNamespace(compact_transformer_projections=nullcontext),
            'pt_agent.lower':SimpleNamespace(lower=lower),'numpy':fake_numpy}
        backend=Backend();program=SimpleNamespace(header=SimpleNamespace(slots=16384),params=[
            SimpleNamespace(name='inputs_embeds',shape=(1,32,896)),
            SimpleNamespace(name='attention_mask'),SimpleNamespace(name='sel')])
        weights={'original.model.embed_tokens.weight':SimpleNamespace(shape=(151936,896))}
        with patch.dict(sys.modules,modules),patch.object(worker,'_compact_generation',nullcontext):
            result=worker.lower_two_tokens(program,weights,backend,dict(levels={}),9,[])
        self.assertEqual(len(backend.inputs),34);self.assertEqual(len(forward_inputs),2)
        self.assertIs(forward_inputs[1][9],embeddings[0])
        self.assertIs(forward_inputs[1][0],forward_inputs[0][0])
        self.assertEqual(len(selectors),2);self.assertEqual(len(embeddings),2)
        self.assertEqual(backend.outputs,selectors);self.assertEqual(result['model_forward_calls'],2)
        self.assertEqual(result['generated_tokens'],2);self.assertFalse(result['kv_cache_used'])

    def test_source_rejects_wrong_prepared_input_binding(self):
        args,_=self.fixture();path=args.inputs.parent/'worker_result.json';record=json.loads(path.read_text())
        record['inputs_jsonl']=str(self.root/'other.jsonl');put(path,record)
        with self.assertRaisesRegex(ValueError,'preparation paths'):worker.source_gate(args)

    def test_cli_preserves_host_visible_nix_interpreter_symlink(self):
        interpreter=self.root/'python';interpreter.symlink_to('/nix/store/not-mounted-on-host/python3')
        with patch.object(worker,'execute',return_value=0) as execute:
            self.assertEqual(worker.main(['--out',str(self.root/'new'),'--through','metadata',
                '--intake-python',str(interpreter)]),0)
        actual=execute.call_args.args[0].intake_python
        self.assertEqual(actual,interpreter.absolute());self.assertTrue(actual.is_symlink())
        self.assertNotEqual(actual,interpreter.resolve())

    def test_local_as24_is_virtual_only_and_default12_remains(self):
        base=['--out',str(self.root/'unused'),'--through','metadata']
        with patch.object(worker,'execute',return_value=0) as execute:
            self.assertEqual(worker.main(base),0)
            self.assertEqual(execute.call_args.args[0].memory_gib,12.)
            self.assertEqual(worker.main(base+['--memory-gib','24']),0)
            args=execute.call_args.args[0]
            self.assertEqual(args.memory_gib,24.);self.assertEqual(args.rss_gib,14.)
            command=worker.phase_command(args,'calibration',self.root/'phase')
            self.assertEqual(command[command.index('--memory-gib')+1],'24.0')
        with patch('sys.stderr',new_callable=io.StringIO),self.assertRaises(SystemExit) as failed:
            worker.main(base+['--memory-gib','25'])
        self.assertEqual(failed.exception.code,2)


if __name__=='__main__':unittest.main()
