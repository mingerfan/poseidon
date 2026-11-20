"""Standalone deterministic SQLite scalar tests; no actual model or replay."""
from contextlib import contextmanager
import gc
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import weakref

from pt_agent.experiments import qwen_sqlite_ids as storage
from pt_agent.experiments.qwen_grouped_ids import GroupedIds


@contextmanager
def no_cyclic_gc():
    enabled=gc.isenabled();gc.disable()
    try:yield
    finally:
        if enabled:gc.enable()


class SQLiteIdsChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.parent=Path(self.temp.name)

    def mapping(self,**kwargs):
        value=storage.SQLiteIds(self.parent,**kwargs);self.addCleanup(value.close);return value

    def test_encoding_equal_shared_and_unshared_structures_and_roundtrip(self):
        pair=('cipher',2**137);shared=(pair,pair)
        independent=(tuple(list(pair)),tuple(list(pair)))
        self.assertIsNot(shared[0],independent[0])
        self.assertEqual(storage.encode_key(('mul_cc',shared,None)),storage.encode_key(('mul_cc',independent,None)))
        keys=[('add',(),None),('add',(),0),('add',(),-1),('rot',(('cipher',2**100),),2**99),
              ('add',(('plain','k\x00"\\\ud800'),('cipher',7)),None),('add',(('cipher',7),('plain','k\x00"\\\ud800')),None)]
        self.assertEqual(len(set(map(storage.encode_key,keys))),len(keys))
        for key in keys:self.assertEqual(storage.decode_key(storage.encode_key(key)),key)

    def test_invalid_keys_bool_and_noncanonical_encoded_fields_reject_before_mutation(self):
        value=self.mapping();invalid=[[],('x',[],None),('x',(('cipher',True),),None),
            ('x',(('cipher',-1),),None),('x',(('plain',7),),None),('x',(('bad',1),),None),
            ('x',((True,1),),None),(False,(),None),('x',(),True),('x',(),1.5)]
        for key in invalid:
            with self.subTest(key=key),self.assertRaises(ValueError):value[key]=0
        for nid in (True,-1,1.5):
            with self.subTest(nid=nid),self.assertRaises(ValueError):value[('x',(),None)]=nid
        self.assertEqual(len(value),0);self.assertEqual(list(value.directory.iterdir()),[])
        good=storage.encode_key(('x',(),None))
        for encoded in (b'',good[:-1],good+b'garbage',storage.PREFIX+b'\x80\x00'):
            with self.subTest(encoded=encoded),self.assertRaises(ValueError):storage.decode_key(encoded)

    def test_order_update_delete_reinsert_clear_and_unbounded_values_match_dict_and_grouped(self):
        disk=self.mapping();original={};grouped=GroupedIds()
        keys=[('add',(('cipher',3),('plain','k0')),None),('rot',(('cipher',3),),9),
              ('add',(('cipher',5),('cipher',5)),None),('rot',(('cipher',3),),-9)]
        for mapping in (original,grouped,disk):
            for i,key in enumerate(keys):mapping[key]=i+2**120
            mapping[keys[0]]=7;del mapping[keys[1]];mapping[keys[1]]=99
        self.assertEqual(list(disk.items()),list(original.items()));self.assertEqual(list(disk.items()),list(grouped.items()))
        self.assertEqual(disk[keys[2]],2**120+2);self.assertEqual(disk.pop(keys[0]),original.pop(keys[0]))
        self.assertEqual(dict(disk),original)
        missing=('boot',(('cipher',404),),None)
        with self.assertRaises(KeyError):del disk[missing]
        with self.assertRaises(KeyError):disk[missing]
        with patch.object(disk,'_lookup',side_effect=AssertionError('len must stay O1')):self.assertEqual(len(disk),3)

    def test_iterator_value_updates_and_structural_mutation_fail_fast(self):
        value=self.mapping();a=('add',(('cipher',1),),None);b=('rot',(('cipher',2),),1)
        value[a]=1;iterator=iter(value);value[a]=2;self.assertEqual(next(iterator),a)
        iterator=iter(value);value[b]=3
        with self.assertRaises(RuntimeError):next(iterator)
        iterator=iter(value);del value[a]
        with self.assertRaises(RuntimeError):next(iterator)
        iterator=iter(value);value.clear()
        with self.assertRaises(RuntimeError):next(iterator)

    def test_fresh_private_lifecycle_no_foreign_reopen_and_exact_SQL_settings(self):
        foreign=self.parent/'ids.sqlite';foreign.write_text('foreign metadata')
        value=self.mapping();key=('rot',(('cipher',1),),1);value[key]=1
        first=value._state['path'];connection=value._state['connection']
        for name,expected in [('cache_size',-storage.CACHE_KIB),('mmap_size',0),('temp_store',1)]:
            self.assertEqual(connection.execute('PRAGMA '+name).fetchone()[0],expected)
        self.assertEqual(connection.execute('PRAGMA journal_mode').fetchone()[0],'delete')
        self.assertEqual(connection.execute('PRAGMA max_page_count').fetchone()[0],value.max_bytes//storage.PAGE_BYTES)
        value.clear();self.assertFalse(first.exists());self.assertNotIn(key,value)
        value[key]=2;self.assertNotEqual(value._state['path'],first)
        directory=value.directory;value.close();value.close();self.assertFalse(directory.exists())
        self.assertEqual(foreign.read_text(),'foreign metadata')
        with self.assertRaises(storage.SQLiteStorageError):value[key]=1
        second=self.mapping();self.assertNotEqual(second.directory,directory);self.assertNotIn(key,second)

    def test_SQL_failure_discards_mapping_and_disk_full_is_bounded_failure(self):
        value=self.mapping()
        with patch.object(storage.sqlite3,'connect',side_effect=sqlite3.OperationalError('fixture unavailable')):
            with self.assertRaises(storage.SQLiteStorageError):value[('neg',(('cipher',0),),None)]=0
        self.assertTrue(value.snapshot()['closed']);self.assertFalse(value.directory.exists())
        tiny=self.mapping(max_bytes=4*storage.PAGE_BYTES)
        with self.assertRaises(storage.SQLiteStorageError):
            for index in range(2000):tiny[('add',(('cipher',index),('plain','k'+str(index)+'x'*100)),None)]=index
        self.assertTrue(tiny.snapshot()['closed']);self.assertEqual(len(tiny),0);self.assertFalse(tiny.directory.exists())

    def test_exception_context_and_gc_disabled_release_without_owner_cycle(self):
        with self.assertRaisesRegex(ValueError,'fixture'):
            with storage.SQLiteIds(self.parent)as value:
                value[('neg',(('cipher',0),),None)]=0;directory=value.directory
                raise ValueError('fixture')
        self.assertFalse(directory.exists())
        with no_cyclic_gc():
            value=storage.SQLiteIds(self.parent);value[('neg',(('cipher',1),),None)]=1
            directory=value.directory;reference=weakref.ref(value);del value
            self.assertIsNone(reference());self.assertFalse(directory.exists())

    def test_constructor_budget_and_update_validation_preserve_existing_content(self):
        for allowance in (True,float('nan'),0,2**50):
            with self.subTest(allowance=allowance),self.assertRaises(ValueError):self.mapping(max_bytes=allowance)
        value=self.mapping();key=('neg',(('cipher',1),),None);value[key]=9
        with self.assertRaises(ValueError):value[key]=False
        self.assertEqual(value[key],9);self.assertEqual(len(value),1)


class SmallOriginalGraphChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pt_agent.experiments.r29_local_compile_matrix import native_metadata_commands
        template=['{hecate_opt}','{input}','--pars','--ckks-config={package}/config/compiler-config.json',
            '--waterline=43','--output-val=10','--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
            '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,'-o','{output_dir}/lowered.mlir']
        cls.metadata=native_metadata_commands(template);cls.metadata.__enter__()
        import numpy as np
        from pt_agent import cost,plan
        from pt_agent.backends import MockBackend,RefreshBackend
        from pt_agent.experiments import qwen_agent_cost_lifetime as lifetime
        cls.np=np;cls.cost=cost;cls.plan=plan;cls.Mock=MockBackend;cls.Refresh=RefreshBackend;cls.lifetime=lifetime

    @classmethod
    def tearDownClass(cls):cls.metadata.__exit__(None,None,None)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.parent=Path(self.temp.name)

    def planner(self):
        life=self.lifetime.Lifetime();namespace=self.lifetime.private_planner(life);original=namespace['CostDag'];mappings=[]
        def factory(*a,**k):
            active=original(*a,**k);value=storage.SQLiteIds(self.parent);mappings.append(value)
            active.numbering.ids=value;return active
        namespace['CostDag']=factory;return namespace,life,mappings

    @staticmethod
    def toy(active):
        x,y=active.input(0),active.input(1);active.op_context=0;active.stage('first')
        a=x*x;z=a-(x*x)+y
        for _ in range(7):z=z*z+x*.125
        active.op_context=1;active.stage('last');z=z.rotate(-1)+z.rotate(1)-y
        dead=z*z;active.output([z,z*.5])

    def test_original_ScalarNumbering_live_identity_dead_nid_and_cleanup(self):
        planner,life,mappings=self.planner()
        with no_cyclic_gc():
            active=planner['CostDag'](64);x=active.input(0);a=x*x;nid=a.nid
            self.assertIs(x*x,a);reference=weakref.ref(a);del a;self.assertIsNone(reference())
            recreated=x*x;self.assertEqual(recreated.nid,nid);self.assertEqual(len(active.kind),2);self.assertIs(x*x,recreated)
            owner=weakref.ref(active);numbering=active.numbering;del x,recreated
            self.lifetime.release_backend(active);del active
            self.assertIsNone(owner());self.assertIsNone(numbering.owner());self.assertEqual(len(mappings[0]),0)
            mappings[0].close()

    def test_original_DAG_plan_Count_and_strict_Mock_outputs_equal(self):
        original,old_record,_=self.plan.record_dag(self.toy,64,frontend_normalize=True)
        planner,life,mappings=self.planner();new,record,_=planner['record_dag'](self.toy,64,frontend_normalize=True)
        try:
            for name in ('kind','args','prog_op','operands','steps','_depth','_total_depth','normalization'):
                self.assertEqual(getattr(original,name),getattr(new,name),name)
            self.assertEqual(original.stats(),new.stats());self.assertEqual(old_record.calls,record.calls)
            plan=self.plan.express(original,old_record,self.cost.depth_dp_estimate(original))
            observed=planner['express'](new,record,self.cost.depth_dp_estimate(new));self.assertEqual(plan,observed)
            a=self.plan.verify(self.toy,64,plan);b=planner['verify'](self.toy,64,observed)
            self.assertEqual({k:v for k,v in a.items()if k!='apply_s'},{k:v for k,v in b.items()if k!='apply_s'})
            outputs=[]
            for candidate in (None,observed):
                backend=self.Mock(64);backend.bind([self.np.linspace(-.125,.125,64),self.np.linspace(.25,.375,64)])
                active=self.Refresh(backend,policy='plan',plan=candidate,**self.plan.levels())if candidate else backend
                self.toy(active);outputs.append([value.value.copy()for value in backend._outputs])
            for expected,actual in zip(*outputs):
                self.assertTrue(self.np.all(self.np.abs(expected-actual)<=1e-12+1e-12*self.np.abs(expected)))
            self.assertTrue(all(len(value)==0 for value in mappings))
        finally:
            life.finish_records();self.lifetime.release_record(original,old_record)
            self.lifetime.release_backend(original);self.lifetime.release_backend(new)
            for value in mappings:value.close()


if __name__=='__main__':unittest.main()
