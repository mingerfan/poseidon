"""Bootstrap frontend dataflow tests, explicitly not a backend implementation."""
import ast
from collections.abc import Iterable
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from probe_bootstrap_frontend import SOURCE, dataflow, STYLES


def recording_frontend():
    import numpy as np
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unaryFactory')
    class Expr:
        def __init__(self,obj):self.obj=obj
    calls=[]
    def create(ctxt,opcode,value,*loc):
        result=('bootstrap',value,len(calls));calls.append(result);return result
    space=dict(Expr=Expr,np=np,inspect=inspect,Iterable=Iterable,ctxt=None,
               lt=SimpleNamespace(createUnary=create))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),space)
    space['unaryFactory']('bootstrap',0)
    return Expr,space['bootstrap'],calls


class BootstrapFrontendTests(unittest.TestCase):
    def test_scalar_is_a_new_expression(self):
        Expr,boot,calls=recording_frontend();x=Expr('x');y=boot(x)
        self.assertEqual(x.obj,'x');self.assertIsNot(x,y)
        self.assertEqual(y.obj,('bootstrap','x',0));self.assertEqual(len(calls),1)

    def test_mutable_list_identity_and_old_expression_alias(self):
        Expr,boot,calls=recording_frontend();x=Expr('x');value=[x,Expr('y')];alias=value
        self.assertIs(boot(value),value);self.assertIs(alias,value)
        self.assertEqual([v.obj[1] for v in value],['x','y'])
        self.assertEqual(x.obj,'x');self.assertEqual(len(calls),2)

    def test_tuple_returns_replaced_entries_without_mutating_original(self):
        Expr,boot,calls=recording_frontend();value=(Expr('x'),Expr('y'));result=boot(value)
        self.assertIsInstance(result,tuple);self.assertIsNot(result,value)
        self.assertEqual([v.obj for v in value],['x','y'])
        self.assertEqual([v.obj[1] for v in result],['x','y'])

    def test_ndarray_shapes_and_noncontiguous_view(self):
        import numpy as np
        for shape in ((),(1,),(2,2),(1,2,1,2)):
            Expr,boot,calls=recording_frontend();a=np.empty(shape,dtype=object)
            for i in range(a.size):a.flat[i]=Expr(i)
            self.assertIs(boot(a),a);self.assertEqual(a.shape,shape)
            self.assertEqual([x.obj[1] for x in a.flat],list(range(a.size)))
        Expr,boot,calls=recording_frontend();a=np.empty(4,dtype=object)
        a[:]=[Expr(i) for i in range(4)];view=a[::-2]
        self.assertIs(boot(view),view)
        self.assertEqual([x.obj for x in a], [0,('bootstrap',1,1),2,('bootstrap',3,0)])

    def test_invalid_cells_and_readonly_fail_before_emission(self):
        import numpy as np
        Expr,boot,calls=recording_frontend();x=Expr('x')
        values=([x,None],[x,[x]],np.array([x,None],dtype=object),np.array([1.,2.]),
                (v for v in [x]),{'x':x},None)
        for value in values:
            with self.subTest(type=type(value).__name__),self.assertRaises(TypeError):boot(value)
            self.assertEqual(calls,[])
        a=np.array([x],dtype=object);a.flags.writeable=False
        with self.assertRaises(TypeError):boot(a)
        self.assertEqual(calls,[]);self.assertEqual(x.obj,'x')

    def test_empty_containers_have_no_invented_bootstraps(self):
        import numpy as np
        Expr,boot,calls=recording_frontend()
        for value in ([],(),np.empty((2,0),dtype=object)):
            result=boot(value);self.assertEqual(len(result),len(value))
        self.assertEqual(calls,[])

    def test_dataflow_rejects_dead_operations_and_tracks_transitive_use(self):
        source='%0 = "earth.bootstrap"(%arg0) : () -> ()\n%1 = "earth.bootstrap"(%arg1) : () -> ()\n'
        dead=dataflow(source+'"func.return"(%arg0) : () -> ()')
        self.assertEqual(dead['live_bootstrap_results'],[])
        live=dataflow(source+'%2 = "earth.add"(%0, %1) : () -> ()\n"func.return"(%2) : () -> ()')
        self.assertEqual(live['live_bootstrap_results'],['%0','%1'])

    def test_agent_bootstrap_remains_closed(self):
        from hecate_contract import validate_function
        for contract in ('hecate-function-v1','hecate-function-v21'):
            with self.subTest(contract=contract),self.assertRaises(ValueError):
                validate_function('@hc.func("c")\ndef golden(x):\n    return hc.bootstrap(x)\n',{},contract=contract)


@unittest.skipUnless(os.environ.get('POSEIDON_BOOTSTRAP_FRONTEND_BEFORE') and
                     os.environ.get('POSEIDON_BOOTSTRAP_FRONTEND_AFTER'),'requires real frontend trace evidence')
class BootstrapFrontendEvidenceTests(unittest.TestCase):
    def test_actual_before_after_ir_hashes_and_reachable_operations(self):
        for var,expected in [('POSEIDON_BOOTSTRAP_FRONTEND_BEFORE','failed'),
                             ('POSEIDON_BOOTSTRAP_FRONTEND_AFTER','passed')]:
            root=Path(os.environ[var]);report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['status'],expected)
            self.assertFalse(report['encrypted_execution']);self.assertFalse(report['bootstrap_execution_validated'])
            self.assertEqual(report['agent_calls'],0)
            self.assertEqual([c['style'] for c in report['cases']],list(STYLES))
            for row in report['cases']:
                if row['trace_exit']!=0:
                    self.assertEqual(expected,'failed');continue
                raw=(root/row['style']/'probe_bootstrap_frontend.mlir').read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['earth_sha256'])
                self.assertEqual(dataflow(raw.decode()),row['dataflow'])
                self.assertEqual(row['verify_exit'],0)
                self.assertIn('--verify-each',row['verify_command'])
            if expected=='passed':
                self.assertTrue(all(c['passed'] for c in report['cases']))
                self.assertEqual(report['frontend_sha256'],hashlib.sha256(SOURCE.read_bytes()).hexdigest())
            else:
                self.assertEqual([c['style'] for c in report['cases'] if c['passed']],['scalar'])


if __name__=='__main__':unittest.main()
