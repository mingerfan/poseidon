"""Trusted launcher branch tests with fake credentials, no Nix/network execution."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/"scripts/baseline/run_candidate.py"
PENDING=Path(__file__).resolve().parent/"pending-patches"
PINNED=Path("/test-only/pinned-venv")
def load_main(proposed):
    source=SOURCE.read_text()
    pin=json.loads((PENDING/"run-candidate-nested-nix.json").read_text())
    if hashlib.sha256(source.encode()).hexdigest()!=pin["source_sha256"]:raise ValueError("Proposal source drift")
    if proposed:
        marker='            return enter_nix(command, seconds=1800 + generation_deadline(args.api_timeout, args.max_repairs + 1, args.provider_retries),'
        addition='''            # A benchmark child may already run in the exact pinned pure shell.
            # Re-entering nix-portable here would deadlock on the parent's
            # exclusive launcher lock. Keep credential loading/cleanup here,
            # then call the same trusted execution path in this environment.
            if os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV:
                return inside(args)
'''
        if source.count(marker)!=1:raise ValueError("Proposal target drift")
        source=source.replace(marker,addition+marker)
        if hashlib.sha256(source.encode()).hexdigest()!=pin["proposed_sha256"]:raise ValueError("Proposal hash")
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="main")
    return ast.Module(body=[node],type_ignores=[])

class NestedNixLauncherTests(unittest.TestCase):
    def run_branch(self,proposed=True,nix=True,pinned=True,prior=None,fail=False):
        calls=[]
        args=types.SimpleNamespace(inside=False,deepseek=True,provider="deepseek",
            api_timeout=1200,max_repairs=3,provider_retries=0)
        def require(condition,message):
            if not condition:raise ValueError(message)
        def inside(args):
            calls.append("inside")
            self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"),"FAKE_TEST_ONLY")
            if fail:raise RuntimeError("fake execution failure")
            return 17
        def enter(*args,**kwargs):
            calls.append("enter_nix")
            self.assertEqual(kwargs.get("keep_env"),("DEEPSEEK_API_KEY",))
            return 23
        creds=types.ModuleType("agent_credentials")
        creds.CredentialError=ValueError
        def load(*args,**kwargs):calls.append("load_fake_key");return "FAKE_TEST_ONLY"
        creds.load_api_key=load
        provider=types.ModuleType("deepseek_provider");provider.generation_deadline=lambda *args:4800
        namespace=dict(parse_args=lambda:args,require=require,Path=Path,ROOT=Path.cwd(),os=os,
            sys=types.SimpleNamespace(prefix=str(PINNED if pinned else Path("/test-only/wrong-venv")),stderr=sys.stderr),
            VENV=PINNED,SCRIPT=SOURCE,forward_options=lambda args:"--live",shlex=shlex,enter_nix=enter,inside=inside)
        with patch.dict(sys.modules,{"agent_credentials":creds,"deepseek_provider":provider}),patch.dict(os.environ,{},clear=True):
            if nix:os.environ["IN_NIX_SHELL"]="pure"
            if prior is not None:os.environ["DEEPSEEK_API_KEY"]=prior
            exec(compile(load_main(proposed),str(SOURCE), "exec"),namespace)
            if fail:
                with self.assertRaisesRegex(RuntimeError,"fake execution failure"):namespace["main"]()
                result=None
            else:result=namespace["main"]()
            self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"),prior)
        return calls,result
    def test_current_code_reenters_nix_from_pinned_shell(self):
        calls,result=self.run_branch(proposed=False)
        self.assertEqual(calls,["load_fake_key","enter_nix"]);self.assertEqual(result,23)
    def test_proposal_reuses_exact_pinned_shell(self):
        calls,result=self.run_branch()
        self.assertEqual(calls,["load_fake_key","inside"]);self.assertEqual(result,17)
    def test_wrong_venv_still_enters_isolated_environment(self):
        self.assertEqual(self.run_branch(pinned=False)[0],["load_fake_key","enter_nix"])
    def test_outside_nix_still_enters_pure_shell(self):
        self.assertEqual(self.run_branch(nix=False)[0],["load_fake_key","enter_nix"])
    def test_proposal_restores_previous_environment_even_on_error(self):
        self.assertEqual(self.run_branch(prior="FAKE_PREVIOUS",fail=True)[0],["load_fake_key","inside"])
if __name__=="__main__":unittest.main()
