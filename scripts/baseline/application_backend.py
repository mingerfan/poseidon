"""Trusted pinned-worker bridge. No benchmark index, batch or historical report."""
from pathlib import Path
import hashlib
from benchmark_graph import digest,require
from application_policy import BACKEND

class SealBackend:
    name=BACKEND
    def identity(self):
        from platform_config import identity
        from python_compiler_smoke import BUILD
        from hecate_python_env import ROOT
        root=ROOT/"scripts/baseline"
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        # Bind implementation, both platform locks and actual loaded SDK binaries.
        paths=list(root.glob("*.py"))+list(root.glob("*.json"))+list(root.glob("patches/*.patch"))
        paths+=list((ROOT/"src/poseidon/tools/dacapo").glob("*lock.json"))
        paths+=list((ROOT/"third_party/dacapo/python/hecate").rglob("*.py"))
        paths+=list((ROOT/"third_party/dacapo/python/poly").rglob("*.py"))
        paths+=[BUILD/"bin/hecate-opt",BUILD/"lib/libHecateFrontend.so",BUILD/"lib/libSEAL_HEVM.so"]
        return dict(backend=self.name,platform=identity(),dependency_digest=digest({str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:sha(p) for p in paths}))
    def qualify(self,request,candidate,level):
        from component_backend import qualify
        return qualify(request,candidate,validation_level=level)
    def export(self,result,destination,level):
        from candidate_bundle import export_bundle
        return export_bundle(Path(result["evidence"]),destination,validation_level=level)
    def check_package(self,folder,level):
        from candidate_bundle import check_bundle
        m=check_bundle(folder)
        actual=m.get("validation",{}).get("level","numerical")
        require(actual==level,"Package acceptance level mismatch")
        return m["binding"]
