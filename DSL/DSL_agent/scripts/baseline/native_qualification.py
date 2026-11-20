"""Native qualification through the existing session, tracer and feedback loop."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from hecate_python_env import ROOT,WORK,digest
from qualification_session import QualificationSession,read_json,SOURCE
from benchmark_graph import require
from native_execution_profile import validate_request,NAME


class NativeQualificationSession(QualificationSession):
    def __init__(self,result,request,report,frozen,fixed,waterline,*,
                 helper_environment=None,unified=True,validation_level='numerical'):
        from poseidon_runtime_paths import checked_sdk,MANIFEST
        from poseidon_backend_identity import identity
        from application_identity import source_closure,source_catalog,_catalog_values,CATALOG
        from native_sdk_manifest import ready_for_agent_qualification,execution_fingerprint
        validate_request(request)
        require(helper_environment is None and unified and waterline==44,'Native qualification scope')
        require(validation_level in ('compiled','numerical'),'Native acceptance level')
        self.sdk=checked_sdk()
        require(type(self.sdk)is dict and 'native'in self.sdk and
                ready_for_agent_qualification(self.sdk),'Measured Native SDK precondition')
        self.result,self.request,self.report=Path(result),copy.deepcopy(request),report
        self.frozen,self.fixed,self.waterline=frozen,copy.deepcopy(fixed),waterline
        self.helper_environment=None;self.validation_level=validation_level;self.unified=True
        require(self.result.resolve().is_relative_to((WORK/'results').resolve())and
                self.fixed['atol']==1e-5 and self.fixed['rtol']==1e-4,'Original result/tolerance boundary')
        self.backend=identity();self.descriptor=Path(MANIFEST)
        self.sdk_fingerprint=execution_fingerprint(self.sdk)
        baseline=ROOT/'scripts/baseline'
        self._roots=('native_qualification.py','native_validation.py','native_sandbox.py','native_argument_binding.py')
        self._catalog=digest(baseline/CATALOG)
        self._closure=source_closure(baseline,self._roots,
            dynamic_authorities=_catalog_values(source_catalog(baseline))[2])
        self._fixtures=digest(SOURCE/'fixtures.json')
        from native_execution_slots import NATIVE_CONCURRENCY
        report.setdefault('native_resources',dict(concurrency=NATIVE_CONCURRENCY,acquisitions=0,wait_seconds=0.))
        report.update(backend_identity=copy.deepcopy(self.backend),native_sdk_execution_fingerprint=self.sdk_fingerprint,
            source_hashes=copy.deepcopy(self._closure['sources']),compiler_profile_sha256=self.sdk['native']['compiler_profile']['sha256'],
            waterline=44,compiler_configuration=copy.deepcopy(request['compiler_configuration']),
            compiler_sha256=self.sdk['artifacts']['compiler'],frontend_sha256=self.sdk['artifacts']['frontend'],
            runtime_sha256=self.sdk['native']['runtime']['sha256'],client_sha256=self.sdk['native']['client']['sha256'],
            operator_spec_sha256=self.sdk['native']['operator_spec']['sha256'],
            tolerance=dict(atol=1e-5,rtol=1e-4),
            qualification=dict(format='poseidon-native-qualification-session-v1',validation_level=validation_level,
                               generation_performed=False,source_catalog_sha256=self._catalog))
        self._report_binding={key:copy.deepcopy(report[key])for key in ['backend_identity','native_sdk_execution_fingerprint',
            'source_hashes','compiler_profile_sha256','waterline','compiler_configuration','compiler_sha256',
            'frontend_sha256','runtime_sha256','client_sha256','operator_spec_sha256','tolerance','qualification']}
        from native_validation import NativeValidationAdapter
        self.validation=NativeValidationAdapter(self)

    def verify_frozen(self):
        from poseidon_runtime_paths import checked_sdk
        from poseidon_backend_identity import identity
        from application_identity import source_closure,source_catalog,_catalog_values,CATALOG
        baseline=ROOT/'scripts/baseline'
        require(checked_sdk()==self.sdk and identity()==self.backend,'Frozen actual Native SDK/toolchain')
        require(digest(baseline/CATALOG)==self._catalog and
            source_closure(baseline,self._roots,dynamic_authorities=_catalog_values(source_catalog(baseline))[2])==
            self._closure,'Frozen Native/normal source authority')
        require(digest(SOURCE/'fixtures.json')==self._fixtures and read_json(SOURCE/'fixtures.json')==self.fixed,
                'Original fixed numerical tolerance')
        require(all(self.report.get(key)==value for key,value in self._report_binding.items()),
                'Host Native qualification binding')
        require(all((self.result/name).resolve().is_relative_to(self.result.resolve())and
                    not(self.result/name).is_symlink()and digest(self.result/name)==value
                    for name,value in self.frozen.items()),'Frozen request/model/arrays/candidate payload')
        validate_request(self.request)


def qualify(request,candidate,*,validation_level='numerical',artifact_observer=None):
    from component_contract import failure
    code,report,evidence=NativeQualificationSession.qualify(request,candidate,
        validation_level=validation_level,artifact_observer=artifact_observer)
    last=report.get('attempts',[])[-1]if report.get('attempts')else{}
    intact=report.get('failure_layer')!='integrity'and last.get('failure_layer')!='integrity'
    output=dict(format='poseidon-agent-component-v1',status=report['status'],exit_code=code,
        request_id=request['request_id'],evidence=str(evidence),paid_calls=0,new_agent_generation=False,
        encrypted_execution=last.get('executed')is True and intact,numerically_validated=False,
        observed_stages=dict(traced=isinstance(last.get('trace'),dict),compiled=last.get('compiled')is True,
            encrypted_execution=last.get('executed')is True,numerically_compared=isinstance(last.get('comparison'),dict),
            evidence_integrity_not_rejected=intact))
    if report['status']=='passed':
        from native_validation import audit_candidate
        audit=audit_candidate(evidence,validation_level=validation_level)
        output.update(validation_level=validation_level,compiled_validated=True,native_sdk=True)
        if validation_level=='numerical':
            output.update(encrypted_execution=True,numerically_validated=True,
                max_absolute_error=audit['comparison']['max_absolute_error'],
                compared_values=audit['comparison']['compared_values'])
    else:
        layer='integrity'if not intact else last.get('failure_layer',report.get('failure_layer','pipeline'))
        output['failure']=failure(layer,last.get('diagnostic',report.get('diagnostic','Native qualification failed')))
    return output
