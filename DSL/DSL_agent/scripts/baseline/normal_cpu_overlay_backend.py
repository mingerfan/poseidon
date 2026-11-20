"""External first-output qualification with the approved ordinary-CPU DSO.

The original fixed-candidate session, compiler, auditor and arithmetic run intact.
This adapter only binds its source/evidence and selects the attested DSO inside
the original sandbox. It cannot generate, repair or substitute candidate source.
"""
import copy
from pathlib import Path

from application_backend import PoseidonBackend
from delivery_component import canonical, checked_bound, read_bytes, require, sha
import normal_cpu_overlay_evidence as evidence
import normal_cpu_overlay_sandbox as sandbox
import normal_cpu_overlay_sdk as sdk


def _sources():
    from application_identity import source_catalog,_catalog_values,source_closure
    from hecate_python_env import ROOT
    root=(ROOT/'scripts/baseline').resolve()
    authorities=_catalog_values(source_catalog(root))[2]
    return source_closure(root,[Path(__file__).name],dynamic_authorities=authorities)['sources']


class NormalCpuOverlayBackend(PoseidonBackend):
    """Admission for further independently measured ordinary-CPU requests only."""
    artifact_monitor_version='poseidon-normal-CPU-approved-overlay-monitor-v1'

    def __init__(self, overlay, canary_plan, canary_result, *, compiler_evaluation_profile=None, evaluation_inputs=None):
        self.overlay=copy.deepcopy(overlay)
        self.canary_plan=copy.deepcopy(canary_plan)
        self.canary_result=copy.deepcopy(canary_result)
        self.compiler_evaluation_profile=copy.deepcopy(compiler_evaluation_profile)
        self.evaluation_inputs=copy.deepcopy(evaluation_inputs)

    def _evidence(self):
        from hecate_python_env import WORK,ROOT
        from platform_config import configuration
        from poseidon_runtime_paths import MANIFEST
        verified=evidence.verify(self.overlay,self.canary_plan,self.canary_result,
            work=WORK,platform=configuration()['id'],agent_root=ROOT)
        context=sdk.verify(self.overlay,work=WORK,platform=configuration()['id'],agent_root=ROOT)
        if self.compiler_evaluation_profile is not None:
            from compiler_managed_evaluation import verify_profile
            verify_profile(self.compiler_evaluation_profile)
        require(str(MANIFEST)==context['outer']['inner_selector']['path'],
                'Select the frozen ordinary CPU SDK before worker imports')
        require(verified['ordinary_CPU_test_admission'] is True,
                'Actual ordinary-CPU canary numerical acceptance required')
        return context,verified

    def identity_manifest(self):
        from hecate_python_env import ROOT
        from poseidon_backend_identity import identity
        from application_identity import identity_manifest
        context,measured=self._evidence()
        low=identity()
        require(low['poseidon_library_sha256']==context['inner']['artifacts']['poseidon_library']
                and low['executor_sha256']==context['inner']['artifacts']['executor'], 'Selected actual ordinary CPU tools')
        low['normal_CPU_approved_overlay']=dict(SDK=self.overlay,canary_evidence_binding=measured['binding'],
            original_configuration=measured['ordinary_configuration'],adapter_source_sha256=_sources(),
            scope='independent original ordinary-CPU trials; no Bootstrap or whole-SDK numerical promotion')
        if self.compiler_evaluation_profile is not None:
            low['external_compiler_evaluation_profile']=copy.deepcopy(self.compiler_evaluation_profile)
        if self.evaluation_inputs is not None:
            low['external_evaluation_inputs']=copy.deepcopy(self.evaluation_inputs)
        return identity_manifest(ROOT,low)

    def check_request(self,request,level):
        context,_=self._evidence()
        expected=(context['outer']['ordinary_compiler_configuration'] if self.compiler_evaluation_profile is None
                  else self.compiler_evaluation_profile['configuration'])
        require(request.get('compiler_configuration')==expected,
                'Request differs from the independently frozen external compiler configuration')
        return super().check_request(request,level)

    def qualify(self,request,candidate,level,*,artifact_observer=None):
        self.check_request(request,level)
        from qualification_session import QualificationSession
        from native_execution_slots import native_slot
        from hecate_python_env import WORK,ROOT
        from first_output_backend import _json
        backend=self
        measured=self._evidence()[1]
        frozen=self.identity_manifest()
        sources=_sources()

        class BoundOverlaySession(QualificationSession):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self._overlay_record=dict(SDK=copy.deepcopy(backend.overlay),identity=frozen,
                    adapter_source_sha256=sources,original_fixed_candidate_session=True,post_output_repairs=0)
                self.report['normal_CPU_approved_overlay']=copy.deepcopy(self._overlay_record)

            def verify_frozen(self):
                super().verify_frozen()
                require(self.report['normal_CPU_approved_overlay']==self._overlay_record,
                        'Qualification overlay report changed')
                require(_sources()==sources and backend._evidence()[1]==measured,
                        'Qualification adapter/SDK/canary evidence changed')

            def native_run(self,*args,**kwargs):
                context,_=backend._evidence()
                if self.helper_environment is not None:kwargs['helpers']=True
                with native_slot(WORK/'cache/agent-native-slots',metrics=self.report['native_resources']):
                    return sandbox.run(*args,**kwargs,selected_library=context['library'])

        options={} if self.evaluation_inputs is None else dict(evaluation_inputs=self.evaluation_inputs)
        code,report,folder=BoundOverlaySession.qualify(request,candidate,validation_level=level,
                                                     artifact_observer=artifact_observer,**options)
        require(self.identity_manifest()==frozen, 'Frozen overlay backend changed during qualification')
        from component_backend import observed_stages
        stages=observed_stages(report)
        output=dict(format='poseidon-agent-component-v1',status=report['status'],exit_code=code,
            request_id=request['request_id'],evidence=str(folder),paid_calls=0,new_agent_generation=False,
            encrypted_execution=stages['encrypted_execution'] and stages['evidence_integrity_not_rejected'],
            numerically_validated=False,observed_stages=stages,normal_CPU_overlay=True)
        if report['status']=='passed':
            if level=='numerical':
                context,_=self._evidence();attempt=folder/'attempt-00';out=attempt/'output'
                actual=sandbox.verify_execution_libraries(out,attempt/'execute.log',context['library'])
                if actual['format']=='poseidon-ordinary-CPU-grouped-selected-DSO-v1':
                    output['native_group_library_evidence']=actual
            from bounded_candidate_audit import verify_candidate
            audit=verify_candidate(folder,validation_level=level,artifact_observer=artifact_observer)
            require(audit['request_id']==request['request_id'],'Original auditor request mismatch')
            output.update(validation_level=level,compiled_validated=True)
            if level=='numerical':output.update(numerically_validated=True,encrypted_execution=True,
                max_absolute_error=audit['comparison']['max_absolute_error'],compared_values=audit['comparison']['compared_values'])
        else:
            from component_contract import failure
            attempt=report['attempts'][0] if report['attempts'] else report
            layer='integrity' if report.get('failure_layer')=='integrity'else attempt.get('failure_layer',report.get('failure_layer','pipeline'))
            output['failure']=failure(layer,attempt.get('diagnostic',report.get('diagnostic',report['status'])))
        return output

    def export(self,result,destination,level,*,model_provenance=None,generation_provenance=None,model_source=None):
        from normal_cpu_overlay_package import export
        return export(self,result,destination,level,model_provenance=model_provenance,
            generation_provenance=generation_provenance,model_source=model_source)

    def check_package(self,folder,level):
        from normal_cpu_overlay_package import check
        return check(self,folder,level)[0]['binding']

    def execute(self,folder,samples,*,artifact_observer=None,sample_format='named'):
        from normal_cpu_overlay_package import execute
        return execute(self,folder,samples,artifact_observer=artifact_observer,sample_format=sample_format)


def trusted_backend(backend, *, approved_overlay_required=False):
    """An explicit closed set; arbitrary subclasses/callbacks cannot admit live HE."""
    require(type(backend) in (PoseidonBackend,NormalCpuOverlayBackend), 'Audited original or approved-overlay backend required')
    require(type(approved_overlay_required) is bool and
            (not approved_overlay_required or type(backend) is NormalCpuOverlayBackend),
            'Approved overlay admission must use its audited sandbox adapter')
    if type(backend) is NormalCpuOverlayBackend:backend._evidence()
    return backend
