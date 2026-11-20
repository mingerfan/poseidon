"""Default computational delivery route; compilation is an external service."""
import copy
from pathlib import Path
import delivery_component as delivery
from delivery_component import bound,canonical,require,sha,write_once
from compiler_configuration import COMPILER_MANAGED_CONFIGURATION
from compiler_managed_evaluation import verify_profile


def prepare(model,options=None):
    """Static model/ABI admission only; no refresh schedule or numerical gate."""
    from component_contract import prepare_task
    selected=copy.deepcopy(options or {})
    selected.setdefault('compiler_configuration',COMPILER_MANAGED_CONFIGURATION)
    require(selected['compiler_configuration']==COMPILER_MANAGED_CONFIGURATION,
            'Computational delivery uses the independent compiler-managed configuration')
    from compiler_managed_layout import select
    selected=select(model,selected)
    request=prepare_task(model,selected)
    # prepare_task returns the original canonical request, not a backend verdict.
    return request


def generator_manifest(version,source_paths,*,profile,**kwargs):
    verify_profile(profile)
    return delivery.generator_manifest(version,[Path(__file__).absolute(),*source_paths],
        compiler_evaluation_profile=profile,**kwargs)


def generate_once(request,provider,output,*,profile,mechanism,**kwargs):
    verify_profile(profile)
    require(request['compiler_configuration']==profile['configuration']
            and mechanism.get('compiler_evaluation_profile')==profile,'Frozen computational delivery/profile binding')
    return delivery.generate_once(request,provider,output,mechanism=mechanism,**kwargs)
