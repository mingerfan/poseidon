"""Staging of the ordinary CPU selector with the already approved repair.

The compiler/executor are unchanged. The outer record preserves library-overlay
provenance; the inner v2 record is only the original runtime selector. Neither
record upgrades the pending normal or Bootstrap numerical qualification.
"""
import copy
from pathlib import Path

import poseidon_sdk_manifest as base
from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once

FORMAT = 'poseidon-normal-cpu-approved-overlay-v1'
ALLOWED_CHANGES = {'poseidon_build', 'artifacts', 'cmake_caches', 'qualification', 'binding'}


def ordinary_parameter_files(agent_root):
    # scripts/baseline is a retained repository compatibility entry point.
    # Freeze its actual files; the artifact reader intentionally rejects aliases.
    return tuple((Path(agent_root)/'scripts/baseline'/name).resolve(strict=True)
                 for name in ('poseidon-compiler-profile-v1.json','poseidon-operator-spec-v1.json'))


def selector(ancestor, parent, *, library_view=None):
    """Keep source/compile ancestry and replace only the explicit library binding."""
    require(type(ancestor) is dict and set(ancestor) == base.FIELDS and ancestor['format'] == base.FORMAT,
            'Original ordinary CPU compiler ancestry')
    require(parent['format'] == 'poseidon-runtime-toolchain-native-v5'
            and ancestor['binding'] == parent['native']['base_sdk']['binding'], 'Explicit compiler ancestry binding')
    candidate = parent['library_overlay']['candidate']
    require(parent['artifacts']['poseidon_library'] == candidate['sha256']
            and candidate['sha256'] != base.REJECTED_KEYGEN_LIBRARY_SHA256, 'Only approved repaired library')
    value = copy.deepcopy(ancestor)
    value.pop('binding', None)
    value['poseidon_build'] = parent['poseidon_build'] if library_view is None else str(Path(library_view).absolute())
    value['artifacts']['poseidon_library'] = candidate['sha256']
    cache = Path(value['poseidon_build'])/'CMakeCache.txt'
    if not cache.exists(): cache = cache.parent.parent/'CMakeCache.txt'
    require(base.file_hash(Path(value['poseidon_build'])/'libposeidon_shared.so') == candidate['sha256'],
            'Materialized selected library differs from approved candidate')
    require(base.file_hash(cache) == ancestor['cmake_caches']['poseidon'],
            'Original configuration ancestry must be retained')
    value['cmake_caches']['poseidon'] = ancestor['cmake_caches']['poseidon']
    value['qualification'] = 'pending_agent_acceptance'
    return bound(value)


def check_selector(value, ancestor, parent):
    checked_bound(value)
    require(set(value) == base.FIELDS and value == selector(ancestor, parent, library_view=value['poseidon_build']), 'Ordinary CPU selector differs from frozen overlay')
    require(all(value[k] == ancestor[k] for k in base.FIELDS-ALLOWED_CHANGES), 'Compiler/source/environment ancestry changed')
    require(all(value['artifacts'][k] == ancestor['artifacts'][k] for k in ('compiler', 'frontend', 'executor')),
            'Original compiler/frontend/executor must be unchanged')
    require(value['cmake_caches']['compiler'] == ancestor['cmake_caches']['compiler']
            and value['cmake_caches']['executor'] == ancestor['cmake_caches']['executor'], 'Original compiler/executor build identity')
    return value


def _parent(path, expected_sha256, work, platform, agent_root):
    path = Path(path).absolute()
    require(sha(read_bytes(path)) == expected_sha256, 'Approved parent descriptor changed')
    import native_sdk_manifest_v5
    parent = native_sdk_manifest_v5.validate(base.read_json(path), work, platform, agent_root)
    # Read and attest public library/build/source provenance. No native execution,
    # old-library selection, private key read or numerical promotion occurs.
    native_sdk_manifest_v5.attest(parent, work, agent_root)
    ancestor_record = parent['native']['base_sdk']
    require(sha(read_bytes(ancestor_record['path'])) == ancestor_record['sha256'], 'Original compiler descriptor changed')
    ancestor = checked_bound(base.read_json(ancestor_record['path']), ancestor_record['binding'])
    return parent, ancestor


def stage(parent_path, output, *, expected_parent_sha256, work, platform, agent_root):
    """Materialize a distinct pending selector; no existing SDK is changed."""
    parent, ancestor = _parent(parent_path, expected_parent_sha256, work, platform, agent_root)
    root = Path(output).absolute()
    require(root.parent.resolve() == root.parent and root.is_relative_to(Path(work)/'results'), 'Accounted new SDK staging directory')
    root.mkdir(mode=0o700)
    # The approved overlay was compiled/linked from retained original objects;
    # it has no new CMake configure step. This explicit SDK view materializes
    # its DSO and the unchanged configuration-ancestry cache. The outer record
    # retains the real overlay build commands and source proof separately.
    library_view = root/'library-view'; library_view.mkdir(mode=0o700)
    library_source = Path(parent['library_overlay']['candidate']['path'])
    original_cache = Path(ancestor['poseidon_build'])/'CMakeCache.txt'
    if not original_cache.exists(): original_cache = original_cache.parent.parent/'CMakeCache.txt'
    require(base.file_hash(original_cache) == ancestor['cmake_caches']['poseidon'], 'Actual original configuration cache')
    write_once(library_view/'libposeidon_shared.so', read_bytes(library_source, 16*1024**2))
    write_once(library_view/'CMakeCache.txt', read_bytes(original_cache, 16*1024**2))
    materialization = dict(format='poseidon-approved-overlay-readonly-SDK-view-v1',
        library_source=copy.deepcopy(parent['library_overlay']['candidate']),
        library_destination=dict(path=str(library_view/'libposeidon_shared.so'),sha256=base.file_hash(library_view/'libposeidon_shared.so')),
        configuration_ancestry=dict(path=str(original_cache),sha256=base.file_hash(original_cache)),
        configuration_view=dict(path=str(library_view/'CMakeCache.txt'),sha256=base.file_hash(library_view/'CMakeCache.txt')),
        new_CMake_configuration_performed=False, real_overlay_build=copy.deepcopy(parent['library_overlay']['build_report']))
    inner = selector(ancestor, parent, library_view=library_view)
    inner_path = root/'ordinary-cpu-selector.json'
    write_once(inner_path, canonical(inner))
    check_selector(inner, ancestor, parent)
    base.attest(base.load(inner_path, work, platform), work, agent_root)
    from compiler_configuration import CURRENT_CONFIGURATION, configuration
    cfg = configuration(CURRENT_CONFIGURATION)
    profile, spec = ordinary_parameter_files(agent_root)
    require(sha(read_bytes(profile)) == cfg['ckks_config_sha256']
            and sha(read_bytes(spec)) == cfg['operator_spec_sha256']
            and cfg['source_lock_sha256'] == inner['source_lock_sha256'], 'Original registered CPU parameters/profile')
    outer = bound(dict(format=FORMAT, parent=dict(path=str(Path(parent_path).absolute()), sha256=expected_parent_sha256, binding=parent['binding']),
        inner_selector=dict(path=str(inner_path), sha256=sha(read_bytes(inner_path)), binding=inner['binding']),
        compiler_ancestor=copy.deepcopy(parent['native']['base_sdk']), approved_library_overlay=copy.deepcopy(parent['library_overlay']),
        artifact_materialization=materialization,
        ordinary_compiler_configuration=cfg,
        profile=dict(path=str(profile.resolve()), sha256=sha(read_bytes(profile))),
        operator_spec=dict(path=str(spec.resolve()), sha256=sha(read_bytes(spec))),
        reader_sha256=sha(read_bytes(__file__)),
        pending=dict(ordinary_CPU_numeric_acceptance=True, Agent_first_output=True, full_network=True),
        bootstrap_certified=False, normal_Application_promoted=False,
        key_lifecycle='unchanged ordinary CPU acceptance owner; process-local ephemeral keys, no persistent epoch',
        source_scope='compiler/executor unchanged; approved repaired DSO and its exact source/build overlay kept separate from upstream ancestry'))
    path = root/'normal-cpu-overlay.json'; write_once(path, canonical(outer))
    return dict(path=str(path), sha256=sha(read_bytes(path)), binding=outer['binding'])


def verify(descriptor, *, work, platform, agent_root):
    require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256', 'binding'}, 'Normal CPU overlay descriptor')
    require(sha(read_bytes(descriptor['path'])) == descriptor['sha256'], 'Normal overlay descriptor changed')
    outer = checked_bound(base.read_json(descriptor['path']), descriptor['binding'])
    require(outer['format'] == FORMAT and outer['reader_sha256'] == sha(read_bytes(__file__)), 'Normal overlay reader identity')
    parent, ancestor = _parent(outer['parent']['path'], outer['parent']['sha256'], work, platform, agent_root)
    require(outer['parent']['binding'] == parent['binding'] and outer['compiler_ancestor'] == parent['native']['base_sdk']
            and outer['approved_library_overlay'] == parent['library_overlay'], 'Actual approved library/source provenance')
    record = outer['inner_selector']
    require(sha(read_bytes(record['path'])) == record['sha256'], 'Ordinary selector changed')
    inner = base.load(record['path'], work, platform); checked_bound(inner, record['binding'])
    check_selector(inner, ancestor, parent); base.attest(inner, work, agent_root)
    materialization = outer['artifact_materialization']
    require(materialization['new_CMake_configuration_performed'] is False
            and materialization['library_source'] == parent['library_overlay']['candidate']
            and materialization['real_overlay_build'] == parent['library_overlay']['build_report'], 'Actual overlay versus configuration ancestry')
    for key in ('library_source', 'library_destination', 'configuration_ancestry', 'configuration_view'):
        item=materialization[key]
        require(base.file_hash(item['path']) == item['sha256'], 'SDK-view original/materialized bytes changed')
    require(materialization['library_destination']['path'] == str(Path(inner['poseidon_build'])/'libposeidon_shared.so')
            and materialization['configuration_view']['path'] == str(Path(inner['poseidon_build'])/'CMakeCache.txt'), 'Exact materialized SDK paths')
    for key in ('profile', 'operator_spec'):
        item = outer[key]
        require(sha(read_bytes(item['path'])) == item['sha256'], 'Registered ordinary CPU parameter file changed')
    from compiler_configuration import CURRENT_CONFIGURATION, configuration
    require(outer['ordinary_compiler_configuration'] == configuration(CURRENT_CONFIGURATION), 'Original ordinary CPU configuration changed')
    require(outer['bootstrap_certified'] is False and outer['normal_Application_promoted'] is False
            and all(outer['pending'].values()), 'Staging cannot promote numerical/Agent acceptance')
    return dict(outer=outer, inner=inner, library=Path(inner['poseidon_build'])/'libposeidon_shared.so',
                numerical_or_Agent_acceptance=False)
