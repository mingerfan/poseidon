"""Portable, explicitly typed parent packages derived from genuine Conv children.

Prepared packages carry no execution/numerical authority. Their ordered children
retain original manifests, identities and generation receipts byte for byte.
The old bundle reader continues to decide each child's current compatibility.
"""
import copy
import hashlib
import os
from pathlib import Path
import stat
import sys

import conv_partition_cover as cover
import cpu_conv_composition_intake as intake
import cpu_ssa_linker as link
import release_planner
from cpu_composition_abi import abi_mapping

FORMAT = 'poseidon-cpu-Conv-composition-package-v1'
REQUEST = 'poseidon-cpu-Conv-composition-request-v1'
PROFILE = 'cpu-same-key-Conv-kernel-sum-v2'
POLICY = 'retain-original-children-requalify-parent-v1'
MAX_BYTES = 512 * 1024**2
MAX_FILES = 4096
SOURCES = ('conv_partition_cover.py', 'cpu_conv_composition_intake.py', 'cpu_ssa_linker.py',
           'release_planner.py', 'cpu_composition_abi.py', 'cpu_composition_package.py',
           'cpu_composition_execution.py', 'application_composition.py', 'cpu_composition_verified.py')

need, canonical, digest, sha, loads = intake.need, intake.canonical, intake.digest, intake.sha, intake.loads


def regular(path, limit=16 * 1024**2):
    path = Path(path)
    need(path.is_absolute() and all(not p.is_symlink() for p in (path, *path.parents)), 'direct absolute package path')
    need(not any(part in {'.env', '.ssh', '.aws'} for part in path.parts) and
         path.suffix not in {'.pem', '.key'}, 'private material is not a composition artifact')
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'bounded regular package file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    need(len(raw) <= limit, 'bounded package read')
    return raw


def bound(value):
    return intake.bound(value)


def seal(value):
    return dict(value, binding=digest(value))


def write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(all(not p.is_symlink() for p in (path, *path.parents)), 'direct package output')
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(raw)


def authority():
    from application_identity import current_identity_manifest
    identity = current_identity_manifest()
    here = Path(__file__).resolve().parent
    for name in SOURCES:
        module = sys.modules.get(name[:-3])
        need(module is None or Path(module.__file__).resolve() == here / name,
             'composition loaded module origin')
    return dict(format='poseidon-CPU-composition-authority-v1', profile=PROFILE,
        sources={name: sha(regular(here / name)) for name in SOURCES},
        application_manifest_sha256=identity['manifest_sha256'],
        stages={stage: identity[stage]['sha256'] for stage in intake.STAGES},
        backend_identity_sha256=digest(identity['backend']))


def child(index, folder, binding, model_sha256):
    from candidate_bundle import check_execution_dependencies
    from application_identity import check_manifest
    from generation_receipt import verify
    folder = Path(folder)
    def read(name):
        return regular(folder / intake.relative(name))
    original = bound(loads(read('manifest.json')))
    need(original['binding'] == binding, 'original full child bundle binding')
    result = intake.inspect_child(index, read, expected_package_binding=binding,
        expected_model_sha256=model_sha256, verify_identities=check_manifest,
        verify_generation=verify, derive_abi=abi_mapping)
    dependency = 'passed'
    try:
        checked, _ = check_execution_dependencies(folder)
        need(checked['binding'] == binding, 'current child binding')
    except ValueError as error:
        # Only this precise original stage refusal can be retained as pending.
        # Other dependency, source-lock, compiler or helper failures reject intake.
        need(str(error) == 'Bundle stage identity changed; needs explicit requalification',
             'unrecognized child dependency refusal')
        dependency = str(error)
    from poseidon_backend_identity import identity
    need(original['dependencies']['backend_identity'] == identity(), 'child actual SDK dependencies')
    plan_raw = read('artifacts/bound/plan.json')
    plan = loads(plan_raw)
    rotations = sorted({op['attrs']['steps'] for op in plan['execution'] if op.get('op') == 'rotate'})
    def blob(name):
        link.content_name(name)
        return read('artifacts/bound/bundle/data/' + name[7:] + '.bin')
    return result, dict(id='component' + str(index), plan_raw=plan_raw,
        spec_raw=read('artifacts/operator-spec.json'), bundle_raw=read('artifacts/bound/bundle/manifest.json'),
        rotations=rotations, read_blob=blob, io=result['io']), original, dependency


def selected_model(parent, node_id, output_name):
    node = next(node for node in parent['nodes'] if node['id'] == node_id)
    return dict(parent, nodes=[copy.deepcopy(node)],
        constants={key: copy.deepcopy(parent['constants'][key]) for key in node['inputs'][1:]},
        inputs=[copy.deepcopy(item) for item in parent['inputs'] if item['name'] == node['inputs'][0]],
        outputs=[dict(name=output_name, value=node['outputs'][0])])


def derive(parent_raw, node_id, entries):
    from cpu_composition_verified import current_operation
    operation = current_operation()
    if operation is not None:
        return operation.derive(parent_raw, node_id, entries)
    return _derive_full(parent_raw, node_id, entries)


def _derive_full(parent_raw, node_id, entries):
    """Trusted preparation only; original absolute inputs never enter the result."""
    need(type(entries) is list and 2 <= len(entries) <= 9, 'ordered composition entries')
    pieces = []
    for i, row in enumerate(entries):
        need(type(row) is dict and set(row) == {'folder', 'binding', 'kernel_positions'}, 'composition entry fields')
        intake.hash_text(row['binding'])
        pieces.append(dict(index=i, model_raw=regular(Path(row['folder']) / 'model.json'),
                           kernel_positions=copy.deepcopy(row['kernel_positions'])))
    proof = cover.verify_cover(parent_raw, source_node_id=node_id, pieces=pieces)
    records, children, manifests, dependencies = [], [], [], []
    for i, row in enumerate(entries):
        record, data, manifest, dependency = child(i, row['folder'], row['binding'], proof['pieces'][i]['model_sha256'])
        records.append(record); children.append(data); manifests.append(manifest); dependencies.append(dependency)
    need(len({r['program_binding'] for r in records}) == len(records), 'unique child packages')
    configuration = manifests[0]['compiler_configuration']
    need(all(m['compiler_configuration'] == configuration for m in manifests), 'one original compiler configuration')
    model = selected_model(loads(parent_raw), node_id, proof['output']['name'])
    layout = loads(regular(Path(entries[0]['folder']) / 'request.json'))['layout']
    need(model['inputs'] == layout['logical_inputs'] and
         layout['outputs'][0]['shape'] == proof['output']['shape'] and
         layout['outputs'][0]['name'] == proof['output']['name'], 'parent logical ABI')
    key = digest(dict(parent=sha(parent_raw), node=node_id, children=[r['program_binding'] for r in records],
                      positions=[row['kernel_positions'] for row in entries]))
    linked = link.link_sum(children, plan_id=str(int(key[:15], 16)), bundle_id='composition-' + key)
    rotations = sorted(set(step for item in children for step in item['rotations']))
    v1 = link.canonical(linked['plan'])
    released = release_planner.insert_sparse_releases(v1, children[0]['spec_raw'], rotations)
    request = seal(dict(format=REQUEST, profile=PROFILE, historical_artifact_policy=POLICY,
        source_parent_raw_sha256=sha(parent_raw), source_node_id=node_id,
        scope='selected source convolution only', model_sha256=digest(model),
        compiler_configuration=configuration, layout=layout, cover_binding=proof['binding'],
        children=[dict(index=i, program_binding=r['program_binding'],
            manifest_raw_sha256=r['original_manifest_raw_sha256'], model_sha256=r['model_sha256'],
            generation_receipt_sha256=r['generation_receipt_sha256'],
            application_identities_sha256=r['application_identities_sha256'],
            stages=r['original_stage_sha256'], io_sha256=r['io_sha256'],
            kernel_positions=copy.deepcopy(entries[i]['kernel_positions']),
            original_current_dependency_result=dependencies[i]) for i, r in enumerate(records)],
        authority=authority(), addition_order=linked['proof']['addition_order'],
        plan_v1_raw_sha256=sha(v1), plan_v2_raw_sha256=sha(released['raw']),
        bundle_manifest_raw_sha256=sha(linked['bundle_raw']), spec_raw_sha256=sha(children[0]['spec_raw']),
        ssa_binding=digest(linked['proof']), release_binding=released['proof']['binding'],
        agent_generated=False, derived_from_genuine_children=True))
    return dict(request=request, model=model, cover=proof, children=records, entries=entries,
        manifests=manifests, inputs=children, linked=linked, plan_v1=v1, released=released)


def reuse_key(request):
    bound(request)
    need(request.get('format') == REQUEST and request.get('profile') == PROFILE and
         request.get('historical_artifact_policy') == POLICY and
         request.get('agent_generated') is False and request.get('derived_from_genuine_children') is True,
         'composition request discriminator')
    # Hash every request field, including order, cover, ABI, derived bytes and
    # current compositor/stage identities. No legacy model-only cache alias.
    return digest(dict(format='poseidon-composition-reuse-v1', request=request))


def export_prepared(parent_raw, node_id, entries, destination, *, checkpoint=lambda: None):
    data = derive(parent_raw, node_id, entries)
    destination = Path(destination)
    need(destination.is_absolute() and not destination.exists() and not destination.is_symlink(), 'fresh composition package')
    destination.mkdir(mode=0o700)
    inventory, total = {}, 0
    def put(name, raw):
        nonlocal total
        intake.relative(name)
        total += len(raw)
        need(total <= MAX_BYTES and len(inventory) < MAX_FILES, 'parent package storage bound')
        checkpoint()
        write(destination / name, raw)
        inventory[name] = sha(raw)
    put('source-parent.json', parent_raw)
    for name, value in [('request.json', data['request']), ('model.json', data['model']),
        ('cover.json', data['cover']), ('ssa.json', data['linked']['proof']),
        ('release.json', data['released']['proof'])]:
        put(name, canonical(value))
    put('plan.v1.json', data['plan_v1']); put('plan.v2.json', data['released']['raw'])
    put('operator-spec.json', data['inputs'][0]['spec_raw'])
    put('bundle/manifest.json', data['linked']['bundle_raw'])
    for i, (row, manifest) in enumerate(zip(entries, data['manifests'])):
        folder = Path(row['folder'])
        for name, expected in [('manifest.json', data['request']['children'][i]['manifest_raw_sha256']), *manifest['files'].items()]:
            raw = regular(folder / intake.relative(name))
            need(sha(raw) == expected, 'child changed during export')
            put('children/' + str(i) + '/' + name, raw)
    for content, source in data['linked']['bundle_sources'].items():
        raw = data['inputs'][source['child']]['read_blob'](content)
        need(len(raw) == source['byte_length'] and sha(raw) == content[7:], 'original public blob bytes')
        put('bundle/data/' + content[7:] + '.bin', raw)
    need(authority() == data['request']['authority'], 'composition authority changed during export')
    manifest = seal(dict(format=FORMAT, profile=PROFILE, files=inventory,
        request_binding=data['request']['binding'], reuse_key=reuse_key(data['request']),
        validation=dict(level='prepared', encrypted_execution=False, numerically_validated=False),
        agent_generated=False, derived_from_genuine_children=True))
    write(destination / 'manifest.json', canonical(manifest))
    checkpoint()
    return manifest


def check_bundle(folder, *, level='prepared', checkpoint=lambda: None):
    folder = Path(folder)
    manifest = bound(loads(regular(folder / 'manifest.json', 1024**2)))
    need(set(manifest) == {'format', 'profile', 'files', 'request_binding', 'reuse_key', 'validation',
                          'agent_generated', 'derived_from_genuine_children', 'binding'} and
         manifest['format'] == FORMAT and manifest['profile'] == PROFILE and
         manifest['agent_generated'] is False and manifest['derived_from_genuine_children'] is True,
         'composition package discriminator')
    need(level in ('prepared', 'numerical'), 'composition validation grade')
    need(type(manifest['validation']) is dict and manifest['validation'].get('level') in ('prepared', 'numerical'),
         'composition grade discriminator')
    expected_grade = dict(level=manifest['validation']['level'],
        encrypted_execution=manifest['validation']['level'] == 'numerical',
        numerically_validated=manifest['validation']['level'] == 'numerical')
    need(canonical(manifest['validation']) == canonical(expected_grade), 'strict composition grade')
    if level == 'numerical':
        need(canonical(manifest['validation']) == canonical(dict(level='numerical', encrypted_execution=True, numerically_validated=True)),
             'fresh parent requalification required')
    files = manifest['files']
    need(type(files) is dict and 1 <= len(files) <= MAX_FILES, 'composition inventory bound')
    actual, total = set(), 0
    for path in folder.rglob('*'):
        need(not path.is_symlink(), 'indirect composition artifact')
        info = path.lstat()
        need(stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), 'composition artifact type')
        if stat.S_ISREG(info.st_mode):
            actual.add(str(path.relative_to(folder)))
            total += info.st_size
            need(len(actual) <= MAX_FILES + 1 and total <= MAX_BYTES, 'composition aggregate storage bound')
    need(actual == {'manifest.json', *files}, 'exact composition inventory')
    for name, expected in files.items():
        intake.relative(name); intake.hash_text(expected)
        need(sha(regular(folder / name)) == expected, 'composition artifact changed')
    request = bound(loads(regular(folder / 'request.json')))
    need(request['binding'] == manifest['request_binding'] and reuse_key(request) == manifest['reuse_key'],
         'parent request/cache binding')
    need(request['authority'] == authority(), 'current composition authority changed')
    entries = [dict(folder=folder / 'children' / str(i), binding=row['program_binding'],
                    kernel_positions=row['kernel_positions']) for i, row in enumerate(request['children'])]
    derived = derive(regular(folder / 'source-parent.json'), request['source_node_id'], entries)
    need(canonical(derived['request']) == canonical(request), 'composition request rederivation')
    for name, raw in [('model.json', canonical(derived['model'])), ('cover.json', canonical(derived['cover'])),
        ('ssa.json', canonical(derived['linked']['proof'])), ('release.json', canonical(derived['released']['proof'])),
        ('plan.v1.json', derived['plan_v1']), ('plan.v2.json', derived['released']['raw']),
        ('operator-spec.json', derived['inputs'][0]['spec_raw']), ('bundle/manifest.json', derived['linked']['bundle_raw'])]:
        need(regular(folder / name) == raw, 'exact parent derivation bytes')
    # The composed bundle must also contain exactly the source-derived contents.
    expected = {'bundle/data/' + name[7:] + '.bin' for name in derived['linked']['bundle_sources']}
    need({name for name in files if name.startswith('bundle/data/')} == expected, 'composed blob inventory')
    for content, source in derived['linked']['bundle_sources'].items():
        raw = regular(folder / ('bundle/data/' + content[7:] + '.bin'))
        need(len(raw) == source['byte_length'] and sha(raw) == content[7:], 'composed blob content address')
    need(level in ('prepared', 'numerical'), 'composition validation grade')
    validation = manifest['validation']
    if validation['level'] == 'prepared':
        need(canonical(validation) == canonical(dict(level='prepared', encrypted_execution=False, numerically_validated=False)), 'prepared grade')
    else:
        from cpu_composition_execution import verify_qualification
        verify_qualification(folder, manifest, request)
    need(level == 'prepared' or validation['level'] == 'numerical', 'fresh parent requalification required')
    checkpoint()
    return manifest, request


def check_execution_dependencies(folder, *, checkpoint=lambda: None):
    manifest, request = check_bundle(folder, level='numerical', checkpoint=checkpoint)
    need(request['historical_artifact_policy'] == POLICY, 'parent requalification policy')
    # Historical child refusals remain in request.json. New authority belongs
    # solely to this freshly qualified parent, never to a re-signed child.
    return manifest, request
