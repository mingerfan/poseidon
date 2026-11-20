"""Read current CPU CKKS SSA/export/layout data without compiling or executing.

Every supported operation follows the original RuntimePlan exporter. This is a
compiler-data correspondence proof, never an SDK, arithmetic-error or HE claim.
"""
import ast
import copy
import math
import re
import struct

from delivery_component import bound, canonical, checked_bound, require, sha

FORMAT = 'poseidon-compiler-SSA-runtime-layout-provenance-v1'
POLY = r'tensor<1x!ckks\.poly<([0-9]+)\s*\*\s*([0-9]+)\s*\*\s*([0-9]+)>>'
SSA = r'%[A-Za-z0-9_.$-]+'
MAX_VALUES = 8192
MAX_OPERATIONS = 4096
OPS = dict(addcc='add_cc', addcp='add_cp', mulcc='mul_cc', mulcp='mul_cp',
           negatec='negate', relinearize='relinearize', rotatec='rotate',
           rescalec='rescale', modswitchc='mod_switch', bootstrapc='boot')


def split_top(text):
    """Split an MLIR list with nested type/attribute syntax and quoted strings."""
    levels = []; quoted = False; escape = False; rows = []; start = 0
    pairs = {')':'(', ']':'[', '}':'{', '>':'<'}
    for index, char in enumerate(text):
        if quoted:
            if escape: escape = False
            elif char == '\\': escape = True
            elif char == '"': quoted = False
            continue
        if char == '"': quoted = True
        elif char in '([{<': levels.append(char)
        elif char in ')]}>':
            require(levels and levels.pop() == pairs[char], 'Balanced MLIR list')
        elif char == ',' and not levels: rows.append(text[start:index].strip()); start = index+1
    require(not quoted and not levels, 'Terminated MLIR list')
    tail = text[start:].strip()
    if tail: rows.append(tail)
    return rows


def attributes(text):
    text = text.strip()
    if not text: return {}
    require(text.startswith('<{') and text.endswith('}>'), 'Original generic MLIR property syntax')
    rows = split_top(text[2:-2]); result = {}
    for row in rows:
        key, mark, value = row.partition('='); key = key.strip()
        require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', key) and key not in result, 'Unique MLIR property')
        result[key] = value.strip() if mark else None
    return result


def integer(value):
    require(type(value) is str and re.fullmatch(r'-?[0-9]+\s*:\s*i64', value), 'Original i64 property')
    return int(value.split(':')[0])


def boolean(value):
    require(value in ('true','false'), 'Original boolean property')
    return value == 'true'


def poly(text):
    match = re.fullmatch(POLY, text.strip())
    require(match is not None, 'One physical CKKS polynomial per exported value')
    c, s, level = map(int, match.groups()); require(1 <= c <= 3 and 0 <= s <= 1048576 and 0 <= level <= 1048576, 'Finite polynomial metadata')
    return dict(components=c, scale_log2=s, level=level, kind='plaintext' if c == 1 else 'ciphertext')


def dense(value):
    match = re.fullmatch(r'dense<(.*)>\s*:\s*tensor<((?:[0-9]+x)+)(f64|i64)>', value, re.S)
    require(match is not None, 'Explicit bounded original dense payload')
    dims = [int(n) for n in match[2][:-1].split('x')]; count = math.prod(dims)
    require(all(1 <= n <= 16384 for n in dims) and 1 <= count <= 16384, 'Original dense payload extent')
    body = match[1].strip()
    if body.startswith('"0x') and body.endswith('"'):
        raw = bytes.fromhex(body[3:-1]); require(match[3] == 'f64' and len(raw) == count*8, 'Original LE dense-f64 buffer')
        values = list(struct.unpack('<'+'d'*count, raw))
    else:
        flat = body.replace('[','').replace(']','')
        tokens = [x.strip() for x in flat.split(',')]
        values = []
        for token in tokens:
            if match[3] == 'i64':
                require(re.fullmatch(r'-?[0-9]+',token), 'Dense i64 literal'); number = float(int(token))
            elif re.fullmatch(r'0x[0-9a-fA-F]{16}',token): number = struct.unpack('>d',bytes.fromhex(token[2:]))[0]
            else:
                require(re.fullmatch(r'[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',token), 'Dense f64 literal')
                number = float(token)
            values.append(number)
        if len(values) == 1: values *= count
        require(len(values) == count, 'Complete dense payload, including tails')
    require(all(math.isfinite(n) for n in values), 'Finite public payload')
    return struct.pack('<'+'d'*len(values), *values)


def command_context(command):
    require(type(command) is list and command and command[0] == '/hecate-opt' and all(type(x) is str for x in command), 'Original compiler command')
    def flag(name, default=None):
        rows = [x.split('=',1)[1] for x in command if x.startswith(name+'=')]
        require(len(rows) == 1 or not rows and default is not None, 'Unique compiler target flag: '+name)
        return rows[0] if rows else default
    require(flag('--runtime-plan-format','json') == 'json', 'Audited JSON export')
    require(flag('--runtime-plan-ntt') in ('true','false'), 'NTT flag')
    return dict(context=flag('--runtime-plan-context-id'), ntt=flag('--runtime-plan-ntt') == 'true',
        boot_profile=flag('--boot-profile',''), boot_implementation=flag('--runtime-plan-boot-implementation','native'),
        plan_id=flag('--runtime-plan-id'), target_id=flag('--runtime-plan-target-id'),
        capability_version=int(flag('--runtime-plan-capability-version')),
        operator_spec=dict(id=flag('--runtime-plan-operator-spec-id'), version=int(flag('--runtime-plan-operator-spec-version')),
                           source_sha256=flag('--runtime-plan-operator-spec-sha256')))


def _lowered(text):
    require(type(text) is str and len(text.encode()) <= 16*1024**2, 'Original IR byte limit')
    rows = text.splitlines(); starts = [i for i,row in enumerate(rows) if re.match(r'\s*func\.func @_hecate_golden\(',row)]
    require(len(starts) == 1, 'One original lowered golden entry')
    index = starts[0]; header = rows[index].strip()
    require(header.endswith(' {'), 'Canonical original lowered header')
    header = header[:-2]
    match = re.fullmatch(r'func\.func @_hecate_golden\((.*)\) -> (.*?)(?: attributes (\{.*\}))?',header)
    require(match is not None, 'Original lowered argument/result signature')
    parameters = []
    for item in split_top(match[1]):
        name, sep, typ = item.partition(':'); require(sep and re.fullmatch(SSA,name.strip()), 'Original SSA argument')
        parameters.append((name.strip(),poly(typ.strip())))
    require(1 <= len(parameters) <= 17 and len({n for n,t in parameters}) == len(parameters), 'Original parameter count')
    results = match[2].strip(); results = results[1:-1] if results.startswith('(') and results.endswith(')') else results
    result_types = [poly(t) for t in split_top(results)]; require(1 <= len(result_types) <= 16, 'Original result count')
    require('dist.device_counts' not in (match[3] or ''), 'Placed/distributed IR needs its own audited place adapter')
    ops = []; returns = None
    for line in rows[index+1:]:
        line = line.strip()
        if not line: continue
        if line == '}': break
        if line.startswith('return '):
            require(returns is None, 'Unique lowered terminator')
            parts = line[7:].split(' : ',1); require(len(parts) == 2, 'Typed lowered return')
            names = split_top(parts[0]); require(all(re.fullmatch(SSA,n) for n in names), 'Explicit return operands')
            require([poly(t) for t in split_top(parts[1])] == result_types, 'Return/header type order')
            returns = names; continue
        require(returns is None, 'No operation after final return')
        found = re.fullmatch(r'(?:('+SSA+r')\s*=\s*)?"((?:ckks|dist)\.[a-z_]+)"\(([^)]*)\)\s*(.*?)\s*:\s*\((.*?)\)\s*->\s*(.*)',line)
        require(found is not None, 'Audited single-block generic CKKS/Dist operation')
        name, operation, args, attrs, operand_types, result = found.groups()
        args = split_top(args); require(all(re.fullmatch(SSA,n) for n in args), 'Explicit SSA operands')
        types = [poly(t) for t in split_top(operand_types)]; require(len(args) == len(types), 'Operand type count')
        result_type = None if result.strip() == '()' else poly(result.strip())
        require((name is None) == (result_type is None), 'One exported result or explicit no-result operation')
        ops.append(dict(name=name, operation=operation, inputs=args, input_types=types, attrs=attributes(attrs), type=result_type))
    require(returns is not None and len(ops) <= MAX_OPERATIONS, 'Complete bounded lowered entry')
    return parameters, result_types, ops, returns


def audit_dag(lowered, raw, command, payload_reader):
    """Compare every original SSA value and instruction, not just I/O counts."""
    ctx = command_context(command)
    require(raw.get('format_version') == 1 and raw.get('plan_id') == ctx['plan_id'] and raw.get('finalization') == [], 'Original RuntimePlan header')
    target = raw['target']
    require(target == dict(target_id=ctx['target_id'],capability_version=ctx['capability_version'],operator_spec=ctx['operator_spec'],
                          device_counts=[0],world_size=1), 'Original single-host CPU target')
    parameters, result_types, ops, returns = _lowered(lowered)
    ids = {}; types = {}; values = {}; implicit = 0; next_id = 0
    def register(name, ident, typ):
        nonlocal next_id
        require(name not in ids and str(ident) not in values and 0 <= ident < 2**64-1, 'Unique exporter ValueId/SSA definition')
        ids[name] = str(ident); types[name] = typ
        values[str(ident)] = dict(typ,id=str(ident),context=ctx['context'],ntt=ctx['ntt'],place=dict(kind='host',rank=0))
        next_id = max(next_id,ident+1)
    for name,typ in parameters: register(name,implicit,typ); implicit += 1
    for op in ops:
        attrs = op['attrs']; stable = attrs.get('runtime.value_id'); logical = attrs.get('dist.logical_id')
        require(not any(k in attrs for k in ('dist.rank','dist.device')), 'Placed values need separate audited place adapter')
        if op['operation'].startswith('ckks.'):
            require(op['name'] is not None, 'Original CKKS result required')
            if stable is not None and logical is not None: require(integer(stable) == integer(logical), 'Consistent stable/logical ValueId')
            if stable is not None: ident = integer(stable)
            elif logical is not None: ident = integer(logical)
            else: ident = implicit; implicit += 1
            register(op['name'],ident,op['type'])
    for op in ops:
        if op['operation'] == 'dist.transfer':
            register(op['name'],integer(op['attrs']['runtime.value_id']) if 'runtime.value_id' in op['attrs'] else next_id,op['type'])
    require(len(values) <= MAX_VALUES and raw['values'] == [values[k] for k in sorted(values,key=int)], 'All original typed ValueIds')
    require(raw['external_inputs'] == [ids[n] for n,t in parameters], 'Original parameter-to-ValueId order')
    defined = {n for n,t in parameters}; instructions = []
    for op in ops:
        attrs = dict(op['attrs']); common = {'runtime.value_id','dist.logical_id'}
        require(all(n in defined and types[n] == t for n,t in zip(op['inputs'],op['input_types'])), 'Exact SSA operand/type association')
        name = op['operation']; operands = [ids[n] for n in op['inputs']]; dst = ids.get(op['name'])
        if name == 'ckks.encode':
            require(not operands and set(attrs) <= common|{'payload','runtime.streaming'} and 'payload' in attrs, 'Original Encode properties')
            data = dense(attrs['payload']); inst = dict(kind='encode',output=dst,_payload=data)
            initial = 'runtime.streaming' not in attrs
        elif name.startswith('ckks.'):
            suffix = name[5:]; require(suffix in OPS, 'Supported original exporter operation')
            expected_count = 2 if suffix in ('addcc','addcp','mulcc','mulcp') else 1
            require(len(operands) == expected_count, 'Original compute arity')
            allowed = common|{'runtime.reuse_input'}
            inst = dict(kind='compute',op=OPS[suffix],inputs=operands,output=dst,place=dict(kind='host',rank=0)); initial = False
            if 'runtime.reuse_input' in attrs:
                require(integer(attrs['runtime.reuse_input']) == 0, 'Original reuse-input convention'); inst['reuse_input'] = 0
            if suffix == 'rotatec':
                allowed.add('offset'); m = re.fullmatch(r'array<i64:\s*(-?[0-9]+)>',attrs['offset'])
                require(m is not None and 0 < abs(int(m[1])) < 2**31, 'Original one-step rotation'); inst['attrs'] = dict(steps=int(m[1]))
            elif suffix in ('rescalec','modswitchc','bootstrapc'):
                inst['attrs'] = dict(target_level=op['type']['level'])
                if suffix == 'modswitchc':
                    allowed.add('downFactor'); require(types[op['inputs'][0]]['level']-integer(attrs['downFactor']) == op['type']['level'], 'Actual modswitch level')
                if suffix in ('rescalec','bootstrapc'): inst['attrs']['target_scale_log2'] = op['type']['scale_log2']
                if suffix == 'bootstrapc':
                    require(ctx['boot_profile'], 'Original explicit boot profile required')
                    inst['attrs'].update(target_components=op['type']['components'],operator_profile=ctx['boot_profile'],implementation=ctx['boot_implementation'])
            require(set(attrs) <= allowed, 'No unaudited compute attributes')
        elif name == 'dist.fence':
            require(not attrs and not operands and dst is None, 'Original fence'); inst = dict(kind='fence'); initial = False
        elif name == 'dist.release':
            require(set(attrs) == {'initialization'} and len(operands) == 1 and dst is None, 'Original release')
            inst = dict(kind='release',value=operands[0]); initial = boolean(attrs['initialization'])
        elif name == 'dist.transfer':
            require(set(attrs) <= common|{'transfer_id','source_rank','source_device','destination_rank','destination_device','initialization'} and len(operands) == 1,
                    'Original transfer properties')
            require(all(integer(attrs[k]) == v for k,v in [('source_rank',0),('source_device',-1),('destination_rank',0),('destination_device',-1)]), 'CPU host-only transfer')
            inst = dict(kind='transfer',transfer_id=str(integer(attrs['transfer_id'])),hint='point_to_point',inputs=operands,outputs=[dst],
                sources=[dict(kind='host',rank=0)],destinations=[dict(kind='host',rank=0)],output_kinds=[op['type']['kind']]); initial = boolean(attrs['initialization'])
        else: raise ValueError('Unknown original exporter operation')
        instructions.append((initial,inst))
        if op['name'] is not None: defined.add(op['name'])
    scheduled = [inst for phase in (True,False) for initial,inst in instructions if initial is phase]
    require(len(raw['initialization']) == sum(initial for initial,inst in instructions)
            and len(raw['execution']) == sum(not initial for initial,inst in instructions), 'Original initialization/online phase boundary')
    actual = [*raw['initialization'],*raw['execution']]
    require(len(scheduled) == len(actual), 'All original instructions retained')
    payloads = []
    for ordinal,(expected,recorded) in enumerate(zip(scheduled,actual)):
        expected = dict(expected,ordinal=ordinal)
        if '_payload' in expected:
            data = expected.pop('_payload'); require(payload_reader(raw,recorded,'raw') == data, 'Original IR/RuntimePlan public payload bits')
            expected['payload'] = recorded['payload']; payloads.append(dict(value_id=expected['output'],bytes=len(data),sha256=sha(data)))
        require(expected == recorded, 'Every original scheduled instruction/operand/attribute')
    require(all(n in defined and types[n] == t for n,t in zip(returns,result_types)) and len(returns) == len(result_types), 'Complete ordered final SSA types')
    require(raw['final_outputs'] == [ids[n] for n in returns], 'Actual return-to-runtime ValueId order')
    return bound(dict(format=FORMAT,source_kind='typed_CKKS_to_RuntimePlan_data',lowered_sha256=sha(lowered.encode()),
        raw_plan_sha256=sha(canonical(raw)),parameters=[dict(ssa=n,value_id=ids[n],type=t)for n,t in parameters],
        returns=[dict(ssa=n,value_id=ids[n],type=t)for n,t in zip(returns,result_types)],value_count=len(values),
        instruction_count=len(scheduled),encode_payloads=payloads,source_model_semantics_or_native_execution_certified=False))


def audit_periodic_binding(raw, packed, period, payload_reader):
    require(type(period) is int and 1 <= period <= 16384 and 16384 % period == 0, 'Original periodic layout')
    clean = lambda p:{k:v for k,v in p.items() if k not in ('initialization','execution','plaintext_bundle')}
    require(clean(raw) == clean(packed), 'Periodic binding must preserve all original plan/ValueId/I/O/target metadata')
    payloads = []
    for phase in ('initialization','execution'):
        require(len(raw[phase]) == len(packed[phase]), 'Periodic instruction count')
        for before,after in zip(raw[phase],packed[phase]):
            if before['kind'] != 'encode': require(before == after, 'No arithmetic or ordering changes in constant binding'); continue
            require({k:v for k,v in before.items()if k!='payload'} == {k:v for k,v in after.items()if k!='payload'}, 'Only periodic Encode payload may change')
            original = payload_reader(raw,before,'raw'); count = len(original)//8
            require(len(original) % 8 == 0 and count > 0, 'Typed original payload')
            if count <= period and period % count == 0: expected = original*(16384//count)
            else:
                require(count == 16384 and all(original[i*8:(i+1)*8] == original[(i%period)*8:(i%period+1)*8]for i in range(count)), 'Original full payload already periodic')
                expected = original
            observed = payload_reader(packed,after,'bound'); require(observed == expected, 'Bit-preserving original periodic payload materialization')
            payloads.append(dict(value_id=before['output'],original_sha256=sha(original),bound_sha256=sha(observed),original_values=count,bound_values=16384))
    return bound(dict(format='poseidon-original-periodic-payload-correspondence-v1',input_slot_period=period,encodes=payloads,
        all_non_payload_fields_unchanged=True,plaintext_values_changed=False,native_execution_certified=False))


def layout_mapping(request, source, selected, dag):
    from component_contract import reconstruct_request
    from candidate_contract import request_input_names
    from benchmark_graph import validate
    from unified_output_aliases import policy as alias_policy
    checked_bound(dag)
    reconstruct_request(copy.deepcopy(request)); checked = validate(request['model']); layout = request['layout']
    tree = ast.parse(source); functions = [n for n in tree.body if type(n) is ast.FunctionDef and n.name == 'golden']
    require(len(functions) == 1, 'One exact original DSL entry')
    args = functions[0].args; names = list(request_input_names(request))
    require(not args.posonlyargs and not args.kwonlyargs and not args.defaults and not args.kw_defaults
            and args.vararg is args.kwarg is None and [a.arg for a in args.args] == names, 'Original DSL parameter order')
    lines = selected.splitlines(); starts = [i for i,row in enumerate(lines) if '"func.func"()' in row and re.search(r'sym_name = "_hecate_golden"(?:,|\s*\}>)',row)]
    require(len(starts) == 1, 'Exact selected frontend golden entry')
    body = []
    for row in lines[starts[0]+1:]:
        if re.match(r'^\s*\}\) : \(\) -> \(\)',row): break
        body.append(row)
    blocks = [r for r in body if '^bb' in r]; require(len(blocks) == 1, 'Single original frontend entry block')
    found = re.findall(r'%arg([0-9]+):',blocks[0]); require(found == [str(i)for i in range(len(names))], 'Frontend argument positions match original DSL/layout')
    returns = [r for r in body if '"func.return"(' in r]; require(len(returns) == 1, 'Original frontend terminator')
    m = re.search(r'"func.return"\(([^)]*)\)',returns[0]); require(m is not None, 'Original frontend return')
    require(len(split_top(m[1])) == len(dag['returns']) == layout['output_ciphertexts'], 'Frontend/lowered/physical result count')
    require(len(names) == len(dag['parameters']), 'Full lowered parameter count')
    require(len({r['value_id']for r in dag['returns']}) == len(dag['returns']), 'Distinct physical result ValueIds; logical aliases stay in the layout')
    require(all(r['type']['kind'] == 'ciphertext' for r in dag['parameters']+dag['returns']), 'All physical input/output roles are ciphertexts')
    require(len(re.findall(r'%arg[0-9]+:\s*tensor<1x!earth\.ci<',blocks[0])) == len(names), 'Original encrypted frontend argument types')
    period = layout['input_slot_period']; physical = list(layout['inputs'])+list(layout.get('auxiliary_ciphertexts',[]))
    require([x['dsl_name']for x in physical] == names, 'All physical input roles including zero')
    inputs = [dict(abi_index=i,ssa=dag['parameters'][i]['ssa'],value_id=dag['parameters'][i]['value_id'],role=copy.deepcopy(row))for i,row in enumerate(physical)]
    outputs = []; selectors = []; slots = {}; model = request['model']; require(len(model['outputs']) == len(layout['outputs']), 'All named model outputs')
    owners = {}; is_alias = alias_policy(layout) is not None
    for model_output,declared in zip(model['outputs'],layout['outputs']):
        name, value = model_output['name'],model_output['value']; shape = list(checked['shapes'][value]); size = math.prod(shape)
        require(declared['name'] == name and declared['shape'] == shape, 'Original logical output identity/shape/order')
        chunks = declared.get('chunks',[dict(ciphertext=declared.get('ciphertext'),offset=0,elements=size)])
        cursor = 0; segments = []
        for chunk in chunks:
            ct,offset,count = (chunk[k]for k in ('ciphertext','offset','elements'))
            require(type(ct)is int and 0 <= ct < len(dag['returns']) and offset == cursor and type(count)is int and 1 <= count <= period, 'All output chunks, including tails')
            runtime = dag['returns'][ct]['value_id']; segments.append(dict(ciphertext=ct,value_id=runtime,logical_start=offset,slot_start=0,elements=count))
            selectors.extend([[ct,i]for i in range(count)]); cursor += count
            for slot in range(count):
                key = (ct,slot); old = slots.get(key)
                require(old is None or is_alias and old == value and declared.get('alias_of') == owners.get(value), 'Only exact model-value aliases may share physical cells')
                slots[key] = value
        require(cursor == size, 'No missing logical output cells')
        owners.setdefault(value,name)
        outputs.append(dict(name=name,model_value=value,shape=shape,alias_of=declared.get('alias_of'),segments=segments))
    require(selectors == layout['output_selectors'] and set(ct for ct,slot in slots) == set(range(len(dag['returns']))), 'Complete original output selector map')
    return bound(dict(format='poseidon-generic-compiler-logical-I-O-map-v1',request_id=request['request_id'],source_sha256=sha(source.encode()),
        selected_IR_sha256=sha(selected.encode()),compiler_DAG_binding=dag['binding'],inputs=inputs,outputs=outputs,
        output_selectors_sha256=sha(canonical(selectors)),logical_values=len(selectors),unique_physical_cells=len(slots),alias_cells=len(selectors)-len(slots),
        actual_raw_to_encrypted_input_or_complex_decoder_or_numeric_execution_certified=False))
