"""Own R31 source variant: prepare public keys/plaintexts before inference.

This module only generates source and estimates public metadata storage. It
never builds or runs a client. The original runtime and native library stay
unchanged; the generated source retains their arithmetic and CKKS comparison.
"""
from collections import Counter
import math
from pathlib import Path


def storage_inventory(plan, residual, bootstrap, *, dft_bytes=0):
    values = {row['id']: row for row in plan['values']}
    unique = {}
    for row in plan['initialization']:
        value = values[row['output']]
        if row['kind'] != 'encode' or value['kind'] != 'plaintext':
            raise ValueError('Use the original explicit plaintext initialization')
        key = (value['level'], value['scale_log2'], row['payload']['content'])
        unique[key] = value
    plain = sum(residual['n'] * (row['level'] + 1) * 8 for row in unique.values())
    key_bytes = (residual['galois_count'] * residual['one_key_bytes']
                 + bootstrap['galois_count'] * bootstrap['one_key_bytes'])
    # Native serialization adds small metadata. Reserve 1 MiB per object,
    # plus 1 GiB for the remaining key/context files and process evidence.
    objects = len(unique) + residual['galois_count'] + bootstrap['galois_count']
    projected = plain + key_bytes + objects * 1024**2 + 1024**3
    return dict(unique_encoded_plaintexts=len(unique), plaintext_coefficient_bytes=plain,
                galois_coefficient_bytes=key_bytes,
                resident_payload_with_dft_bytes=plain + key_bytes + dft_bytes,
                disk_projection_bytes=projected,
                level_counts=dict(Counter(row['level'] for row in unique.values())),
                excludes='Native scratch/allocator; disk projection is a conservative preparation admission estimate')


def disk_admission(free_bytes, projection, *, reserve_bytes=10 * 1024**3):
    if any(type(x) is not int or x < 0 for x in (free_bytes, projection, reserve_bytes)):
        raise ValueError('Nonnegative integer byte allowances required')
    return dict(accepted=free_bytes >= projection + reserve_bytes,
                free_bytes=free_bytes, projected_new_bytes=projection,
                required_remaining_bytes=reserve_bytes)


def check_client(client, plan, expected):
    """Validate the quiet disk-backed run without requiring old per-op logs."""
    counts = dict(Counter(op['op'] for op in plan['execution']))
    boots = counts.get('boot', 0)
    if (client.get('status') != 'ok' or client.get('encrypted_plan_completed') is not True
            or client.get('ckks_inference_completed') is not True
            or client.get('no_keygen_or_constant_encoding_during_inference') is not True
            or client.get('per_operation_recording_enabled') is not False
            or client.get('input_ssa_check_enabled') is not False
            or client.get('public_key_request_trace_enabled') is not False
            or client.get('input_ssa_observations') or client.get('bootstrap_observations')
            or client.get('intermediate_decryptions') != 0
            or any(client.get(key) != len(plan['execution']) for key in ('compute_attempts', 'compute_completed'))
            or any(client.get(key) != counts for key in ('op_attempts_by_kind', 'op_completed_by_kind'))
            or any(client.get(key) != boots for key in ('bootstrap_attempts', 'bootstrap_completed',
                       'native_bootstrap_api_attempts', 'native_bootstrap_api_completed'))
            or any(client.get(key) != 1 for key in ('encryption_attempts', 'encryption_completed',
                       'final_decryption_attempts', 'final_decryption_completed'))):
        raise ValueError('Prepared run counts, timing or logging contract differs')
    duration = client.get('ckks_inference_s')
    if (type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
            or not math.isclose(duration * 1e9, client.get('ckks_inference_observed_ns', -1), abs_tol=1)):
        raise ValueError('Inference clock is missing or inconsistent')
    got, want = client.get('final_selected_values'), expected.get('values')
    if (not isinstance(got, list) or not isinstance(want, list) or not got or len(got) != len(want)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in got + want)):
        raise ValueError('Finite aligned original-unit output logits required')
    errors = [abs(a - b) / (1e-5 + 1e-4 * abs(b)) for a, b in zip(got, want)]
    comparison = client.get('comparison') or {}
    if (max(errors) > 1 or comparison.get('passed') is not True
            or comparison.get('absolute_tolerance') != 1e-5 or comparison.get('relative_tolerance') != 1e-4):
        raise ValueError('Original CKKS output comparison failed')
    return dict(passed=True, bootstrap_count=boots, execution_count=len(plan['execution']),
                max_error_ratio=max(errors), ckks_inference_s=duration,
                inference_scope=client.get('prepared_inference_scope'))


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Original source anchor differs: ' + old[:90])
    return source.replace(old, new, 1)


def _function(source, first, following, replacement):
    begin = source.index(first)
    end = source.index(following, begin)
    return source[:begin] + replacement + '\n\n' + source[end:]


def build_source(base=None, extension=None):
    directory = Path(__file__).parent
    source = (directory / 'r31_lattigo_runtime_plan.go').read_text() if base is None else base
    extra = (directory / 'a4_prepared_disk_client.go').read_text() if extension is None else extension
    source = _once(source, '\t"encoding/binary"', '\t"io"\n\t"syscall"\n\t"encoding/binary"')
    source = _once(source, 'func save(force ...bool) {',
                   'func save(force ...bool) {\n\tif preparedTimed { return }')
    source = _once(source, 'func emit(row map[string]any) {',
                   'func emit(row map[string]any) {\n\tif preparedTimed { return }')
    source = _once(source, 'func stage(name string, metadata map[string]any, fn func() error) (err error) {',
                   'func stage(name string, metadata map[string]any, fn func() error) (err error) {\n'
                   '\tif preparedSilentStage(name) {\n'
                   '\t\tdefer func(){if r:=recover();r!=nil {err=fmt.Errorf("panic: %v",r)}}()\n'
                   '\t\treturn fn()\n\t}')
    source = _function(source, 'func keygen(name string, fn func()) error {', 'func readJSON(',
        '''func keygen(name string, fn func()) error {
    if preparedTimed { return fmt.Errorf("key generation forbidden during prepared inference") }
    count("keygen_api_attempts")
    return stage(name,nil,func() error {
        began:=time.Now();fn();preparedKeygenNS+=time.Since(began).Nanoseconds()
        count("keygen_api_completed");return nil
    })
}''')
    source = _once(source, '\tvar key *rlwe.GaloisKey\n\terr := keygen(l.name+".galois", func() { key = l.kg.GenGaloisKeyNew(g, l.sk) })',
                   '\tkey, err := preparedReadKey(l, g)')
    source = _once(source, '\tl.generations++', '\tpreparedBump("prepared_key_loads")')
    source = _once(source, '"serialized_keys": 0}', '"serialized_keys": len(l.list)}')
    source = _once(source,
        '\tlazy = newLazy("bootstrap_lazy", bigp, extended, bp.GaloisElements(bigp), bootstrapCacheCapacity)\n',
        '\tlazy = newLazy("bootstrap_lazy", bigp, extended, bp.GaloisElements(bigp), bootstrapCacheCapacity)\n'
        '\tif err = preparedWriteKeys(lazy); err != nil { return }\n')
    source = _once(source, '\tlazy := newLazy("residual_lazy", p, sk, gals)\n',
        '\tlazy := newLazy("residual_lazy", p, sk, gals)\n'
        '\tif err := preparedWriteKeys(lazy); err != nil { return err }\n')
    source = _function(source, 'func (c *plaintextCache) get(id string)',
        'func (c *plaintextCache) summary()', '''func (c *plaintextCache) get(id string) (*rlwe.Plaintext,error) {
    return preparedReadPlaintext(c,id)
}''')
    source = _once(source, '"peak_coefficient_bytes": c.peak, "raw_vectors_cached": 0}',
                   '"peak_coefficient_bytes": c.peak, "raw_vectors_cached": 0, "read_only_preencoded_store": true}')
    marker = 'func execute(plan Plan, values map[string]Value, encodes map[string]string, last map[string]int, bundle, input, expected string, p ckks.Parameters, bp bootstrapping.Parameters, allowBoot bool) error {'
    source = _once(source, marker, marker + '\n\tdefer func(){if preparedTimed {preparedStopInference(false)}}()')
    source = _once(source,
        '\tcache := &plaintextCache{p: p, encoder: encoder, bundle: bundle, values: values, encodes: encodes}\n',
        '\tkg = nil // All key generators are closed before inference.\n'
        '\tcache := &plaintextCache{p: p, encoder: encoder, bundle: bundle, values: values, encodes: encodes}\n'
        '\tif err := preparedWritePlaintexts(cache, plan.Initialization); err != nil { return err }\n')
    source = _once(source, '\t\tct, err := rlwe.NewEncryptor(p, pk).EncryptNew(pt)',
        '\t\tif err := preparedStartInference(); err != nil { return err }\n'
        '\t\tct, err := rlwe.NewEncryptor(p, pk).EncryptNew(pt)')
    begin = source.index('\t\t\t\tbeforeCache := bootLazy.summary()')
    end = source.index('\n\t\t\t\tif err == nil {\n\t\t\t\t\tcount("bootstrap_completed")', begin)
    source = source[:begin] + '''                count("native_bootstrap_api_attempts")
                out, err = bootstrapper.Bootstrap(bootstrapInput)
                if err == nil {count("native_bootstrap_api_completed")}
                if a.Level()!=entryLevel || a.Scale.Log2()!=entryScale {
                    return fmt.Errorf("native bootstrap mutated its original SSA input metadata")
                }
                // Preserve existing arithmetic bookkeeping without per-bootstrap records.
                _ = internalInputScale
                row:=map[string]any{}
''' + source[end:]
    source = _once(source, '\t\tgot := make([]float64, len(reference.Indices))',
        '\t\tif err := preparedStopInference(true); err != nil {return err}\n'
        '\t\tgot := make([]float64, len(reference.Indices))')
    source = _once(source, '\tflag.Parse()',
        '\tpreparedDirFlag:=flag.String("prepared-dir","","fresh disk store for prepared public keys/plaintexts")\n'
        '\tflag.Parse()\n\tpreparedRoot=*preparedDirFlag')
    source = _once(source, '\tplan, values, encodes, _, last, err := loadPlan(*planPath, *bundle)',
        '\tif *mode == "execute" {\n'
        '\t\tif checkInputSSA || traceKeyRequests {panic("per-operation observations must be disabled")}\n'
        '\t\tif preparedRoot=="" {panic("fresh prepared-dir is required")}\n'
        '\t\tif err:=os.Mkdir(preparedRoot,0700);err!=nil {panic(err)}\n'
        '\t\tif err:=preparedDiskAllowance();err!=nil {panic(err)}\n'
        '\t}\n'
        '\tplan, values, encodes, _, last, err := loadPlan(*planPath, *bundle)')
    return source + '\n\n' + extra
