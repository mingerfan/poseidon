"""Read complete public Native lifecycle evidence; never run a candidate or backend.

All counts come from distinct original shared operations and retained final
decodes. The small mathematical evaluator is an independent reference for the
explicit Native graph domain, not a ciphertext executor or fallback.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile
from benchmark_graph import canonical,digest,require,validate
import native_boot_plan_link_v3 as linker

FORMAT='poseidon-native-Agent-program-acceptance-v2'
LIMITS={'Max address space':[4294967296,4294967296],'Max core file size':[0,0],
        'Max cpu time':[150,155],'Max file size':[16777216,16777216],'Max open files':[128,128]}


def direct(path,work,*,directory=False):
    path=Path(path);work=Path(work)
    require(path.is_absolute()and path.resolve().is_relative_to(work.resolve())and
            '.env'not in path.parts and 'client-private'not in path.parts and
            not any(p.is_symlink()for p in (path,*path.parents)), 'Public direct Native release evidence')
    require(path.is_dir()if directory else path.is_file()and path.stat().st_size<=16*1024**2,
            'Bounded Native release evidence')
    return path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path,work):
    path=direct(path,work)
    require(path.suffix=='.json','Native release JSON evidence')
    def pairs(items):
        value={}
        for k,v in items:require(k not in value,'Duplicate release JSON key');value[k]=v
        return value
    return json.loads(path.read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda _ :require(False,'Nonfinite release JSON'))


def record(value,work,*,bound=False):
    require(type(value)is dict and set(value)==({'path','sha256','binding'}if bound else{'path','sha256'}),
            'Exact Native release evidence reference')
    p=direct(value['path'],work);require(p.suffix=='.json'and sha(p)==value['sha256'],'Native release referenced bytes')
    document=read(p,work)
    if bound:require(document.get('binding')==value['binding']==digest({k:v for k,v in document.items()if k!='binding'}),
                     'Native release referenced binding')
    return p,document


def npy(raw):
    require(type(raw)is bytes and len(raw)<=4*1024**2 and raw[:6]==b'\x93NUMPY','Bounded numeric array')
    major,minor=raw[6:8];require(major in(1,2)and minor==0,'Numeric array version')
    offset=10 if major==1 else 12
    length=struct.unpack_from('<H'if major==1 else'<I',raw,8)[0]
    require(length<=65536 and offset+length<=len(raw),'Numeric array header')
    header=ast.literal_eval(raw[offset:offset+length].decode('latin1'))
    require(type(header)is dict and set(header)=={'descr','fortran_order','shape'}and
            header['descr']=='<f8'and header['fortran_order']is False and
            type(header['shape'])is tuple and all(type(n)is int and 0<n<=16384 for n in header['shape']),
            'Only finite C-order float64 reference arrays')
    count=math.prod(header['shape']);require(len(raw)==offset+length+8*count,'Exact numeric array bytes')
    values=list(struct.unpack_from('<'+str(count)+'d',raw,offset+length))
    require(all(math.isfinite(v)for v in values),'Finite reference array')
    return header['shape'],values


def probes(model):
    count=math.prod(model['inputs'][0]['shape']);seed=233;random=[]
    for _ in range(count):seed=(1664525*seed+1013904223)&0xffffffff;random.append(float(seed>>8)/16777215-.5)
    signed=[.125,-.25,.375,-.5];boundary=[-.5,.5,-.499,.499,0.,1/1024,-1/1024,.25]
    return [[0.]*count,[signed[i%4]for i in range(count)],random,[boundary[i%8]for i in range(count)]]


def flatten(value,shape):
    if not shape:
        require(type(value)in(int,float)and math.isfinite(value)and abs(value)<=.5,'Native supplied input domain')
        return [float(value)]
    require(type(value)is list and len(value)==shape[0],'Native supplied input shape')
    return [x for child in value for x in flatten(child,shape[1:])]


def reference(model,sample):
    shapes=validate(model)['shapes'];values={model['inputs'][0]['name']:list(sample)}
    for node in model['nodes']:
        args=[values[name]for name in node['inputs']];op=node['op'];outshape=shapes[node['outputs'][0]]
        if op in('reshape','flatten'):result=list(args[0])
        elif op in('square','power'):
            exponent=2 if op=='square'else node['attrs']['exponent'];result=[x**exponent for x in args[0]]
        elif op=='multiply':
            def broadcast(name,data,index):
                source=shapes[name];coordinates=[]
                for dim in reversed(outshape):coordinates.append(index%dim);index//=dim
                coordinates=list(reversed(coordinates))[-len(source):];flat=0
                for n,c in zip(source,coordinates):flat=flat*n+(0 if n==1 else c)
                return data[flat]
            result=[broadcast(node['inputs'][0],args[0],i)*broadcast(node['inputs'][1],args[1],i)
                    for i in range(math.prod(outshape))]
        else:raise ValueError('Uncertified Native mathematical reference operation')
        for name in node['outputs']:values[name]=result
    return values[model['outputs'][0]['value']]


def current_identities(sdk,backend):
    """Rebuild current source stages without recursively loading this SDK."""
    from native_sdk_manifest import execution_fingerprint,EPOCH
    expected=dict(source_lock_sha256=sdk['source_lock_sha256'],
        runtime_dacapo_tree_sha256=sdk['source_trees']['runtime'],api_tree_sha256=sdk['source_trees']['api'],
        runner_tree_sha256=sdk['source_trees']['runner'],compiler_sha256=sdk['artifacts']['compiler'],
        frontend_sha256=sdk['artifacts']['frontend'],poseidon_library_sha256=sdk['artifacts']['poseidon_library'],
        executor_sha256=sdk['native']['runtime']['sha256'],native_runtime_sha256=sdk['native']['runtime']['sha256'],
        native_client_sha256=sdk['native']['client']['sha256'],native_key_epoch=EPOCH,
        native_capability_binding=sdk['native']['capability']['binding'],
        native_sdk_execution_fingerprint=execution_fingerprint(sdk),
        execution_profile='original-epoch-public-native-v1',inherited_base_executor_sha256=sdk['artifacts']['executor'],
        compiler_profile_sha256=sdk['native']['compiler_profile']['sha256'],
        operator_spec_sha256=sdk['native']['operator_spec']['sha256'])
    require(all(backend.get(k)==v for k,v in expected.items()),'Exact SDK Native backend resource projection')
    require(backend['build_configuration']['environment_files']==sdk['environment_files']and
        backend['build_configuration']['cmake_cache_sha256']==sdk['cmake_caches']and
        backend['platform']['configured_platform']==sdk['platform'],'Exact SDK build/environment/platform projection')
    module=Path(sdk['native']['modules']['native_sdk_manifest.py']['path'])
    root=module.parents[4]if module.parents[2].name=='DSL_agent'and module.parents[3].name=='DSL'else module.parents[2]
    from application_identity import identity_manifest
    return identity_manifest(root,backend)


def ledger(work):
    """Verify the original live coordinator's own digest/policy, without a byte scan."""
    from shared_execution import digest as shared_digest
    base=Path(work).parent.parent;folder=base/'coordination/v1'
    value=read(folder/'ledger.json',base);policy=read(folder/'policy.json',base)
    require(policy.get('binding')==shared_digest({k:v for k,v in policy.items()if k!='binding'}),
        'Original coordinator policy binding')
    require(value.get('binding')==shared_digest({k:v for k,v in value.items()if k!='binding'})and
        value['policy']=={k:v for k,v in policy.items()if k!='binding'},'Original coordinator ledger binding/policy')
    return value


def operation(ref,work,package,kind,used_operations):
    path,op=record(ref,work,bound=True);name=op['batch_name']
    require(name not in used_operations,'Distinct original Native lifecycle operations')
    used_operations.add(name)
    require(op['format']=='poseidon-application-shared-operation-v1'and
            op['action']==('application-execute'if kind=='execute'else'application-run'),
            'Original Native application operation kind')
    store=direct(op['task_store'],work,directory=True);task=direct(store/op['task_id'],work,directory=True)
    require(path==task/(name+'.json')and store==package.parent.parent,'Original Native operation task scope')
    scope=direct(op['scope'],work,directory=True)
    require(scope.parent==Path(work)/'results'and scope.name==name,'Original shared operation scope')
    data=ledger(work)
    row=data['batches'][name];plan=read(scope/'plan.json',work);result=read(scope/'report.json',work)
    require(row['status']=='succeeded'and row['finished']is not None and row['active']is None and
            row['owner']==op['owner']and row['plan_binding']==op['plan_binding']==plan['plan_binding']and
            digest(plan['plan'])==op['plan_binding']and row['plan']==plan['plan']and
            plan['plan']['source_binding']==digest(plan['authority'])and
            result['plan_binding']==op['plan_binding']and result['batch_name']==name and
            result['input_sha256']==plan['authority']['input_sha256']and result['status']=='succeeded'and
            result['stage_usage']==row['used'],'Completed original shared operation; interrupted is not release acceptance')
    require(plan['authority']['action']==op['action']and
            plan['authority']['input_sha256']==digest(op['operation_input']),'Original operation input binding')
    meta=read(task/'task.json',work);spec=op['operation_input']['metadata']['task.json']
    require(sha(task/'task.json')==spec['sha256']and (task/'task.json').stat().st_size==spec['bytes']and
            meta['binding']==digest({k:v for k,v in meta.items()if k!='binding'}),'Original task metadata bytes')
    require(meta['request']==read(package/'request.json',work)and meta['validation_level']=='numerical',
            'Native original operation request')
    state=read(task/'state.json',work)
    require(state['status']=='succeeded'and state['package']['task_id']==package.parent.name and
            state['package']['binding']==read(package/'manifest.json',work)['binding'],'Original program delivery')
    require(plan['authority']['host']['backend_identity']==read(package/'manifest.json',work)['dependencies']['backend_identity'],
            'Exact original operation backend/source identity')
    expected=kind=='execute'
    require(row['used']['encrypted']==int(expected)and row['used']['build']==row['used']['compile']==row['used']['link']==0,
            'Measured original Native operation stages')
    if kind=='qualification':require(row['used']['qualify']==1 and row['used']['api']>=1,'Real original Agent qualification operation')
    else:require(row['used']['qualify']==row['used']['api']==0,'No new Agent/compiler in reuse operation')
    if kind=='cache':require(state['package']['reused']is True,'Actual original cache delivery')
    return op,row


def registered(root,row):
    st=Path(root).stat()
    require(any(v['path']==str(root)and v['device']==st.st_dev and v['inode']==st.st_ino for v in row['artifacts']),
            'Original shared artifact ownership')


def native_outputs(root,execution,q,sdk,derived,link,owner,samples,work):
    root=direct(root,work,directory=True);bound=root/'runtime-input'
    require(sha(direct(bound/'source-plan.json',work))==link['source_plan_sha256']and
            sha(direct(bound/'plan.json',work))==link['derived_plan_sha256']and
            read(bound/'link-record.json',work)==link and
            sha(direct(bound/'operator-spec.json',work))==sdk['native']['operator_spec']['sha256'],
            'Original Native execution complete compiler inputs')
    link_sha=sha(bound/'link-record.json')
    require(len(execution['samples'])==len(samples)and execution['physical_native_processes']==3*len(samples)and
            execution['native_Boot_calls']==len(samples)*link['derived_Boot_calls']and
            execution['new_keys']==execution['intermediate_decryptions']==execution['reencryptions']==0 and
            execution['only_final_outputs_decrypted']is True,'Actual finite Native execution counters')
    maximum=0.;compared=0
    for index,(sample,row)in enumerate(zip(samples,execution['samples'])):
        require(row['index']==index and row['identity']['public_manifest_sha256']==sdk['native']['assets']['public_manifest_sha256']and
                row['identity']['runtime_plan_sha256']==link['derived_plan_sha256']and
                row['identity']['source_plan_sha256']==link['source_plan_sha256']and
                row['identity']['link_record_sha256']==link_sha and
                row['parameter_receipt']['request_id']==q['request_id'],'Native execution request/epoch/plan binding')
        group=direct(root/('sample-'+str(index)),work,directory=True);decoded=[]
        from candidate_contract import request_input_names
        names=request_input_names(q);period=q['layout']['input_slot_period'];inputs=[];bindings=[]
        for source_id in link['retained_source_external_input_ids']:
            position=link['original_source_external_input_ids'].index(source_id);name=names[position]
            derived_id=link['source_input_to_derived_external_ids'][source_id]
            if name==q['layout']['inputs'][0]['dsl_name']:
                values=list(sample)+[0.]*(period-len(sample));kind='model_input'
            else:
                require(name=='zero_ct','Declared retained Native input');values=[0.]*period;kind='encrypted_zero'
            inputs.append(dict(value_id=derived_id,slot_period=period,values=values))
            bindings.append(dict(source_parameter_index=position,source_value_id=source_id,
                source_parameter=name,derived_value_id=derived_id,kind=kind))
        plain=group/'plain-input/input.json'
        require(read(plain,work)==dict(format='poseidon-native-sdk-plain-input-v1',
                task_binding=row['identity']['task_binding'],inputs=inputs)and
                row['parameter_receipt']['retained_parameter_bindings']==bindings and
                row['parameter_receipt']['link_record_sha256']==link_sha,
                'Actual new input plaintext/zero/parameter association')
        require([f['value_id']for f in row['final_client']['files']]==derived['final_outputs'],'Native declared final output order')
        for f in row['final_client']['files']:
            filename='output-'+f['value_id']+'.actual-complex.f64';p=direct(group/'validation'/filename,work)
            require(f['name']==filename and f['slots']==16384 and f['bytes']==16384*16 and
                    p.is_file()and not p.is_symlink()and p.stat().st_size==16384*16 and sha(p)==f['sha256'],
                    'Actual original final complex decode bytes')
            values=list(struct.iter_unpack('<dd',p.read_bytes()))
            require(all(math.isfinite(x)and math.isfinite(y)for x,y in values),'Finite actual complex decode')
            decoded.append(values)
        expected=reference(q['model'],sample)
        require(len(expected)==len(q['layout']['output_selectors']),'Native logical output count')
        for (cipher,slot),ref in zip(q['layout']['output_selectors'],expected):
            real,imag=decoded[cipher][slot];error=math.hypot(real-ref,imag)
            require(error<=1e-5+1e-4*abs(ref),'Recomputed original Native complex numerical tolerance')
            maximum=max(maximum,error);compared+=1
        for offset,role in enumerate(('encryptor','runtime','validator'),1):
            folder=direct(root/('native-'+str(3*index+offset)+'-'+role),work,directory=True)
            proof=read(folder/'host-native-proof.json',work);outcome=read(folder/'outcome.json',work)
            invocation=read(folder/'invocation.json',work)
            expected_mounts={'client-evidence':False,'client-private':role=='validator','evaluated':role!='encryptor',
                'home/lhohy':False,'inputs':role!='validator','plain-input':role=='encryptor','public':True,
                'runtime-input':True,'validation':role=='validator'}
            require(proof['driver']==owner and proof['checked_outside_native_PID_namespace']is True and
                    proof['isolated_network_namespace_verified']is True and proof['role']==role and
                    proof['role_mount_existence_verified']==expected_mounts and proof['unchanged_native_limits']==LIMITS and
                    proof['host_private_file_reads']==0 and outcome['returncode']==0 and outcome['owner_proof']is True,
                    'Original Native host owner/isolation/resource proof')
            expected_binary=sdk['native']['runtime'if role=='runtime'else'client']['sha256']
            require(proof['actual_executable_sha256']==expected_binary==invocation['executable_sha256']and
                    invocation['role']==role and invocation['driver']==owner and invocation['private_file_reads_by_host']==0,
                    'Actual exact Native executable and original invocation')
            binding_sha=(sha(plain)if role=='encryptor'else
                         sha(direct(group/('inputs'if role=='runtime'else'evaluated')/'manifest.json',work)))
            suffix=[row['identity'][key]for key in ('public_manifest_sha256','runtime_plan_sha256',
                'operator_spec_sha256','source_plan_sha256','link_record_sha256','task_binding')]+[binding_sha]
            argv=invocation['argv'];position=max(i for i,v in enumerate(argv)if v=='/bridge');actual=argv[position+1:]
            from native_sdk_manifest import EPOCH,ORIGIN
            if role=='encryptor':expected=['encrypt',EPOCH,ORIGIN,*suffix,'public-only']
            elif role=='runtime':expected=[EPOCH,ORIGIN,*suffix]
            else:expected=['decrypt-final',EPOCH,ORIGIN,*suffix,sdk['native']['assets']['client_private_manifest_sha256']]
            require(actual==expected,'Original Native command binds plaintext/input/output hashes')
        runtime=row['runtime'];client=row['final_client'];encryptor=row['encryptor']
        require(runtime['engine_native_boot_calls']==runtime['runtime_boot_calls']==link['derived_Boot_calls']and
                runtime['runtime_compute_calls']==len(derived['execution'])and
                runtime['evaluator_decryptions']==runtime['reencryptions']==runtime['new_keys']==0 and
                runtime['actual_scale_not_reassigned']is True and encryptor['new_keys']==0 and
                encryptor['secret_key_access']is False and
                client['client_final_decryptions']==len(derived['final_outputs'])and
                client['intermediate_decryptions']==client['reencryptions']==client['new_keys']==0,
                'Actual original native operations and final-only key boundary')
    return dict(compared_values=compared,max_absolute_error=maximum,Native_Boot_calls=execution['native_Boot_calls'])


def verify_case(sdk,case,work,used_operations,*,expected_identities=None):
    require(type(case)is dict and set(case)=={'request_id','package_manifest','qualification','cache_operations','new_input_operations'},
            'Native release cases require measured evidence, not summary counters')
    package_path,claimed=record(case['package_manifest'],work,bound=True);folder=package_path.parent
    require(package_path.name=='manifest.json','Native package manifest reference')
    from native_program_package import check_bundle
    from native_sdk_manifest import execution_fingerprint
    from native_execution_profile import validate_request
    package=check_bundle(folder);q=read(folder/'request.json',work);validate_request(q)
    require(package==claimed and package['agent_generated']is True and
            package['validation']==dict(level='numerical',encrypted_execution=True,numerically_validated=True)and
            case['request_id']==q['request_id']and
            package['dependencies']['backend_identity']['native_sdk_execution_fingerprint']==execution_fingerprint(sdk),
            'Complete genuine Native package/current SDK/source identity')
    identities=read(folder/'artifacts/application-identities.json',work)
    if expected_identities is None:expected_identities=current_identities(sdk,package['dependencies']['backend_identity'])
    require(identities==expected_identities,'Exact current Native application/source stage identities')
    artifacts=folder/'artifacts';source=direct(artifacts/'source-plan.json',work).read_bytes()
    spec=direct(artifacts/'operator-spec.json',work).read_bytes()
    capability=Path(sdk['native']['capability']['path'])
    capability=direct(capability,capability.parent)
    require(capability.suffix=='.json'and sha(capability)==sdk['native']['capability']['sha256'],'Exact public Native capability source')
    derived,link=linker.link_fragment(source,capability.read_bytes(),spec)
    require((artifacts/'plan.json').read_bytes()==linker.encode(derived)and
            read(artifacts/'link-record.json',work)==link and
            sha(artifacts/'operator-spec.json')==sdk['native']['operator_spec']['sha256']and
            sha(artifacts/'compiler-profile.json')==sdk['native']['compiler_profile']['sha256'],
            'Complete original Native compiler/profile/spec/derivation')
    qualification=case['qualification'];require(type(qualification)is dict and set(qualification)=={'report','operation'},'Qualification evidence fields')
    path,report=record(qualification['report'],work);evidence=path.parent
    op,row=operation(qualification['operation'],work,folder,'qualification',used_operations);registered(evidence,row)
    require(path.name=='report.json'and report['status']=='passed'and report['validation_level']=='numerical'and
            report['backend_identity']==package['dependencies']['backend_identity']and
            report['native_sdk_execution_fingerprint']==execution_fingerprint(sdk)and
            package['origin']['report_sha256']==sha(path),'Original current Native numerical qualification')
    require(read(evidence/'request.json',work)==q and len(report['attempts'])==1,'Original Native qualification request/attempt')
    from generation_receipt import verify,FILE as GENERATION
    origin=verify(read(folder/GENERATION,work),q,(folder/'candidate.py').read_text(),
        generation_identity_sha256=identities['generation']['sha256'])
    require(row['used']['api']==origin['http_attempts'],'Exact original real Agent/shared API count')
    for name,expected in report['frozen_hashes'].items():
        require(type(name)is str and not Path(name).is_absolute()and '..'not in Path(name).parts,'Relative qualification artifact')
        require(sha(direct(evidence/name,work))==expected,'Original frozen qualification artifact')
    attempt=evidence/'attempt-00';item=read(attempt/'report.json',work)
    from native_execution_profile import command
    require(item['compile_command']==command(q['compiler_configuration']),'Exact original Native compiler command')
    require(item==report['attempts'][0]and item['status']=='passed'and item['numerically_correct']is True and
            (attempt/'candidate.py').read_bytes()==(folder/'candidate.py').read_bytes(),'Exact qualified original Agent candidate')
    for name,expected in item['artifact_hashes'].items():
        require(type(name)is str and not Path(name).is_absolute()and '..'not in Path(name).parts,'Relative compiled Native artifact')
        require(sha(direct(attempt/'output'/name,work))==expected,'All original compiled Native artifacts')
    vectors=probes(q['model']);references=[v for sample in vectors for v in reference(q['model'],sample)]
    with zipfile.ZipFile(direct(evidence/'arrays.npz',work))as archive:
        require(set(archive.namelist())=={'inputs.npy','reference.npy'}and
                all(info.file_size<=4*1024**2 for info in archive.infolist()),'Bounded original Native reference archive')
        shape,stored=npy(archive.read('reference.npy'))
        require(shape==(4,len(references)//4)and stored==references,'Independent original Native mathematical reference')
        _,stored=npy(archive.read('inputs.npy'))
        period=q['layout']['input_slot_period'];packed=[v for sample in vectors for v in sample+[0.]*(period-len(sample))]
        require(stored==packed,'Original fixed bounded input probes')
    measured=native_outputs(attempt/'native-execution',read(attempt/'native-execution-report.json',work),
        q,sdk,derived,link,op['owner'],vectors,work)
    require(item['comparison']['passed']is True and item['comparison']['atol']==1e-5 and item['comparison']['rtol']==1e-4 and
            item['comparison']['compared_values']==measured['compared_values'],'Original qualification tolerance/count')
    require(type(case['cache_operations'])is list and 1<=len(case['cache_operations'])<=4 and
            type(case['new_input_operations'])is list and 2<=len(case['new_input_operations'])<=4,'Finite measured Native reuse operations')
    for cached in case['cache_operations']:operation(cached,work,folder,'cache',used_operations)
    input_hashes=set();new_values=0;boots=measured['Native_Boot_calls']
    for execution in case['new_input_operations']:
        require(type(execution)is dict and set(execution)=={'operation','report','inputs'},'Original new-input evidence fields')
        op,row=operation(execution['operation'],work,folder,'execute',used_operations)
        path,result=record(execution['report'],work);_,supplied=record(execution['inputs'],work)
        require(type(supplied)is list and 1<=len(supplied)<=4 and op['operation_input']['sample_format']=='named'and
                op['operation_input']['inputs']==dict(sha256=digest(supplied))and digest(supplied)not in input_hashes,
                'Distinct exact original supplied inputs')
        input_hashes.add(digest(supplied));root=direct(result['evidence'],work,directory=True);registered(root,row)
        require(path==root/'report.json'and result['status']=='executed'and result['native_sdk']is True and
                result['bundle_binding']==package['binding']and result['request_id']==q['request_id']and
                result['encrypted_execution']is True and result['executor_secret_key']is False and
                result['agent_calls']==result['compiler_calls']==0,'Actual original Native new-input execution')
        name=q['model']['inputs'][0]['name'];shape=q['model']['inputs'][0]['shape'];vectors=[]
        for sample in supplied:
            require(type(sample)is dict and set(sample)=={name},'Declared Native supplied input')
            vectors.append(flatten(sample[name],shape))
        measured=native_outputs(root/'native-execution',result['execution'],q,sdk,derived,link,op['owner'],vectors,work)
        new_values+=measured['compared_values'];boots+=measured['Native_Boot_calls']
    return dict(request_id=q['request_id'],package_binding=package['binding'],cache_hits=len(case['cache_operations']),
        new_input_executions=len(case['new_input_operations']),native_qualification_compared_values=len(references),
        new_input_compared_values=new_values,native_Boot_calls=boots,passed=True)


def verify_release(sdk,document,work):
    from native_sdk_manifest import execution_fingerprint
    require(type(document)is dict and set(document)=={'format','execution_fingerprint','cases','old_SDK_and_defaults_preserved','binding'}and
            document['format']==FORMAT and document['old_SDK_and_defaults_preserved']is True and
            document['execution_fingerprint']==execution_fingerprint(sdk),'Versioned complete Native release evidence')
    require(type(document['cases'])is list and len(document['cases'])==3,'Three independently measured genuine Native program contexts')
    first_path,first=record(document['cases'][0]['package_manifest'],work,bound=True)
    from native_program_package import check_bundle
    require(check_bundle(first_path.parent)==first,'Complete first Native release package')
    expected=current_identities(sdk,first['dependencies']['backend_identity'])
    used=set();rows=[verify_case(sdk,case,work,used,expected_identities=expected)for case in document['cases']]
    require(len({v['request_id']for v in rows})==len({v['package_binding']for v in rows})==3,'Distinct genuine Native requests/packages')
    return rows
