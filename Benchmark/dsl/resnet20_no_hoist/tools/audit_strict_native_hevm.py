"""Compare actual compiler bytecode to every literal CKKS lowering instruction."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from verify_native_scale_sidecar import validate


def audit(hevm, manifest_path):
    data=Path(hevm).read_bytes()
    manifest=json.loads(Path(manifest_path).read_text())
    magic,header,nargs,nresults=struct.unpack_from('<IIQQ',data)
    assert magic==0x4845564D and nargs==27 and nresults==1
    body,nops,nct,npt,level=struct.unpack_from('<5Q',data,header)
    assert body==40+(2*nargs+3*nresults)*8
    assert nops==len(manifest['raw_operations'])
    assert (nct,npt)==(manifest['cipher_registers'],manifest['plain_registers'])
    assert level==49
    config=struct.unpack_from('<'+'Q'*(2*nargs+3*nresults),data,header+40)
    assert list(config[:27])==[40]*27
    assert list(config[27:54])==[31]*27
    assert config[-1]==manifest['result_register']
    assert header+body+8*nops==len(data)
    boot=[]
    for index,expected in enumerate(manifest['raw_operations']):
        actual=struct.unpack_from('<4H',data,header+body+8*index)
        if expected['opcode']==-1:
            assert actual[0]==65535,index
        else:
            wanted=tuple(expected[key]&65535 for key in ('opcode','dst','lhs','rhs'))
            assert actual==wanted,(index,actual,wanted)
            if actual[0]==10:
                boot.append(dict(raw_index=index,origin=expected['origin'],output_level=actual[3]))
    assert len(boot)==18 and all(op['output_level']==30 for op in boot)
    scales=validate(data,Path(hevm+'.native_scales').read_text())
    annotated={row['op_index']:row for row in scales}
    for index,expected in enumerate(manifest['raw_operations']):
        if expected['opcode']==0:
            assert index in annotated
            assert int(annotated[index]['bits'],16)==int(expected['native_scale_bits'],16)
    assert len(scales)==sum(row['opcode']==0 for row in manifest['raw_operations'])
    return dict(status='PASS',scope='literal lowering vs real compiler bytecode audit; not native GPU acceptance',
                raw_operations=nops,input_count=27,bootstrap_count=18,f64_encodes=len(scales),
                bootstraps=boot,hevm_sha256=hashlib.sha256(data).hexdigest(),
                runtime_gate='PENDING native multi-rescale coalescing, scale reconcile and 27-input binding')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hevm');parser.add_argument('manifest');parser.add_argument('report')
    args=parser.parse_args()
    result=audit(args.hevm,args.manifest)
    Path(args.report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='bootstraps'}))
