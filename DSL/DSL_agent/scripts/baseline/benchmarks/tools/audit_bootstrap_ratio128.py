"""Read-only mathematical diagnostics for the consumed r235 native bootstrap trial."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/'scripts/baseline'))
from bootstrap_math_audit import read_heap,inverse_coefficient,sine_surrogate,encode_math,decode_math
E=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-bootstrap-ratio128-r236')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=E/'math-diagnostics.json'
 if out.exists():raise RuntimeError('Preserve the existing diagnostic')
 report=json.loads((E/'report.json').read_text());assert report['status']=='failed'
 ratio=report['parameters']['bootstrap']['output_ratio'];assert ratio==128
 source=ROOT/'src/poseidon/advance/bootstrapper.cpp';inverse=inverse_coefficient(read_heap(source)[0])
 stages={}
 for line in (E/'execute.log').read_text().splitlines():
  if line.startswith('value '):
   v=dict(p.split('=',1) for p in line.split()[1:]);key=(v['group'],v['stage'])
   stages.setdefault(key,[]).append(v)
 rows=[]
 for group in ['zero','signed']:
  groups={stage:stages[(group,stage)] for stage in ['first_refresh','square_rescale','second_refresh']}
  assert all([int(x['index']) for x in v]==list(range(16384)) for v in groups.values())
  original=[float(x['reference']) for x in groups['first_refresh']]
  squared=[float(x['reference']) for x in groups['second_refresh']]
  assert all(x*x==y for x,y in zip(original,squared))
  predicted1=sine_surrogate(original,ratio,inverse)
  predicted2=sine_surrogate([x*x for x in predicted1],ratio,inverse)
  actual1=[float(x['actual']) for x in groups['first_refresh']]
  actualsquare=[float(x['actual']) for x in groups['square_rescale']]
  actual2=[float(x['actual']) for x in groups['second_refresh']]
  predicted_from_actual_square=sine_surrogate(actualsquare,ratio,inverse)
  def stats(actual,expected):
   errors=[abs(x-y) for x,y in zip(actual,expected)]
   return dict(max_absolute_error=max(errors),mae=sum(errors)/len(errors),
    failed_values=sum(e>1e-5+1e-4*abs(v) for e,v in zip(errors,expected)))
  by_reference=[]
  for value in dict.fromkeys(squared):
   ix=[i for i,v in enumerate(squared) if v==value]
   by_reference.append(dict(reference=value,count=len(ix),
    failed_values=sum(abs(actual2[i]-value)>1e-5+1e-4*abs(value) for i in ix),
    second_refresh_total_mean_error=sum(actual2[i]-value for i in ix)/len(ix),
    inherited_square_mean_error=sum(actualsquare[i]-value for i in ix)/len(ix),
    additional_second_mean_error=sum(actual2[i]-actualsquare[i] for i in ix)/len(ix)))
  rows.append(dict(group=group,compared_per_stage=16384,
   embedding_roundtrip_max=max(abs(a-b) for a,b in zip(original,decode_math(encode_math(original)))),
   first_refresh_actual_vs_reference=stats(actual1,original),
   first_surrogate_vs_reference=stats(predicted1,original),
   first_actual_minus_surrogate=stats(actual1,predicted1),
   repeated_surrogate_vs_reference=stats(predicted2,squared),
   second_actual_vs_reference=stats(actual2,squared),
   additional_second_vs_previous_square=stats(actual2,actualsquare),
   second_actual_minus_surrogate_of_actual_square=stats(actual2,predicted_from_actual_square),
   by_reference=by_reference))
 result=dict(format='poseidon-bootstrap-ratio128-math-diagnostics-v1',evidence_report_sha256=sha(E/'report.json'),
  evidence_log_sha256=sha(E/'execute.log'),source_sha256=sha(source),inverse_coefficient=inverse,ratio=ratio,
  rows=rows,actual_compile=False,additional_encrypted_execution=False,parameters_changed=False,paid_calls=0,
  inference='Repeated-refresh error contains both inherited square error and additional normalization/refresh error; a sine surrogate does not isolate polynomial, carry or encryption noise.',
  limits=['Not a new trial','Normalized intermediate was not separately decrypted in the frozen probe',
   'Surrogate assumes ideal carry/polynomial behavior','Cannot distinguish every internal CKKS error source',
   'Uses the original independent references for acceptance; surrogate is diagnostic only'])
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='limits'},indent=2))
if __name__=='__main__':main()
