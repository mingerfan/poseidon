"""Finite diagnosis of the retained nonlocal-write candidate; no DSL execution."""
import argparse,json
from pathlib import Path
from numerical_failure_r149 import arrays,sha,canonical,polynomial

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 parent=a.results/'stage2-recovery-aggregate-r150.json'
 aggregate=json.loads(parent.read_text())
 rows=[x for x in aggregate['rows'] if x['id']=='construct_119_1']
 if len(rows)!=1 or rows[0]['terminal_failure_layer']!='numerical_comparison':raise ValueError('Wrong frozen failure')
 row=rows[0];e=Path(row['evidence']);rp=e/'report.json'
 if sha(rp)!=row['report_sha256']:raise ValueError('Changed report')
 report=json.loads(rp.read_text());q=json.loads((e/'request.json').read_text())
 for name,h in report['frozen_hashes'].items():
  if sha(e/name)!=h:raise ValueError('Changed frozen input')
 attempt=report['attempts'][-1]
 if attempt['index']!=3 or not attempt['executed'] or attempt['failure_layer']!='numerical_comparison':raise ValueError('Expected final encrypted failure')
 data=arrays(e/'arrays.npz');comparison=attempt['comparison']
 if data['reference']!=comparison['reference'] or len(data['inputs'])!=4:raise ValueError('Frozen reference')
 coeff=[.0625,-.125,.03125,.015625]
 if q['model']['constants']!={'c0':coeff,'c1':coeff} or q['layout']['input_slot_period']!=8:raise ValueError('Model changed')
 predicted=[];independent=[]
 for inputs in data['inputs']:
  x,y,z,t=inputs
  if any(len(v)!=8 for v in inputs):raise ValueError('Packing changed')
  px=[polynomial(v,coeff) for v in x];py=[polynomial(v,coeff) for v in y]
  mean=(sum(z)+sum(t))/3
  # rotate(1).rotate(4) is a positive five-slot shift in period eight.
  predicted.append([px[j]+py[(j+5)%8]+mean for j in range(6)])
  independent.append([polynomial(v,coeff)+mean for v in x[:3]+y[:3]])
 if any(abs(x-y)>1e-12 for aa,bb in zip(independent,data['reference']) for x,y in zip(aa,bb)):raise ValueError('Manual model differs from reference')
 actual=comparison['actual']
 errors=[abs(x-y) for aa,bb in zip(actual,predicted) for x,y in zip(aa,bb)]
 if not all(abs(x-y)<=1e-5+1e-4*abs(y) for aa,bb in zip(actual,predicted) for x,y in zip(aa,bb)):raise ValueError('Candidate formula not confirmed')
 offsets=[x-y for aa,bb in zip(predicted,independent) for x,y in zip(aa,bb)]
 if not all(abs(x-.0625)<1e-12 for x in offsets):raise ValueError('Padding hypothesis not confirmed')
 paths=[parent,rp,e/'request.json',e/'arrays.npz',e/'attempt-03/candidate.py',Path(__file__).resolve(),Path(__file__).with_name('numerical_failure_r149.py')]
 result=dict(format='poseidon-padding-diagnosis-r151',id=row['id'],evidence=str(e),parents={str(p):sha(p) for p in paths},
  original_status='failed',original_failure_layer='numerical_comparison',candidate_source_sha256=sha(e/'attempt-03/candidate.py'),
  confirmed_explanation='Both polynomial branches add 1/16 to padding. The positive-five rotation correctly places the second branch; adding two unmasked branches adds an unwanted 1/16 to every logical output. Public nonlocal counter runs once and multiplies by one.',
  four_input_sets_checked=4,compared_values=len(errors),manual_vs_decryption_max_absolute_error=max(errors),
  manual_vs_reference_offset=.0625,reference_unchanged=True,new_paid_calls=0,new_compilations=0,new_encrypted_executions=0,
  new_agent_passes=0,stage2_complete=False,limitation='Finite manually derived candidate semantics, not a general interpreter or formal proof.')
 result['binding']=__import__('hashlib').sha256(canonical(result)).hexdigest()
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ('binding','compared_values','manual_vs_decryption_max_absolute_error','manual_vs_reference_offset')}))
if __name__=='__main__':main()
