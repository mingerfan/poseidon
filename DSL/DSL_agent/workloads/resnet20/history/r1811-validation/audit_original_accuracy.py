import ast,hashlib,json,math,struct
from collections import Counter
from pathlib import Path
A=Path('/home/lhohy/Code/Poseidon-generalization');B=Path('/home/lhohy/Code/Poseidon-resnet20');ROOT=Path('/home/lhohy/Code/Poseidon');W=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux')
R=W/'results/agent-resnet20-original-accuracy-r1805';DATA=W/'results/agent-cifar-offline-records-r1804/cifar10-records'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 x=dict(x);x['binding']=hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
 return x
def npy(p,descriptor,shape):
 raw=Path(p).read_bytes();assert raw[:6]==b'\x93NUMPY'
 version=tuple(raw[6:8]);assert version in ((1,0),(2,0))
 size=2 if version==(1,0) else 4;length=int.from_bytes(raw[8:8+size],'little');start=8+size+length
 header=ast.literal_eval(raw[8+size:start].decode('latin1').strip())
 assert header==dict(descr=descriptor,fortran_order=False,shape=shape)
 return raw[start:]
report=json.loads((R/'report.json').read_text());assert report['status']=='succeeded'and report['images']==10000 and report['batch_size']==1 and report['dtype']=='torch.float32'
assert report['weight_sha256']=='12fca82f0bebc4135bf1f32f6e3710e61d5108578464b84fd6d7f5c1b04036c8'
assert report['original_exact_relu']and report['weights_and_model_state_unchanged']and not report['downloaded_code_executed']and report['restricted_weights_only_loader']
assert report['api_calls']==report['encrypted_calls']==report['new_downloads']==0
dataset=json.loads((DATA/'dataset.json').read_text());assert sha(DATA/'dataset.json')==report['dataset_sha256']and dataset['source_md5']=='c32a1d4ab5d03f1284b67883e8d87530'
test=sorted([x for x in dataset['shards']if x['split']=='test'],key=lambda x:x['start']);assert len(test)==10
confusion=[[0]*10 for _ in range(10)];correct=0;all_logits=[];states=set()
for i,part in enumerate(test):
 case=R/('test-%02d'%i);item=json.loads((case/'report.json').read_text());assert item==report['shards'][i]
 for name,record in item['files'].items():assert sha(case/name)==record['sha256']and(case/name).stat().st_size==record['bytes']
 raw=(DATA/part['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==part['sha256']
 labels=raw[::3073];assert len(labels)==1000 and labels==npy(case/'labels.npy','|u1',(1000,))
 logits=npy(case/'logits.npy','<f4',(1000,10));assert len(logits)==40000
 predictions=npy(case/'predictions.npy','|u1',(1000,));assert len(predictions)==1000
 local=0
 for row,(expected,predicted)in enumerate(zip(labels,predictions)):
  values=struct.unpack_from('<10f',logits,40*row);assert all(math.isfinite(x)for x in values)
  independent=max(range(10),key=lambda x:values[x]);assert predicted==independent
  local+=predicted==expected;confusion[expected][predicted]+=1
 assert local==item['correct']
 correct+=local;states.add(item['model_state_sha256']);all_logits.append(logits)
assert len(states)==1 and correct==report['correct']==9173
assert confusion==report['confusion_matrix']and [sum(row)for row in confusion]==[1000]*10
assert npy(R/'full-test-logits.npy','<f4',(10000,10))==b''.join(all_logits)
selections=json.loads((R/'frozen-selections.json').read_text());assert sha(R/'frozen-selections.json')==report['selections_sha256']
cache={}
for key,count,per_class,split in [('calibration',1000,100,'train'),('encrypted_test',40,4,'test')]:
 selected=selections[key];assert len(selected)==count and Counter(x['label']for x in selected)==Counter({i:per_class for i in range(10)})
 assert len({x['index']for x in selected})==count
 for row in selected:
  assert row['shard'].startswith(split+'-')
  if row['shard']not in cache:cache[row['shard']]=(DATA/row['shard']).read_bytes()
  data=cache[row['shard']][3073*row['row']:3073*(row['row']+1)];assert len(data)==3073 and data[0]==row['label']and hashlib.sha256(data).hexdigest()==row['original_row_sha256']
assert not selections['test_fit_for_approximation']
proof=json.loads((R/'host-owner-proof.json').read_text());assert proof['verified_from_host']
ledger=json.loads(Path('/home/lhohy/poseidon-work/coordination/v1/ledger.json').read_text());batch=ledger['batches']['resnet20-original-accuracy-r1805'];assert batch['status']=='succeeded'and batch['active']is None and all(v==0 for v in batch['used'].values())
doc=B/'DSL/DSL_agent/docs/whole_models'
value=dict(format='poseidon-original-resnet20-complete-test-certificate-v1',status='succeeded',original_model_top1_percent=correct/100.,correct=correct,total=10000,official_archive_md5=dataset['source_md5'],dataset_sha256=report['dataset_sha256'],weights_sha256=report['weight_sha256'],model_source_sha256=report['model_source_sha256'],dtype='torch.float32',batch_size=1,exact_original_relu=True,normalization=report['preprocessing'],training_calibration_frozen=1000,training_calibration_per_class=100,encrypted_test_frozen=40,encrypted_test_per_class=4,independent_stored_logits_and_labels_top1_recount=True,independent_standard_library_auditor=True,model_state_identical_across_all_ten_workers=True,test_set_not_fit_for_approximation=True,source_report=str(R/'report.json'),source_report_sha256=sha(R/'report.json'),source_plan_binding=report['plan_binding'],frozen_selections=str(R/'frozen-selections.json'),frozen_selections_sha256=sha(R/'frozen-selections.json'),full_test_logits_sha256=sha(R/'full-test-logits.npy'),original_four_valid_domain_reference_certificate=str(doc/'resnet20-original-reference-r1768.json'),source_sdk_native_limits_unchanged=True,api_calls=0,encrypted_calls=0,new_downloads=0,approximation_calibrated=False,approximate_top1_not_measured=True,required_original_vs_approximate_top1_absolute_difference_percentage_points=1.0,whole_network_agent_generation_not_done=True,whole_network_ckks_not_done=True,b4_complete=False,b5_complete=False,goal_complete=False)
saved=save(doc/'resnet20-original-accuracy-r1811.json',value)
with(doc/'resnet20-original-accuracy-r1811.md').open('x')as f:f.write('# 原 ResNet20 完整测试集准确率 r1811\n\n官方CIFAR10全部10000张测试图，原float32模型/权重、精确ReLU、batch1和原归一化下，预测正确9173张，Top1为**91.73%**。独立标准库审计器重读100000个保存logits、原标签和每图预测，重新计算Top1与混淆矩阵，10个worker的模型状态摘要一致。源模型已在r1768对四组有效域探针、12中间阶段通过独立NumPy/Torch float64差分。\n\n训练集每类首次100张共1000张校准选择已冻结；测试集每类首次4张共40张后续密态选择已冻结，原索引/类别/记录SHA完整保存。测试集未用于近似拟合。使用原受限weights_only加载器，无下载Python执行，新增API/密态/下载均0，原资源限额保持。\n\n后续以原模型与近似模型Top1绝对差不超过1个百分点验收；近似尚未校准，整网新Agent生成、44组整网密态及包重载未完成，B4/B5和完整Goal未完成。\n\n完整证书见 [机器记录](resnet20-original-accuracy-r1811.json)。\n')
line='r1811 原ResNet20全官方测试集基线完成：batch1/原float32/精确ReLU/原权重归一化，9173/10000正确，Top1=91.73%。独立标准库逐项重读保存logits与原标签复算一致；10个worker模型状态同一摘要。训练每类100张共1000校准选择、测试每类4张共40密态选择及原记录SHA已冻结，测试集未拟合，API/密态/新下载0。证书B docs/whole_models/resnet20-original-accuracy-r1811.json；后续仍按原模型/近似Top1绝对差≤1个百分点与原CKKS门限验收，近似校准/整网Agent/44组整网CKKS/包复用尚未完成，B4/B5/完整Goal未完成。\n'
for name in ('TASK_STATE.md','PARALLEL_DEVELOPMENT.md'):
 with(ROOT/'DSL/DSL_agent/docs/coordination'/name).open('a')as f:f.write('\n'+line)
print(json.dumps(dict(binding=saved['binding'],original_test_images=10000,correct=correct,top1_percent=correct/100.,calibration_frozen=1000,encrypted_test_frozen=40,api_calls=0,encrypted_calls=0)))
