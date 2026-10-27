from pathlib import Path
from datetime import datetime
from collections import Counter
import csv, hashlib, json, time
from PIL import Image
BASE=Path('/Users/wangyunjie/Documents/ChatGPT/FIT5230/github/FIT5230-Ass1/StealthDiffusion')
ROOT=BASE/'results/m3_lao_v3/validate_20261018_195753_zxupemab'; DATA=ROOT/'data'
OUT=ROOT/'independent_data_audit.json'
CLASSES=(('n01440764','tench'),('n02102040','English springer spaniel'),('n02979186','cassette player'),('n03000684','chain saw'),('n03028079','church'),('n03394916','French horn'),('n03417042','garbage truck'),('n03425413','gas pump'),('n03445777','golf ball'),('n03888257','parachute'))
PRIOR_SHA=['eb475c014ae6453ba6a35d14cde7c6b25d0421beaba4b833840bb83b82993737','15cfb748f7f4a69b85d9b37bce0bf5b3515e039a51563439163acde2f0a66fcc']
checks=[]; hashes={}; start=time.perf_counter(); start_at=datetime.now().astimezone().isoformat()
def require(value,label):
 if not value: raise AssertionError(label)
 checks.append(label)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 p=Path(p).resolve()
 if p not in hashes:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  hashes[p]=h.hexdigest()
 return hashes[p]
def dt(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def rgb(p,size=(224,224),crop=False,resize=False):
 with Image.open(p) as im:
  if not resize:
   require(im.format=='PNG' and im.mode=='RGB' and im.size==size,'canonical image format '+str(p))
  im=im.convert('RGB')
  if crop:
   w,h=im.size; s=min(w,h); l,t=(w-s)//2,(h-s)//2; im=im.crop((l,t,l+s,t+s))
  if resize:im=im.resize(size,Image.Resampling.LANCZOS)
  return hashlib.sha256(im.tobytes()).hexdigest()
def verify_ref(r):require(sha(r['path'])==r['sha256'],'referenced SHA '+r['path']);return read(r['path'])
def csv_rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
require(not OUT.exists(),'audit file exclusive')
m=read(ROOT/'data_manifest.json');p=read(DATA/'generation_plan.json');s=read(DATA/'generation_state.json');runtime=read(DATA/'actual_runtime.json');run=read(ROOT/'plan.json')
require(m['status']==s['status']=='verified' and s['completed_images']==50,'generation completed verified')
require(sha(ROOT/'data_manifest.json')==sha(DATA/'data_manifest.json')==s['data_manifest_sha256'],'manifest mirrors and state SHA')
require(m['generation_plan']==p and m['plan_sha256']==sha(DATA/'generation_plan.json')==s['plan_sha256'],'plan binding')
require(m['candidate_freeze']==p['candidate_freeze'] and m['old_inputs']==p['old_inputs'],'embedded freeze and exclusion equality')
require(m['schema_version']==p['schema_version']==3 and m['n_generated']==m['n_real']==50 and len(m['generated'])==len(m['real'])==50,'exact third 50+50 count')
f=verify_ref(p['candidate_freeze']); ev=verify_ref(f['development_evidence'])
require(f['status']=='frozen' and ev['status']=='verified' and ev['split']=='original ten-image development only','frozen original-dev selection')
require(f['selected_configuration']==ev['selected_configuration']==p['candidate_freeze']['selected_configuration']==run['selected_configuration'],'fixed selected configurations')
require(run['candidate_freeze']==p['candidate_freeze']['path'] and run['candidate_freeze_sha256']==p['candidate_freeze']['sha256'],'runner freeze binding')
require(f['source_sha256']==run['source_sha256']==p['candidate_freeze']['source_sha256'],'runner freeze source maps')
require(f['checkpoint_sha256']==run['checkpoint_sha256'] and f['diffusion_model']==run['diffusion_model'],'runner frozen model and checkpoints')
parent=verify_ref(f['parent_freeze']);amend=verify_ref(f['reporting_amendment'])
require(parent['selected_configuration']==f['selected_configuration'] and parent['development_evidence']==f['development_evidence'],'report amendment preserves selections')
require(parent['checkpoint_sha256']==f['checkpoint_sha256'] and parent['diffusion_model']==f['diffusion_model'] and parent['protocol_sha256']==f['protocol_sha256'],'report amendment preserves models and protocol')
changed=[q for q,h in parent['source_sha256'].items() if f['source_sha256'].get(q)!=h]
require(changed==amend['changed_source'] and all(Path(q).name=='m3_lao_v3_report.py' for q in changed),'only reporting source amended')
require(amend['source_before']==parent['source_sha256'][changed[0]] and amend['source_after']==f['source_sha256'][changed[0]],'amendment before after hashes')
for rel,h in amend['artifacts'].items():require(sha(Path(f['reporting_amendment']['path']).parent/rel)==h,'amendment artifact '+rel)
require(dt(ev['created_at'])<=dt(parent['created_at'])<=dt(amend['created_at'])<=dt(f['created_at'])<=dt(run['created_at'])<=dt(p['created_at'])<=dt(s['prepared_at'])<=dt(s['started_at'])<=dt(m['created_at'])<=dt(s['finished_at']),'chronological dev freeze plan preparation generation completion')
for q,h in f['source_sha256'].items():require(sha(q)==h,'current frozen source '+q)
for q,h in f['checkpoint_sha256'].items():require(sha(q)==h,'current frozen checkpoint '+q)
for q,h in f['diffusion_model']['file_sha256'].items():require(sha(Path(f['diffusion_model']['resolved_local_path'])/q)==h,'current SD21 model '+q)
require(sha(f['protocol'])==f['protocol_sha256']==run['protocol_sha256']==sha(ROOT/'protocol.md'),'frozen protocol equality')
require(p['model']['revision']=='451f4fe16113bff5a5d2269ed5ad43b0592e9a14' and p['model']['id']=='stable-diffusion-v1-5/stable-diffusion-v1-5','pinned SD15 identity')
require(p['model']['file_sha256']['unet/diffusion_pytorch_model.bin']=='c7da0e21ba7ea50637bee26e81c220844defdf01aafca02b2c42ecdadb813de4','pinned SD15 UNet SHA')
for q,h in p['model']['file_sha256'].items():require(sha(Path(p['model']['path'])/q)==h,'SD15 generator file '+q)
require(p['steps']==30 and p['guidance_scale']==7.5 and p['dtype']=='float32' and p['seed_generator_device']=='cpu' and p['attention_slicing']=='auto' and p['batch_size']==1,'fixed generation settings')
require(runtime['device'] in ('mps','mps:0') and runtime['dtype']=='torch.float32' and runtime['scheduler_class']=='PNDMScheduler' and runtime['seed_generator_device']=='cpu' and runtime['model_training'] is False,'actual MPS FP32 PNDM CPU seed runtime')
for k,v in p['model']['scheduler'].items():require(runtime['scheduler_config'].get(k)==v,'scheduler config '+k)
require(p['method']=='fresh_same_prompt_family_sd15_seeds_without_outcome_selection' and p['utility_sha256']==sha(BASE/'m3_lao_newdata_v3.py'),'frozen generation method utility')
old=p['old_inputs'];oldrgb=set(); oldids=set(); oldgen=set(); oldfiles=[]
require(len(old['canonical_images'])==472 and len(old['old_rgb224_sha256'])==446 and len(old['old_real_source_ids'])==210,'frozen exclusion counts')
for r in old['canonical_images']:
 require(sha(r['path'])==r['sha256'],'old canonical SHA '+r['path']);h=rgb(r['path']);require(h==r['rgb224_sha256'],'old canonical RGB '+r['path']);oldrgb.add(h);oldfiles.append(r['path'])
 if r['kind']=='real':oldids.add(Path(r['source'].get('source_id',r['source'].get('source',''))).name)
 elif r['kind']=='gen':oldgen.add((r['source']['model'],int(r['source']['seed'])))
require(oldrgb==set(old['old_rgb224_sha256']) and len(oldrgb)==446,'recomputed old 446 RGB contents')
require(oldids==set(old['old_real_source_ids']) and len(oldids)==210,'recomputed old 210 real source IDs')
require(oldgen=={tuple(x) for x in old['old_generation_ids']} and len(oldgen)==236,'recomputed old 236 generation IDs')
for q,h in old['manifest_sha256'].items():require(sha(q)==h,'old original CSV '+q)
prior_evidence=[]
for i,b in enumerate(old['prior_validation_batches']):
 require(b['manifest_sha256']==PRIOR_SHA[i] and sha(b['manifest_path'])==PRIOR_SHA[i],'locked prior batch manifest '+str(i)); bm=read(b['manifest_path']);bp=read(b['plan_path'])
 require(sha(b['plan_path'])==b['plan_sha256']==bm['plan_sha256'] and (i==0 or bm['generation_plan']==bp),'prior plan binding '+str(i))
 require(bp['model']==p['model'],'same pinned generation model as prior '+str(i))
 require(len(bm['generated'])==len(bm['real'])==50,'prior 50+50 '+str(i))
 for kind in ('generated','real'):
  for r in bm[kind]:
   path=Path(b['data_root'])/r['relative_path'];require(str(path) in oldfiles and sha(path)==r['sha256'],'prior canonical inclusion '+str(path))
   if kind=='generated':require(sha(Path(b['data_root'])/r['source512'])==r['source512_sha256'],'prior full-size SHA '+str(path))
 if i==1:
  bf=verify_ref(b['candidate_freeze']);bs=read(Path(b['data_root'])/'generation_state.json')
  require(dt(bf['created_at'])<=dt(bp['created_at']) and bs['status']=='verified' and bs['data_manifest_sha256']==PRIOR_SHA[i],'second batch freeze time and state')
  require(sha(Path(b['data_root'])/'generation_state.json')==b['generation_state_sha256'] and sha(Path(b['data_root'])/'generation_records.jsonl')==b['generation_records_sha256'],'second batch frozen generation evidence')
 prior_evidence.append({'manifest':b['manifest_path'],'sha256':PRIOR_SHA[i],'canonical_count':100})
newrgb=set();newids=set(); observations=[]; ids=[f'{i:04d}' for i in range(50)]
require([r['image_id'] for r in m['generated']]==[r['image_id'] for r in m['real']]==ids,'all consecutive canonical IDs')
for folder in ('gen/test','real/test','gen512'):require(sorted(q.stem for q in (DATA/folder).glob('*.png'))==ids,'exact PNG coverage '+folder)
require(sha(DATA/'gen/test/manifest.csv')==m['generated_manifest_sha256'] and sha(DATA/'real/test/manifest.csv')==m['real_manifest_sha256']==s['real_manifest_sha256'],'new CSV hashes')
require([json.loads(x) for x in (DATA/'generation_records.jsonl').read_text().splitlines()]==m['generated'],'generation log exact records')
for folder,kind in (('gen','generated'),('real','real')):
 rows=csv_rows(DATA/folder/'test/manifest.csv')
 require(rows==[{k:('' if v is None else str(v)) for k,v in r.items()} for r in m[kind]],'CSV content equals JSON '+kind)
for kind in ('generated','real'):
 rows=m[kind];planned=p['generated_samples' if kind=='generated' else 'real_samples']
 require(Counter(r['wnid'] for r in rows)=={c:5 for c,_ in CLASSES},'five per class '+kind)
 for i,(r,pr) in enumerate(zip(rows,planned)):
  require(all(r.get(k)==v for k,v in pr.items()),'planned record exact '+kind+'/'+str(i))
  path=DATA/r['relative_path'];h=rgb(path);require(sha(path)==r['sha256'] and h==r['rgb224_sha256'],'actual canonical hash '+str(path));require(h not in oldrgb and h not in newrgb,'no prior/internal RGB overlap '+str(path));newrgb.add(h)
  require(path.stat().st_mtime>=dt(f['created_at']).timestamp(),'file creation after candidate freeze '+str(path))
  if kind=='generated':
   wnid,name=CLASSES[i%10];require(r['seed']==5233000+i and r['prompt']=='a photo of a '+name and r['wnid']==wnid,'fixed fresh seed prompt '+str(i))
   require((r['model'],r['seed']) not in oldgen and r['model']==p['model']['id'] and r['model_revision']==p['model']['revision'] and r['steps']==30 and r['guidance']==7.5,'generator identity and unused seed '+str(i))
   full=DATA/r['source512'];rgb(full,(512,512));require(sha(full)==r['source512_sha256'] and rgb(full,resize=True)==h,'actual gen512 SHA and LANCZOS source '+str(i))
  else:
   require(r['source_id']==Path(r['source']).name and r['source_id'] not in oldids and r['source_id'] not in newids,'fresh real source identity '+str(i));newids.add(r['source_id']);require(sha(r['source'])==r['source_sha256'] and rgb(r['source'],crop=True,resize=True)==h,'actual real source SHA and crop '+str(i))
  observations.append({'kind':kind,'image_id':r['image_id'],'path':str(path),'sha256':r['sha256'],'rgb224_sha256':h,'seed':r.get('seed'),'source_id':r.get('source_id'),'source512_sha256':r.get('source512_sha256')})
selected=[];seen=set()
for wnid,name in CLASSES:
 for q in sorted((BASE/'data15/imagenette2-320/val'/wnid).iterdir()):
  if q.suffix.lower() not in ('.jpeg','.jpg','.png') or q.name in oldids:continue
  h=rgb(q,crop=True,resize=True)
  if h in oldrgb:continue
  require(h not in seen,'deterministic real selection no duplicate');seen.add(h);selected.append(str(q))
  if sum(Path(x).parent.name==wnid for x in selected)==5:break
require(selected==[r['source'] for r in m['real']],'exact first five lexicographic unused real images per class')
require(len(newrgb)==100 and len(newids)==50,'actual final unique counts')
require(m['novelty']=={'old_canonical_unique_rgb224':446,'new_unique_rgb224':100,'old_overlap':0,'new_internal_duplicates':0,'source_seed_overlap':0,'replacement_count':0,'detector_scores_used_for_selection':False},'novelty metadata agrees with audit')
report={'status':'verified','schema_version':1,'started_at':start_at,'finished_at':datetime.now().astimezone().isoformat(),'seconds':time.perf_counter()-start,'root':str(ROOT),'scope':'Direct read-only hashes and PIL pixels; no model execution, no audit_old_inputs call, no scan of new output origins, no outcome-based selection','checks_passed':len(checks),'data_manifest_sha256':sha(ROOT/'data_manifest.json'),'generation_plan_sha256':sha(DATA/'generation_plan.json'),'generation_state_sha256':sha(DATA/'generation_state.json'),'generation_records_sha256':sha(DATA/'generation_records.jsonl'),'actual_runtime_sha256':sha(DATA/'actual_runtime.json'),'candidate_freeze':p['candidate_freeze']['path'],'candidate_freeze_sha256':p['candidate_freeze']['sha256'],'parent_freeze_sha256':f['parent_freeze']['sha256'],'reporting_amendment_sha256':f['reporting_amendment']['sha256'],'development_evidence':f['development_evidence'],'chronology':{'development_evidence':ev['created_at'],'parent_freeze':parent['created_at'],'reporting_amendment':amend['created_at'],'candidate_freeze':f['created_at'],'run_plan':run['created_at'],'data_plan':p['created_at'],'generation_started':s['started_at'],'generation_finished':s['finished_at']},'counts':{'generated224':50,'generated512':50,'real224':50,'new_unique_rgb224':100,'new_real_source_ids':50,'old_canonical_records':472,'old_unique_rgb224':446,'old_real_source_ids':210,'old_generation_ids':236,'old_rgb_overlap':0,'old_real_source_overlap':0,'old_generation_seed_overlap':0,'internal_duplicates':0},'seed_start':5233000,'seed_end':5233049,'classes':CLASSES,'prior_validation_batches':prior_evidence,'freeze_sources_verified':len(f['source_sha256']),'checkpoints_verified':len(f['checkpoint_sha256']),'SD21_files_verified':len(f['diffusion_model']['file_sha256']),'SD15_files_verified':len(p['model']['file_sha256']),'observations':observations,'checks':checks,'limitations':['The recorded fixed seeds, code, hashes and runtime support provenance; images were not regenerated independently.','Audit validates input identity and novelty, not attack outcomes or semantic quality.']}
with OUT.open('x') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
print(json.dumps({'status':report['status'],'checks_passed':len(checks),'counts':report['counts'],'output':str(OUT),'sha256':sha(OUT),'seconds':report['seconds']},indent=2))
