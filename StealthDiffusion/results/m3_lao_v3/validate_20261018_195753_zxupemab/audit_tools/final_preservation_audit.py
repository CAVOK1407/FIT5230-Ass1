from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib,json,os,stat,time
BASE=Path('/Users/wangyunjie/Documents/ChatGPT/FIT5230');REPO=BASE/'github/FIT5230-Ass1'
ARCHIVE=BASE/'snapshots/before_lao_pgd_eot_v3_20261018_183003'
ROOT=REPO/'StealthDiffusion/results/m3_lao_v3/validate_20261018_195753_zxupemab'
OUT=ROOT/'independent_final_preservation.json'
PREVIOUS=REPO/'StealthDiffusion/results/m3_lao_v3/dev_20261018_184038_1mmcu3yj/independent_preservation.json'
if OUT.exists():raise FileExistsError(OUT)
start=time.perf_counter();started=datetime.now().astimezone().isoformat();cache={}
def sha(p):
 p=Path(p).resolve()
 if p not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  cache[p]=h.hexdigest()
 return cache[p]
def excluded(p):return '.git' in p.parts or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo')
SUFFIXES={'.py','.ipynb','.sh','.bash','.zsh','.js','.ts','.tsx','.jsx','.html','.css','.md','.txt','.yaml','.yml','.toml','.json','.ini','.cfg','.csv'}
def group(p):
 if p.is_relative_to('StealthDiffusion/results'):return 'old_result/'+p.parts[2]
 if p.is_relative_to('peer_eval'):return 'peer_assets'
 if p.suffix.lower() in SUFFIXES or p.name in ('Makefile','Dockerfile','requirements.txt','.gitignore','.gitattributes'):return 'old_source_config_docs_manifests'
 return None
manifest=json.loads((ARCHIVE/'SHA256SUMS.json').read_text());previous=json.loads(PREVIOUS.read_text())
assert sha(PREVIOUS)=='1d1a84d183fe76b1a67a61c3cca8b34ae94ca4a02db7cbdf3da2e15f9a24a2b5'
assert previous['status']=='verified' and previous['snapshot_manifest_sha256']==sha(ARCHIVE/'SHA256SUMS.json')
counts=Counter();sizes=Counter();mismatches=[];observations={};oldpaths=set();oldresults=set();skipped=Counter()
for k,e in manifest.items():
 if not k.startswith('repository/'):continue
 p=Path(k).relative_to('repository');oldpaths.add(p.as_posix())
 if excluded(p):skipped['git_or_python_bytecode']+=1;continue
 category=group(p)
 if category is None:skipped['outside_targeted_recheck_scope']+=1;continue
 if not stat.S_ISREG((ARCHIVE/k).lstat().st_mode):skipped['non_regular_snapshot_entry']+=1;continue
 if p.is_relative_to('StealthDiffusion/results'):oldresults.add(p.parts[2])
 actual=None;reason=None;bytes_now=None
 try:
  st=(REPO/p).lstat()
  if not stat.S_ISREG(st.st_mode):reason='not_regular'
  else:
   actual=sha(REPO/p);bytes_now=st.st_size
   if actual!=e['sha256']:reason='sha256_mismatch'
   elif bytes_now!=e['bytes']:reason='byte_count_mismatch'
 except FileNotFoundError:reason='missing'
 counts[category]+=1;sizes[category]+=e['bytes']
 observations[p.as_posix()]={'sha256':actual,'expected_sha256':e['sha256'],'bytes':bytes_now,'matches':reason is None}
 if reason:mismatches.append({'path':p.as_posix(),'category':category,'reason':reason,**observations[p.as_posix()]})
assert counts['old_result/m3_lao_v2']==1809
unexpected_new=[]
for folder in [REPO/'peer_eval',*[REPO/'StealthDiffusion/results'/n for n in sorted(oldresults)]]:
 for current,dirs,files in os.walk(folder):
  dirs[:]=[d for d in dirs if d not in ('.git','__pycache__')]
  for name in files:
   p=Path(current)/name;rel=p.relative_to(REPO)
   if not excluded(rel) and stat.S_ISREG(p.lstat().st_mode) and rel.as_posix() not in oldpaths:unexpected_new.append(rel.as_posix())
report={'status':'verified' if not mismatches and not unexpected_new else 'mismatch','schema_version':1,'audit':'independent targeted final preservation audit','started_at':started,'finished_at':datetime.now().astimezone().isoformat(),'seconds':time.perf_counter()-start,'archive':str(ARCHIVE),'archive_manifest_sha256':sha(ARCHIVE/'SHA256SUMS.json'),'prior_full_audit':{'path':str(PREVIOUS),'sha256':sha(PREVIOUS),'checked_regular_files':previous['repository_preservation']['checked_regular_files'],'status':previous['status']},'scope':{'all_snapshot_old_results_under_StealthDiffusion_results':True,'all_snapshot_peer_eval_assets':True,'source_configuration_document_manifest_extensions':sorted(SUFFIXES),'explicit_exclusions':['.git metadata','__pycache__','*.pyc','*.pyo','non-regular snapshot entries'],'other_large_data_and_model_weights_not_rehashed':True,'new_v3_files_allowed':True,'active_validation_may_continue':True,'sources_or_protocol_modified_by_audit':False,'model_or_GPU_executed':False},'checked_files':sum(counts.values()),'checked_bytes':sum(sizes.values()),'matched_files':sum(counts.values())-len(mismatches),'categories':{k:{'files':counts[k],'bytes':sizes[k]} for k in sorted(counts)},'skipped_counts':dict(skipped),'mismatch_count':len(mismatches),'mismatches':mismatches,'unexpected_additions_under_old_result_or_peer_roots':unexpected_new,'file_hashes':observations,'audit_script_sha256':sha(__file__)}
with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':report['status'],'checked_files':report['checked_files'],'checked_bytes':report['checked_bytes'],'categories':report['categories'],'mismatches':mismatches,'unexpected_new':unexpected_new,'output':str(OUT),'sha256':sha(OUT),'seconds':report['seconds']},ensure_ascii=False,indent=2))
