"""Finite old-development search, then a single frozen third-batch validation."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

from m3_lao_v2_run import sha, save, source_files, stage, verify_inputs, check_sources

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[2] / 'snapshots/before_lao_pgd_eot_v3_20261018_183003'
V2 = HERE / 'results/m3_lao_v2/validate_20261018_172351_7bu0ig1v'
PROTOCOL = HERE.parent / 'docs/M3_LAO_V3_PROTOCOL_ZH.md'


def now():
    return datetime.now().astimezone().isoformat()


def candidates():
    configs = []
    for eot in (False, True):
        suffix = '_EOT' if eot else ''
        for alpha in (.25, .5, 1.0):
            for weight in (1.0, 10.0):
                configs.append(dict(name=f'P{suffix}_a{alpha:g}_w{weight:g}', kind='P', eot=eot,
                    alpha_pixels=alpha, lpips_weight=weight, max_steps=80, restarts=3))
    configs += [dict(name='L', kind='L', eot=False, lao_steps=40, lao_lr=.02, lao_lpips_weight=1.)]
    configs += [dict(name=f'L_EOT_w{w:g}',kind='L',eot=True,lao_steps=40,lao_lr=.02,
                     lao_lpips_weight=w) for w in (1.,10.)]
    configs += [dict(name='Z',kind='Z',eot=False,lao_steps=0,lao_lr=.02,lao_lpips_weight=1.)]
    return configs


def dev_metrics(evaluation, configs, methods):
    from PIL import Image
    import numpy as np
    rows = list(csv.DictReader((evaluation/'per_image.csv').open()))
    metrics = {}
    canonical = {f'{i:04d}': np.array(Image.open(p).convert('RGB'))
                 for i,p in enumerate(sorted((HERE/'data15/gen/dev').glob('*.png')))}
    if len(canonical)!=10:
        raise RuntimeError('Development must contain exactly the original ten images')
    for c in configs:
        name=c['name']; folder=methods[name]
        cell={(r['condition'],r['image_id']):r for r in rows if r['run']==name and r['detector']=='E'}
        if len(cell)!=30:
            raise RuntimeError('Incomplete dev evaluation '+name)
        views=['none','jpeg80','jpeg90'] if c['eot'] else ['none']
        records=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
        if {r['image_id'] for r in records}!=set(canonical) or len(records)!=10:
            raise RuntimeError('Incomplete dev records '+name)
        quality={}; lp=[]
        for iid,clean in canonical.items():
            origin=np.array(Image.open(folder/f'{iid}_originImage.png').convert('RGB'))
            adv=np.array(Image.open(folder/f'{iid}_adv_image.png').convert('RGB'))
            if not np.array_equal(origin,clean) or np.abs(adv.astype('int16')-clean.astype('int16')).max()>8:
                raise RuntimeError('Source or budget violation '+name+'/'+iid)
            r=cell['none',iid]; value=float(r['lpips']); lp.append(value)
            quality[iid]=value<=.03 and float(r['psnr'])>=30
        eligible=[i for i in canonical if all(cell[v,i]['clean_pred']=='0' for v in views)]
        success=[i for i in eligible if quality[i] and all(float(cell[v,i]['adv_margin'])>=.1 for v in views)]
        calls=sum(r.get('compute',{}).get('detector_forwards',r.get('detector_forwards',0)) for r in records)
        metrics[name]=dict(n=10,eligible=len(eligible),success=len(success),success_ids=success,
            mean_lpips=statistics.mean(lp),max_lpips=max(lp),quality_pass=sum(quality.values()),
            mean_guard=statistics.mean(lp)<=.015,detector_forwards=calls,
            output_sha256={i:sha(folder/f'{i}_adv_image.png') for i in canonical})
    return metrics


def select(configs, metrics):
    if any(metrics[c['name']]['eligible']==0 for c in configs):
        raise RuntimeError('Empty development eligibility; cannot select')
    pool=[c for c in configs if metrics[c['name']]['mean_guard']] or configs
    return min(pool,key=lambda c:(-metrics[c['name']]['success'],metrics[c['name']]['mean_lpips'],
        metrics[c['name']]['detector_forwards'],configs.index(c)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['dev','validate'],required=True)
    p.add_argument('--root',type=Path)
    p.add_argument('--candidate_freeze',type=Path)
    p.add_argument('--device',choices=['mps','cpu','cuda'],default='mps')
    a=p.parse_args(argv)
    if json.loads((ARCHIVE/'snapshot.json').read_text()).get('status')!='verified':
        raise RuntimeError('Snapshot not verified')
    prior=json.loads((V2/'plan.json').read_text())
    model=prior['diffusion_model']
    for rel,h in model['file_sha256'].items():
        if sha(Path(model['resolved_local_path'])/rel)!=h: raise RuntimeError('Model changed '+rel)
    parent=HERE/'results/m3_lao_v3'; parent.mkdir(exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix=a.phase+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_',dir=parent)) if a.root is None else a.root.resolve()
    if a.root is not None: root.mkdir(parents=True,exist_ok=False)
    import lossers.lpips
    lpips_root=Path(lossers.lpips.__file__).resolve().parent
    sources={str(f):sha(f) for f in source_files(HERE)+sorted(lpips_root.rglob('*.py'))}
    checkpoints={**prior['checkpoint_sha256'],
        str(HERE/'.cache/checkpoints/vgg16-397923af.pth'):sha(HERE/'.cache/checkpoints/vgg16-397923af.pth'),
        str(lpips_root/'vgg_v0.1.pth'):sha(lpips_root/'vgg_v0.1.pth')}
    plan=dict(created_at=now(),phase=a.phase,working_directory=str(HERE),snapshot=str(ARCHIVE),
        source_sha256=sources,checkpoint_sha256=checkpoints,diffusion_model=model,
        protocol=str(PROTOCOL),protocol_sha256=sha(PROTOCOL),device=a.device,model_training=False,
        budget_pixels=8,quality=dict(lpips_max=.03,psnr_min=30,mean_lpips_max=.015))
    for source in sources:
        f=Path(source)
        rel=f.relative_to(HERE) if f.is_relative_to(HERE) else Path('environment/lossers/lpips')/f.relative_to(lpips_root)
        dest=root/'frozen_code'/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
    shutil.copy2(PROTOCOL,root/'protocol.md')
    py=[sys.executable,'-B','-u']; state=dict(status='planned',created_at=now(),completed=[])
    def run(label, command):
        for rel,h in model['file_sha256'].items():
            if sha(Path(model['resolved_local_path'])/rel)!=h:
                raise RuntimeError('Diffusion weights changed before '+label)
        plan.setdefault('commands',{})[label]=command;save(root/'plan.json',plan)
        stage(root,label,command,plan,state)
    def attack(label,configs,inputs):
        cfg=root/(label+'_config.json');save(cfg,configs)
        run(label,py+['m3_lao_v3.py','--device',a.device,'--images_root',str(inputs),'--out_root',str(root/'methods'),
            '--config',str(cfg),'--pretrained_diffusion_path',model['resolved_local_path']])
        return {c['name']:root/'methods'/c['name'] for c in configs}
    def evaluate(label,methods,real,full=False):
        run(label,py+['m2_evaluate.py','--device',a.device,'--real_dir',str(real),'--out',str(root/label),
            '--detectors','E,R,D,S' if full else 'E','--conditions','none,jpeg75,jpeg85,jpeg95' if full else 'none,jpeg80,jpeg90',
            '--runs',*[n+'='+str(d) for n,d in methods.items()]])
    try:
        if a.phase=='dev':
            inputs=HERE/'data15/gen/dev'; real=HERE/'data15/real/dev'
            plan.update(input_sha256={f.name:sha(f) for f in inputs.glob('*.png')},real_sha256={f.name:sha(f) for f in real.glob('*.png')},candidates=candidates())
            configs=candidates()
            methods=attack('base_candidates',configs,inputs)
            evaluate('eval_base',methods,real)
            metrics=dev_metrics(root/'eval_base',configs,methods)
            chosen={};hybrids=[]
            for eot in (False,True):
                suffix='_EOT' if eot else ''
                pc=select([c for c in configs if c['kind']=='P' and c['eot']==eot],metrics)
                lc=select([c for c in configs if c['kind']=='L' and c['eot']==eot],metrics)
                chosen['P'+suffix]={**pc,'name':'P'+suffix}
                chosen['L'+suffix]={**lc,'name':'L'+suffix}
                hybrids.append({**pc,**{k:v for k,v in lc.items() if k.startswith('lao_')},'name':'H'+suffix,'kind':'H','max_steps':40})
            save(root/'base_selection.json',dict(created_at=now(),metrics=metrics,chosen=chosen,hybrids=hybrids))
            verify_inputs(inputs,plan['input_sha256']); verify_inputs(real,plan['real_sha256'])
            hm=attack('hybrids',hybrids,inputs); methods.update(hm)
            evaluate('eval_hybrids',hm,real)
            metrics.update(dev_metrics(root/'eval_hybrids',hybrids,hm))
            verify_inputs(inputs,plan['input_sha256']);verify_inputs(real,plan['real_sha256'])
            chosen.update({c['name']:c for c in hybrids}); chosen['Z']=configs[-1]
            evidence=dict(created_at=now(),status='verified',split='original ten-image development only',metrics=metrics,
                selected_configuration=chosen,method_paths={n:str(d) for n,d in methods.items()},
                selection_rule='mean LPIPS <= .015, most quality success, least mean LPIPS, least detector calls, fixed candidate order; no grid expansion',
                evaluation_sha256={str(root/e/'per_image.csv'):sha(root/e/'per_image.csv') for e in ('eval_base','eval_hybrids')})
            save(root/'selection.json',evidence)
            check_sources(plan)
            save(root/'freeze.json',dict(status='frozen',created_at=now(),selected_configuration=chosen,
                development_evidence=dict(path=str(root/'selection.json'),sha256=sha(root/'selection.json')),
                source_sha256=sources,checkpoint_sha256=plan['checkpoint_sha256'],diffusion_model=model,
                protocol=str(PROTOCOL),protocol_sha256=sha(PROTOCOL),snapshot=str(ARCHIVE)))
            print('FREEZE '+str(root/'freeze.json'),flush=True)
        else:
            if a.candidate_freeze is None: raise ValueError('--candidate_freeze required')
            fp=a.candidate_freeze.resolve();freeze=json.loads(fp.read_text())
            if freeze['status']!='frozen' or freeze['source_sha256']!=sources or freeze['protocol_sha256']!=sha(PROTOCOL):
                raise RuntimeError('Frozen code or protocol differs')
            if freeze['checkpoint_sha256']!=plan['checkpoint_sha256'] or freeze['diffusion_model']!=model:
                raise RuntimeError('Frozen detector or diffusion model differs')
            ev=freeze['development_evidence']
            if sha(ev['path'])!=ev['sha256']: raise RuntimeError('Selection evidence changed')
            evidence=json.loads(Path(ev['path']).read_text())
            if evidence.get('status')!='verified' or evidence.get('split')!='original ten-image development only':
                raise RuntimeError('Development evidence is not the expected verified split')
            if evidence['selected_configuration']!=freeze['selected_configuration']:
                raise RuntimeError('Frozen configuration differs from development selection')
            plan.update(candidate_freeze=str(fp),candidate_freeze_sha256=sha(fp),selected_configuration=freeze['selected_configuration'])
            run('data',py+['m3_lao_newdata_v3.py','--out',str(root/'data'),'--candidate_freeze',str(fp),
                '--model_path',prior['commands']['data'][prior['commands']['data'].index('--model_path')+1], '--device',a.device,'--n','50'])
            inputs=root/'data/gen/test';real=root/'data/real/test'
            plan.update(input_sha256={f.name:sha(f) for f in inputs.glob('*.png')},real_sha256={f.name:sha(f) for f in real.glob('*.png')})
            if len(plan['input_sha256'])!=50 or len(plan['real_sha256'])!=50: raise RuntimeError('Incomplete data')
            methods=attack('attacks',list(freeze['selected_configuration'].values()),inputs)
            verify_inputs(inputs,plan['input_sha256']);verify_inputs(real,plan['real_sha256'])
            evaluate('evaluation',methods,real,full=True)
            save(root/'methods.json',{n:str(d) for n,d in methods.items()})
            run('report',py+['m3_lao_v3_report.py','--evaluation',str(root/'evaluation'),'--methods',str(root/'methods.json'),'--out',str(root/'report'),'--n','50'])
        state.update(status='complete',finished_at=now());save(root/'session.json',state)
        print('COMPLETE '+str(root),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=str(exc),finished_at=now());save(root/'session.json',state);raise


if __name__=='__main__': main()
