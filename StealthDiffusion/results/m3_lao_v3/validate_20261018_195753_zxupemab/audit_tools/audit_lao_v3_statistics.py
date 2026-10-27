#!/usr/bin/env python3
"""Independent saved-file/CSV statistical audit. Never imports attack/report code.

Only run after the frozen evaluation and report exist. Output is a new file.
LPIPS is audited as independent evaluator output and checked against attack
records; this script does not execute LPIPS or any other model.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
from PIL import Image
from scipy.stats import beta, binomtest

METHODS = ('Z','P','L','H','P_EOT','L_EOT','H_EOT')
DETECTORS = ('E','R','D','S')
CONDITIONS = ('none','jpeg75','jpeg85','jpeg95')
SEED, REPLICATES = 20261018, 10000


def check(condition, reason):
    if not condition:
        raise AssertionError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def close(a, b, label, atol=1e-10, rtol=1e-10):
    if a is None or b is None:
        check(a is None and b is None, label+' null mismatch')
    elif isinstance(a, (int,float,np.number)) and not isinstance(a, bool):
        check(math.isclose(float(a),float(b),abs_tol=atol,rel_tol=rtol), f'{label}: {a} != {b}')
    else:
        check(a == b, f'{label}: {a} != {b}')


def compare_fields(computed, reported, label):
    for key, value in computed.items():
        check(key in reported, f'{label} missing {key}')
        close(value,reported[key],label+'/'+key)


def load_pixels(path):
    with Image.open(path) as image:
        check(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224,224), 'not canonical RGB PNG '+str(path))
        return np.array(image,dtype=np.uint8)


def cost_value(record, paths):
    for keys in paths:
        value=record
        for key in keys.split('.'):
            value=value.get(key) if isinstance(value,dict) else None
        if value is not None:
            return len(value) if isinstance(value,list) else float(value)
    return None


def paired_outcomes(left, right, ids):
    a=np.asarray(left,dtype=np.int8); b=np.asarray(right,dtype=np.int8)
    check(len(a)==len(b)==len(ids), 'unpaired data')
    positive=[i for i,x,y in zip(ids,a,b) if x and not y]
    negative=[i for i,x,y in zip(ids,a,b) if y and not x]
    n=len(ids); n10=len(positive); n01=len(negative)
    result=dict(n_eligible=n,left_success=int(a.sum()),right_success=int(b.sum()),left_only=n10,right_only=n01,
                left_only_ids=positive,right_only_ids=negative,both_success=int(np.logical_and(a,b).sum()),
                both_failure=int(np.logical_and(a==0,b==0).sum()),bootstrap_seed=SEED,bootstrap_resamples=REPLICATES)
    if not n:
        result.update(paired_difference=None,ci95_lower=None,ci95_upper=None,exact_mcnemar_p=None,
                      conservative_ci95_lower=None,conservative_ci95_upper=None)
        return result
    delta=(a-b).tolist(); rng=random.Random(SEED)
    # Preserve the prespecified image-order and Python RNG stream; use NumPy
    # for independent percentile aggregation and SciPy for exact binomial/CP.
    samples=np.array([sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(REPLICATES)])
    interval=np.quantile(samples,[.025,.975],method='linear')
    cp=lambda k: (0. if k==0 else float(beta.ppf(.0125,k,n-k+1)),
                  1. if k==n else float(beta.ppf(.9875,k+1,n-k)))
    ci10,ci01=cp(n10),cp(n01)
    result.update(paired_difference=float(np.mean(a-b)),ci95_lower=float(interval[0]),ci95_upper=float(interval[1]),
                  exact_mcnemar_p=float(binomtest(n10,n10+n01,.5,alternative='two-sided').pvalue) if n10+n01 else 1.,
                  conservative_ci95_lower=ci10[0]-ci01[1],conservative_ci95_upper=ci10[1]-ci01[0])
    return result


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    return value


def audit(root, n):
    root=Path(root).resolve(); evaluation=root/'evaluation'; report=read(root/'report/report.json')
    mapping=read(root/'methods.json'); check(set(mapping)==set(METHODS),'method coverage')
    check(report['status']=='verified' and report['n_generated']==report['n_real']==n,'report counts/status')
    check(report['budget_pixels']==8 and report['quality_screen']=={'psnr_min':30.,'lpips_max':.03},'quality/budget changed')
    check(report['mean_lpips_max']==.015,'mean guard changed')
    check(tuple(report['conditions'])==CONDITIONS and tuple(report['detectors'])==DETECTORS,'report condition/detector coverage')
    rows=csv_rows(evaluation/'per_image.csv'); cells={}
    for row in rows:
        key=tuple(row[field] for field in ('run','detector','condition','image_id'))
        check(key not in cells,'duplicate CSV '+str(key)); cells[key]=row
    records={}; ids=None; originals={}; qualities={}; costs={}; integrity=[]; margin_diffs=[]; counter_checks=0
    source_dir=root/'data/gen/test'; source_files={p.name:p for p in source_dir.glob('*.png')}
    check(len(source_files)==n,'canonical source count')
    for name in METHODS:
        folder=Path(mapping[name]); conf=read(folder/'run_config.json')
        rr=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines() if line.strip()]
        rec={str(row['image_id']):row for row in rr}; records[name]=rec
        check(len(rr)==len(rec)==n,'record count '+name)
        group_ids=sorted(rec)
        if ids is None: ids=group_ids
        check(ids==group_ids,'method IDs changed')
        check({row['source'] for row in rr}==set(source_files),'source selection differs '+name)
        if 'train_views' in conf:
            expected=('none','jpeg80','jpeg90') if name.endswith('_EOT') else ('none',)
            check(tuple(conf['train_views'])==expected,'heldout view used in optimization '+name)
        all_lp=[]; all_psnr=[]; maximums=[]
        for key in ids:
            record=rec[key]; src=source_files[record['source']]; original_path=folder/f'{key}_originImage.png'; output_path=folder/f'{key}_adv_image.png'
            x0=load_pixels(src); x=load_pixels(original_path); y=load_pixels(output_path)
            check(np.array_equal(x0,x),'actual canonical source pixels differ '+name+'/'+key)
            original_hash=sha(original_path)
            if key not in originals: originals[key]=original_hash
            check(original_hash==originals[key],'original PNG bytes differ across methods')
            delta=y.astype(np.int16)-x0.astype(np.int16); maximum=int(np.abs(delta).max())
            check(maximum<=8,'strict integer budget violation '+name+'/'+key)
            mse=float(np.mean(delta.astype(np.float64)**2)); psnr=math.inf if mse==0 else float(10*np.log10(255**2/mse))
            own=[]
            for det in DETECTORS:
                row=cells[name,det,'none',key]; lp=float(row['lpips']); own.append(lp)
                close(psnr,float(row['psnr']),'actual PNG PSNR',atol=1e-4,rtol=1e-6)
                close(maximum/255,float(row['linf']),'actual PNG Linf',atol=1e-7,rtol=1e-6)
            check(len(set(own))==1,'LPIPS differs across detector evaluations')
            lp=own[0]; check(math.isfinite(lp) and lp>=0,'invalid LPIPS')
            if 'lpips' in record: close(lp,record['lpips'],'record/eval LPIPS',atol=1e-6,rtol=1e-6)
            output_hash=sha(output_path)
            if record.get('output_sha256'): check(output_hash==record['output_sha256'],'record output hash')
            if record.get('source_file_sha256'): check(sha(src)==record['source_file_sha256'],'record source hash')
            if record.get('final',{}).get('E'):
                final=record['final']['E']; eval_e=cells[name,'E','none',key]
                check(final['pred']==int(eval_e['adv_pred']),'saved/eval classification changed')
                close(final['real_margin'],float(eval_e['adv_margin']),'saved/eval E margin',atol=2e-5,rtol=1e-5)
                margin_diffs.append({'run':name,'id':key,'absolute_difference':abs(final['real_margin']-float(eval_e['adv_margin']))})
            if 'latent_gradient_steps' in record and 'pixel_gradient_steps' in record:
                for stage in ('latent','pixel'):
                    steps=record[stage+'_gradient_steps']; check(isinstance(steps,list),'gradient list missing')
                    if 'actual_'+stage+'_gradient_steps' in record:
                        check(record['actual_'+stage+'_gradient_steps']==len(steps),'actual stage gradient count')
                    for entry in steps:
                        for field in ('grad_l2','grad_linf','update_l2','update_linf'):
                            if field in entry: check(math.isfinite(entry[field]) and entry[field]>=0,'invalid gradient/update norm')
                check(record['actual_gradient_steps']==len(record['latent_gradient_steps'])+len(record['pixel_gradient_steps']),'actual gradient total')
                counter_checks+=1
            qualities[name,key]={'lpips':lp,'psnr':psnr,'max_difference_pixels':maximum,
                                 'qualified':lp<=.03 and psnr>=30,'legacy_qualified':lp<=.15 and psnr>=30}
            all_lp.append(lp); all_psnr.append(psnr); maximums.append(maximum)
            integrity.append({'run':name,'id':key,'source':record['source'],'origin_sha256':original_hash,'output_sha256':output_hash,'max_difference_pixels':maximum})
        qsummary=dict(n=n,qualified=sum(qualities[name,key]['qualified'] for key in ids),legacy_qualified=sum(qualities[name,key]['legacy_qualified'] for key in ids),
                      lpips_mean=statistics.mean(all_lp),lpips_median=statistics.median(all_lp),lpips_max=max(all_lp),lpips_p95=float(np.quantile(all_lp,.95,method='linear')),
                      psnr_mean=statistics.mean(all_psnr),psnr_median=statistics.median(all_psnr),maximum_pixel_difference=max(maximums),mean_lpips_guard_pass=statistics.mean(all_lp)<=.015)
        compare_fields(qsummary,report['methods'][name]['quality'],name+' quality')
        costs[name]={'quality':qsummary}
        fields={'seconds':('seconds','elapsed_sec'),'gradient_steps':('actual_gradient_steps','actual_steps'),
                'latent_gradient_steps':('actual_latent_gradient_steps','compute.latent_gradient_steps','latent_gradient_steps'),
                'pixel_gradient_steps':('actual_pixel_gradient_steps','compute.pixel_gradient_steps','pixel_gradient_steps'),
                'detector_forwards':('detector_forwards','compute.detector_forwards','compute.E_forwards','objective_detector_forwards'),
                'detector_view_examples':('detector_view_examples','compute.detector_view_examples','compute.E_view_examples'),
                'lpips_forwards':('lpips_forwards','compute.lpips_forwards','compute.LPIPS_forwards','objective_lpips_forwards'),
                'renderer_forwards':('renderer_forwards','compute.renderer_forwards','objective_renderer_forwards'),
                'jpeg_forwards':('jpeg_forwards','compute.jpeg_forwards','compute.JPEG_encodes')}
        for field,paths in fields.items():
            values=[cost_value(row,paths) for row in rr]; present=[v for v in values if v is not None]
            check(all(math.isfinite(v) and v>=0 for v in present),'invalid cost '+field)
            stat=dict(available_n=len(present),total=sum(present) if present else None,mean=statistics.mean(present) if present else None,
                      median=statistics.median(present) if present else None,complete=len(present)==n)
            compare_fields(stat,report['methods'][name]['compute'][field],name+' cost/'+field); costs[name][field]=stat
        shared=sum(bool(row.get('seconds_is_shared',row.get('timing_is_shared',False))) for row in rr)
        close(shared,report['methods'][name]['compute']['shared_timing_records'],'shared timing count')
        costs[name]['shared_timing_records']=shared
        costs[name]['cross_method_totals_must_not_be_summed']=shared>0
    expected={(name,det,cond,key) for name in METHODS for det in DETECTORS for cond in CONDITIONS for key in ids}
    check(set(cells)==expected,'evaluation coverage')
    for det in DETECTORS:
        for cond in CONDITIONS:
            for key in ids:
                cp={cells[name,det,cond,key]['clean_pred'] for name in METHODS}
                check(len(cp)==1,'clean eligibility differs across methods')
                clean_margins=[float(cells[name,det,cond,key]['clean_margin']) for name in METHODS]
                check(max(clean_margins)-min(clean_margins)<=2e-5,'clean margins differ')
    reset_checks=0; stage_checks=0
    for name in ('L','H','L_EOT','H_EOT'):
        for key in ids:
            check(sha(Path(mapping[name])/f'{key}_reset_image.png')==sha(Path(mapping['Z'])/f'{key}_adv_image.png'),'reset != independent Z')
            reset_checks+=1
    for hybrid,latent in (('H','L'),('H_EOT','L_EOT')):
        for key in ids:
            check(sha(Path(mapping[hybrid])/f'{key}_lao_image.png')==sha(Path(mapping[latent])/f'{key}_adv_image.png'),'hybrid latent intermediate differs')
            stage_checks+=1
    metrics=[]; robust=[]; robust_flags={}; cohorts={}; joint_details={}
    for name in METHODS:
        for det in DETECTORS:
            for cond in CONDITIONS:
                eligible=[key for key in ids if int(cells[name,det,cond,key]['clean_pred'])==0]
                wins=[key for key in eligible if int(cells[name,det,cond,key]['adv_pred'])==1]
                q=[key for key in wins if qualities[name,key]['qualified']]; legacy=[key for key in wins if qualities[name,key]['legacy_qualified']]
                cell=dict(run=name,detector=det,condition=cond,n=n,n_clean_detected=len(eligible),success=len(wins),cond_asr=len(wins)/len(eligible) if eligible else None,
                          quality_success=len(q),quality_asr=len(q)/len(eligible) if eligible else None,legacy_quality_success=len(legacy),legacy_quality_asr=len(legacy)/len(eligible) if eligible else None,
                          uncond_escape=sum(int(cells[name,det,cond,key]['adv_pred'])==1 for key in ids)/n)
                metrics.append(cell)
            cohort=[key for key in ids if all(int(cells[name,det,cond,key]['clean_pred'])==0 for cond in CONDITIONS)]
            if det in cohorts:check(cohort==cohorts[det],'joint cohort differs')
            cohorts[det]=cohort
            success={key:all(int(cells[name,det,cond,key]['adv_pred'])==1 for cond in CONDITIONS) for key in ids}
            qualified={key:success[key] and qualities[name,key]['qualified'] for key in ids}; robust_flags[name,det]=qualified
            win=[key for key in cohort if success[key]]; qwin=[key for key in cohort if qualified[key]]
            robust.append(dict(run=name,detector=det,conditions=list(CONDITIONS),n=n,n_joint_clean_detected=len(cohort),joint_success=len(win),joint_asr=len(win)/len(cohort) if cohort else None,
                               quality_joint_success=len(qwin),quality_joint_asr=len(qwin)/len(cohort) if cohort else None,uncond_joint_escape=sum(success.values())/n,eligible_ids=cohort,quality_success_ids=qwin))
            for key in ids:
                joint_details[name,det,key]=dict(clean_joint_eligible=key in cohort,all_conditions_real=success[key],original_quality_qualified=qualities[name,key]['qualified'],eligible_quality_success=key in qwin,**qualities[name,key])
    reported_metrics={(row['run'],row['detector'],row['condition']):row for row in report['metrics']}
    check(len(reported_metrics)==len(report['metrics'])==len(metrics)==112,'metric cell coverage')
    for metric in metrics:compare_fields(metric,reported_metrics[metric['run'],metric['detector'],metric['condition']],'metric')
    reported_robust={(row['run'],row['detector']):row for row in report['robustness']}
    check(len(reported_robust)==len(report['robustness'])==28,'robust cell coverage')
    for cell in robust:compare_fields(cell,reported_robust[cell['run'],cell['detector']],'robustness')
    reported_joint={(row['run'],row['detector'],row['image_id']):row for row in report['per_image_joint_outcomes']}
    check(set(reported_joint)==set(joint_details),'per-image joint coverage')
    for key,detail in joint_details.items():compare_fields(detail,reported_joint[key],'per-image joint '+str(key))
    real_rows=csv_rows(evaluation/'real_predictions.csv'); real={}
    for row in real_rows:
        key=row['detector'],row['condition'],row['image'];check(key not in real,'duplicate real prediction');real[key]=int(row['pred'])
    real_ids=sorted({key[2] for key in real});check(len(real_ids)==n,'real count')
    check(set(real)=={(d,c,i) for d in DETECTORS for c in CONDITIONS for i in real_ids},'real coverage')
    check({p.name for p in (root/'data/real/test').glob('*.png')}==set(real_ids),'canonical real identities')
    fprs=[]
    for det in DETECTORS:
        for cond in CONDITIONS:
            count=sum(real[det,cond,i]==0 for i in real_ids)
            fprs.append(dict(detector=det,condition=cond,n_real=n,false_positives=count,real_fpr=count/n))
    reported_fpr={(row['detector'],row['condition']):row for row in report['real_fpr']}
    check(len(reported_fpr)==len(report['real_fpr'])==16,'FPR coverage')
    for fpr in fprs:compare_fields(fpr,reported_fpr[fpr['detector'],fpr['condition']],'FPR')
    comparisons=[]
    specs=[('H_EOT','P_EOT','primary'),('L_EOT','P_EOT','secondary'),('H_EOT','L_EOT','secondary'),
           ('P_EOT','P','descriptive_eot'),('L_EOT','L','descriptive_eot'),('H_EOT','H','descriptive_eot')]
    comparison_index={(row['left'],row['right']):row for row in report['paired_comparisons']}
    check(len(comparison_index)==len(report['paired_comparisons'])==len(specs),'paired comparison coverage')
    for left,right,role in specs:
        cohort=cohorts['E']; values=paired_outcomes([robust_flags[left,'E'][key] for key in cohort],[robust_flags[right,'E'][key] for key in cohort],cohort)
        guard=costs[left]['quality']['mean_lpips_guard_pass'] and costs[right]['quality']['mean_lpips_guard_pass']
        superior=bool(values['n_eligible'] and values['ci95_lower']>0 and values['exact_mcnemar_p']<.05 and guard)
        values.update(left=left,right=right,role=role,detector='E',both_mean_lpips_guards_pass=guard,superiority_criterion_met=superior)
        compare_fields(values,comparison_index[left,right],'paired '+left+'-'+right);comparisons.append(values)
    compare_fields(comparisons[0],report['primary'],'primary alias')
    summary=csv_rows(evaluation/'summary.csv'); summary_index={(row['run'],row['detector'],row['condition']):row for row in summary}
    check(len(summary_index)==len(summary)==112,'summary count')
    for cell in metrics:
        key=cell['run'],cell['detector'],cell['condition']; actual=summary_index[key]
        for field in ('n','n_clean_detected','uncond_escape'):close(cell[field],float(actual[field]),'eval summary '+field)
        asr=float(actual['cond_asr']);close(cell['cond_asr'],None if math.isnan(asr) else asr,'eval summary ASR')
        close(costs[cell['run']]['quality']['lpips_mean'],float(actual['lpips']),'eval summary LPIPS')
        close(reported_fpr[cell['detector'],cell['condition']]['real_fpr'],float(actual['real_fpr']),'eval summary FPR')
    for filename in ('per_image.csv','real_predictions.csv','summary.csv'):
        check(sha(evaluation/filename)==report['integrity']['input_sha256'][filename],'evaluation file changed')
    checks=dict(pixel_checks=len(integrity),pixel_budget_violations=0,canonical_source_consistency_checks=n*len(METHODS),csv_rows=len(cells),summary_cells=112,real_csv_rows=len(real),independent_reset_checks=reset_checks,hybrid_latent_stage_checks=stage_checks)
    compare_fields(checks,report['integrity'],'integrity counts')
    return dict(status='verified',created_at=datetime.now().astimezone().isoformat(),root=str(root),n_generated=n,n_real=n,
                scope='Independent CSV/actual PNG statistics, quality and compute audit; no attack/report module imports and no model execution',
                primary=comparisons[0],paired_comparisons=comparisons,metrics=metrics,robustness=robust,real_fpr=fprs,quality_and_cost=costs,
                integrity={**checks,'gradient_counter_records_verified':counter_checks,'actual_pixel_evidence':integrity},
                saved_vs_evaluation_margin={'records_checked':len(margin_diffs),'maximum_absolute_difference':max((x['absolute_difference'] for x in margin_diffs),default=None),'classification_changes':0,'relative_tolerance':1e-5,'absolute_tolerance':2e-5},
                exact_methods='Python seeded paired image bootstrap + NumPy quantiles; SciPy exact binomial and beta-quantile CP bounds',
                input_sha256={str(path.relative_to(root)):sha(path) for path in (root/'methods.json',evaluation/'per_image.csv',evaluation/'real_predictions.csv',evaluation/'summary.csv',root/'report/report.json')},
                limitations=['LPIPS is not rerun; this audit checks saved-PNG evaluator values and records.','This script performs no parameter or output selection from formal validation outcomes.','Secondary comparisons are descriptive, not multiplicity-adjusted confirmatory claims.','FPR and the fixed joint clean cohort must accompany any robustness claim.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--expected-n',type=int,default=50,help='50 for formal validation; smaller values only for synthetic contract tests')
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve() if args.out else root/'independent_statistics_audit.json'
    check(not out.exists(),'refusing to overwrite '+str(out))
    check(not ((root/'SEAL.json').exists() and out.is_relative_to(root)),'refusing to write into a sealed root; pass --out outside it')
    result=audit(root,args.expected_n)
    with out.open('x') as handle:json.dump(safe(result),handle,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'out':str(out),'primary':result['primary'],'metric_cells_verified':len(result['metrics']),'robust_cells_verified':len(result['robustness']),'fpr_cells_verified':len(result['real_fpr'])}))


if __name__=='__main__':
    main()
