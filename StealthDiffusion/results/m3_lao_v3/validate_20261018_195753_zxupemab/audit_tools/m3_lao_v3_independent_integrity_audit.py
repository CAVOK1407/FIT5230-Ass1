"""Read-only PNG/record integrity audit; imports no model/runtime implementation.
Writes only the explicitly supplied audit JSON, never input artifacts.
"""
import argparse, hashlib, json, math, statistics
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

METHODS=('Z','P','L','H','P_EOT','L_EOT','H_EOT')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pixels(p): return np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8)
def pixelsha(p): return hashlib.sha256(p.tobytes(order='C')).hexdigest()
def finite_tree(value):
    if isinstance(value,float): assert math.isfinite(value),'nonfinite JSON number'
    elif isinstance(value,dict):
        for x in value.values(): finite_tree(x)
    elif isinstance(value,list):
        for x in value: finite_tree(x)
def best(history):
    successful=[h for h in history if h['success']]
    if successful:return min(successful,key=lambda h:h['lpips'])
    valid=[h for h in history if h['low_distortion']]
    return max(valid,key=lambda h:h['worst_view_margin']) if valid else min(history,key=lambda h:h['lpips'])
def check_metrics(h,views):
    assert set(h['view_margins'])==set(views)
    assert h['worst_view_margin']==min(h['view_margins'].values())
    assert h['real_margin']==h['view_margins']['none']
    assert h['pred']==h['view_predictions']['none']
    assert h['max_difference_pixels']<=8
    quality=h['lpips']<=.03 and (h['psnr'] is None or h['psnr']>=30)
    assert h['low_distortion']==quality
    assert h['success']==(quality and min(h['view_margins'].values())>=.1)
def audit(root,n):
    result={'status':'verified','n':n,'scope':'PNG bytes, canonical inputs, source hashes, strict budget, recorded candidate selection, finite logged gradients/update counts, stage/reset equivalence and recorded compute accounting; no model rescoring or gradient reexecution','methods':{},'stage_comparison':{}}
    records={}
    for method in METHODS:
        folder=root/'methods'/method
        cfg=json.loads((folder/'run_config.json').read_text())
        rows=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
        assert len(rows)==n and {r['image_id'] for r in rows}=={f'{i:04d}' for i in range(n)},(method,'coverage')
        records[method]={r['image_id']:r for r in rows}
        counts={'outputs':0,'max_budget':0,'latent_updates':0,'pixel_updates':0,'refinement_probes':0,'nonzero_latent_updates':0,'nonzero_pixel_updates':0,'E_forwards':0,'renderer_forwards':0,'JPEG_encodes':0}
        views=['none','jpeg80','jpeg90'] if cfg['eot'] else ['none']
        for r in rows:
            finite_tree(r); iid=r['image_id']; source=Path(cfg['images_root'])/r['source']
            assert sha(source)==r['source_file_sha256']==cfg['source_sha256'][r['source']]
            with Image.open(source) as image:
                image=image.convert('RGB')
                if image.size!=(224,224):image=image.resize((224,224),Image.Resampling.LANCZOS)
                original=np.asarray(image,dtype=np.uint8)
            origin_path=folder/f'{iid}_originImage.png';adv_path=folder/f'{iid}_adv_image.png'
            origin,adv=pixels(origin_path),pixels(adv_path)
            assert np.array_equal(origin,original),(method,iid,'canonical')
            assert pixelsha(original)==r['canonical_source_pixels_sha256']
            assert sha(origin_path)==r['canonical_source_png_sha256']
            difference=np.abs(adv.astype(np.int16)-original.astype(np.int16))
            maximum=int(difference.max());assert maximum<=8
            assert maximum==r['max_difference_pixels']==r['selected']['max_difference_pixels']
            assert pixelsha(adv)==r['selected']['pixels_sha256']
            assert sha(adv_path)==r['output_sha256']==r['selected']['sha256']
            expected_mse=float(np.mean(difference.astype(np.float64)**2))
            if expected_mse:
                expected_psnr=10*math.log10(255**2/expected_mse)
                assert abs(expected_psnr-r['psnr'])<1e-10
            else:assert r['psnr'] is None and r['psnr_infinite']
            for h in r['history']:check_metrics(h,views)
            for stage_name,stage in r['stages'].items():
                chosen=best(stage['history']); selected=stage['selected']
                for key in ('pixels_sha256','step','restart','phase','lpips','worst_view_margin','success'):
                    assert selected[key]==chosen[key],(method,iid,stage_name,'candidate selection',key)
                gradients=stage['gradient_steps'];assert stage['actual_gradient_steps']==len(gradients)
                for g in gradients:
                    assert g['stage']==stage_name and g['step']==g['score_step']+1
                    assert min(g['grad_l2'],g['grad_linf'],g['update_l2'],g['update_linf'])>=0
                if stage_name=='latent':
                    assert [g['step'] for g in gradients]==list(range(1,len(gradients)+1))
                    assert len(gradients)<=(0 if method=='Z' else cfg['lao_steps'])
                    assert len(stage['refinement_history']) in (0,5)
                    observations=len(stage['history'])+1
                else:
                    assert len(stage['stop_reasons'])==cfg['restarts']
                    for restart,stop in enumerate(stage['stop_reasons']):
                        relevant=[g for g in gradients if g['restart']==restart]
                        assert [g['step'] for g in relevant]==list(range(1,len(relevant)+1))
                        assert len(relevant)==stop['step']<=cfg['max_steps']
                        probes=[h for h in stage['refinement_history'] if h['restart']==restart]
                        assert len(probes)==(5 if stop['reason'].startswith('first_feasible') else 0)
                    observations=len(stage['history'])
                assert stage['compute']['observations']==observations
                assert stage['compute']['E_forwards']==observations*len(views)
                assert stage['compute']['LPIPS_forwards']==observations
                assert stage['compute']['JPEG_encodes']==observations*(len(views)-1)
            ngl=len(r['latent_gradient_steps']);ngp=len(r['pixel_gradient_steps'])
            assert ngl==r['actual_latent_gradient_steps']==r['compute']['latent_gradient_steps']
            assert ngp==r['actual_pixel_gradient_steps']==r['compute']['pixel_gradient_steps']
            assert ngl+ngp==r['actual_gradient_steps']==len(r['gradient_steps'])
            for key in ('E_forwards','E_view_examples','LPIPS_forwards','JPEG_encodes','observations'):
                total=sum(stage['compute'][key] for stage in r['stages'].values())+r['compute']['verification'][key]
                assert r['compute'][key]==total,(method,iid,'compute',key)
            assert r['compute']['verification']['observations']==2
            assert r['compute']['detector_forwards']==r['compute']['E_forwards']
            counts['outputs']+=1;counts['max_budget']=max(counts['max_budget'],maximum)
            counts['latent_updates']+=ngl;counts['pixel_updates']+=ngp
            counts['refinement_probes']+=len(r['refinement_history'])
            counts['nonzero_latent_updates']+=sum(g['update_l2']>0 for g in r['latent_gradient_steps'])
            counts['nonzero_pixel_updates']+=sum(g['update_l2']>0 for g in r['pixel_gradient_steps'])
            for key in ('E_forwards','renderer_forwards','JPEG_encodes'):counts[key]+=r['compute'][key]
            if method in ('Z','L','H','L_EOT','H_EOT'):
                reset=folder/f'{iid}_reset_image.png';zero=root/'methods/Z'/f'{iid}_adv_image.png'
                assert sha(reset)==sha(zero)==r['reset']['sha256'],(method,iid,'zero reset')
                assert pixelsha(pixels(reset))==r['reset']['pixels_sha256']
        durations=[r['seconds'] for r in rows]
        counts['seconds']={'median':statistics.median(durations),'mean':statistics.mean(durations),'min':min(durations),'max':max(durations)}
        counts['selected_phase_distribution']=dict(Counter(r['selected']['phase'] for r in rows))
        counts['selected_step_zero']=sum(r['selected']['step']==0 for r in rows)
        counts['records_sha256']=sha(folder/'records.jsonl');counts['run_config_sha256']=sha(folder/'run_config.json')
        result['methods'][method]=counts
    for method in ('H','H_EOT'):
        latent='L_EOT' if method.endswith('_EOT') else 'L'
        counters={'lao_stage_matches_independent_L':0,'final_unchanged_from_L':0,'final_equals_original':0,'L_success_preserved':0,'L_failure_pixel_stage_success':0,'L_success_pixel_stage_failure':0,'zero_pixel_gradient_but_output_changed_from_L':0}
        for iid,r in records[method].items():
            folder=root/'methods'/method;lp=root/'methods'/latent
            stage=folder/f'{iid}_lao_image.png';adv=folder/f'{iid}_adv_image.png'
            assert sha(stage)==sha(lp/f'{iid}_adv_image.png')==r['lao_stage']['sha256']
            assert r['lao_stage']['pixels_sha256']==records[latent][iid]['selected']['pixels_sha256']
            counters['lao_stage_matches_independent_L']+=1
            same=sha(adv)==sha(stage)
            counters['final_unchanged_from_L']+=same
            counters['final_equals_original']+=sha(adv)==sha(folder/f'{iid}_originImage.png')
            ls,hs=r['lao_stage']['success'],r['selected']['success']
            counters['L_success_preserved']+=ls and hs
            counters['L_failure_pixel_stage_success']+=(not ls) and hs
            counters['L_success_pixel_stage_failure']+=ls and not hs
            counters['zero_pixel_gradient_but_output_changed_from_L']+=r['actual_pixel_gradient_steps']==0 and not same
        result['stage_comparison'][method]=counters
    result['limitations']=['Recorded finite gradients establish logged execution, not an independent recomputation of autograd.','Stage success counts use all input images, including originally benign predictions; confirmatory ASR requires the predeclared common clean-detected cohort.','H stage relations do not establish LAO necessity or compute-matched superiority.','No detector or LPIPS model is executed by this audit.']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--n',type=int,default=50);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit(a.root.resolve(),a.n)
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'output':str(a.out),'methods':list(result['methods'])}))
