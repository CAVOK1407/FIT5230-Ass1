"""Re-score unchanged saved PNGs only; no optimization or model training."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT.parents[2]
sys.path.insert(0, str(HERE))
import m2_evaluate as evaluation

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

plan = json.loads((ROOT / 'plan.json').read_text())
for name, expected in {**plan['source_sha256'], **plan['checkpoint_sha256']}.items():
    assert sha(Path(name)) == expected, name
evaluation.m2_device.enable('mps')
detector = evaluation.other_attacks.model_selection('E').eval()
detector.requires_grad_(False)
folder = ROOT / 'new/NEW_LAO8'
items, _ = evaluation.load_run(str(folder))
single_pred, single_margin = evaluation.predict_batch(detector, [r['adv'] for r in items], batch_size=1)
batch_pred, batch_margin = evaluation.predict_batch(detector, [r['adv'] for r in items], batch_size=16)
saved = {r['image_id']: r for r in (json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines())}
with (ROOT/'eval/per_image.csv').open() as f:
    evaluated = {r['image_id']:r for r in csv.DictReader(f)
                 if r['run']=='NEW_LAO8' and r['detector']=='E' and r['condition']=='none'}
rows=[]
for i,item in enumerate(items):
    image_id=item['image_id']; record=saved[image_id]['selected']['saved_png']; row=evaluated[image_id]
    assert single_pred[i] == batch_pred[i] == record['pred'] == int(row['adv_pred'])
    assert abs(single_margin[i]-record['real_margin']) <= 1e-5
    assert abs(batch_margin[i]-float(row['adv_margin'])) <= 1e-5
    assert (single_margin[i]>=.1)==(batch_margin[i]>=.1)
    assert sha(folder/(image_id+'_adv_image.png'))==record['sha256']
    rows.append({'image_id':image_id,'png_sha256':record['sha256'],'single_margin':single_margin[i],
                 'batch16_margin':batch_margin[i],'stored_margin':record['real_margin'],
                 'evaluation_margin':float(row['adv_margin']),'prediction':single_pred[i],
                 'single_minus_stored':single_margin[i]-record['real_margin'],
                 'batch16_minus_evaluation':batch_margin[i]-float(row['adv_margin']),
                 'batch_abs_difference':abs(single_margin[i]-batch_margin[i])})
audit={'status':'verified','created_at':datetime.now().astimezone().isoformat(),
       'purpose':'Investigate numerical batching difference; saved PNGs only, no attack rerun',
       'n':len(rows),'device':'mps','script_sha256':sha(Path(__file__)),
       'plan_sha256':sha(ROOT/'plan.json'),'evaluation_sha256':sha(ROOT/'eval/per_image.csv'),
       'records_sha256':sha(folder/'records.jsonl'),
       'max_batch_abs_difference':max(r['batch_abs_difference'] for r in rows),
       'classification_and_margin_buffer_decisions_unchanged':True,'rows':rows}
out=ROOT/'report_recovery/batching_audit.json'
with out.open('x') as f: json.dump(audit,f,indent=2)
print(json.dumps({k:v for k,v in audit.items() if k!='rows'}))
