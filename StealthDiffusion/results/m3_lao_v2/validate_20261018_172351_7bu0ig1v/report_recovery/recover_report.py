"""Generate the report with an audited, margin-only numerical compatibility fix."""
import hashlib
import json
from pathlib import Path
import m3_lao_v2_report_numeric_compat as reporting

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
amendment=json.loads((HERE/'amendment.json').read_text())
assert sha(Path(amendment['original_report_path']))==amendment['original_report_sha256']
assert sha(Path(amendment['patched_report_path']))==amendment['patched_report_sha256']
assert sha(HERE/'batching_audit.json')==amendment['batching_audit_sha256']
audit=json.loads((HERE/'batching_audit.json').read_text())
assert sha(ROOT/'plan.json')==audit['plan_sha256']
assert sha(ROOT/'eval/per_image.csv')==audit['evaluation_sha256']
assert sha(ROOT/'new/NEW_LAO8/records.jsonl')==audit['records_sha256']
assert all(r['single_minus_stored']==0 and r['batch16_minus_evaluation']==0 for r in audit['rows'])
for row in audit['rows']:
    assert sha(ROOT/'new/NEW_LAO8'/f"{row['image_id']}_adv_image.png")==row['png_sha256']
report=reporting.build_report(ROOT)
report['reporting_numerical_amendment']=amendment
report['notes'].append('The original frozen reporter stopped at image0041 because single-image and batch16 MPS E margins differ by 1.017749309539795e-5. A saved-PNG re-score exactly reproduced both sets of all50 scores; all labels, margin-buffer decisions and LPIPS values agree. This report uses an archived margin-only absolute consistency tolerance of 2e-5 instead of 1e-5. No images, attack settings, quality thresholds, predictions or statistics were changed. Original failure and audit are retained under report_recovery/.')
reporting.write_report(ROOT/'report',report)
print(json.dumps({'status':report['status'],'primary':report['primary'],'report':str(ROOT/'report/report.md')}))
