"""Run the fixed 1000-point Q3 verification grid with resumable result checkpoints.
This wrapper does not change Q3's mathematical model or success criterion.
"""
from pathlib import Path
import hashlib
import io
import json
import sqlite3
import sys
import pandas as pd

ROOT = Path('/workspaces/tpsa-grid-hackathon-2026')
AUDIT = ROOT / 'results/problem_3_1/verification_20260909T154123Z'
sys.path.insert(0, str(ROOT / 'src/problem_3_1'))
import question_3_dynamic_line_rating as q

argv = ['q3', '--max-uplifts', '0.1,0.2,0.3,0.4,0.5,0.6',
        '--battery-mode', 'required', '--battery-scales',
        ','.join(f'{i / 1000:.3f}' for i in range(1, 1001))]
fingerprint = hashlib.sha256()
for relative in ['src/problem_3_1/question_3_dynamic_line_rating.py',
                 'data/participant-kit/gridkit.py',
                 'data/participant-kit/networks/WP2033_north-west.nc',
                 'results/problem_3_1/question_2_final/tables/10_recommendation_summary.csv',
                 'results/problem_3_1/question_2_final/tables/01_target_line_summary.csv']:
    fingerprint.update((ROOT / relative).read_bytes())
fingerprint.update(json.dumps(argv).encode())
signature = fingerprint.hexdigest()
connection = sqlite3.connect(AUDIT / 'q3_scale_checkpoints.sqlite3')
connection.execute('CREATE TABLE IF NOT EXISTS trials (signature TEXT, label TEXT, result TEXT, hourly TEXT, PRIMARY KEY(signature, label))')
connection.commit()
original_run = q.run_scenario

def checkpointed_run(*args, **kwargs):
    label = kwargs.get('label', '')
    if not label.startswith('wind_proxy_DLR_plus_battery_scale_') or kwargs.get('return_network', False):
        return original_run(*args, **kwargs)
    cached = connection.execute('SELECT result, hourly FROM trials WHERE signature=? AND label=?', (signature, label)).fetchone()
    if cached is not None:
        print('[checkpoint] ' + label, flush=True)
        hourly = pd.read_json(io.StringIO(cached[1]), orient='split')
        hourly.index = pd.to_datetime(hourly.index)
        return json.loads(cached[0]), None, hourly
    row, network, hourly = original_run(*args, **kwargs)
    connection.execute('INSERT OR REPLACE INTO trials VALUES (?, ?, ?, ?)',
                       (signature, label, json.dumps(row, default=lambda value: value.item()),
                        hourly.to_json(orient='split', date_format='iso', double_precision=15)))
    connection.commit()
    return row, network, hourly

print('Verification fingerprint: ' + signature, flush=True)
print('Grid: 0.001 to 1.000 inclusive; 1000 points; fixed Q2 sites and MW/MWh ratio.', flush=True)
q.run_scenario = checkpointed_run
sys.argv = argv
try:
    status = q.main()
finally:
    connection.close()
raise SystemExit(status)
