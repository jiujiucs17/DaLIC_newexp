"""Check repeated-run infrastructure failures and saved prediction provenance."""
from pathlib import Path
import sys,json,hashlib,tomllib
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts import evaluate as ev
from scripts import compute_requested_statistical_tests as st
OUT=Path(__file__).parent
ids=sorted(tomllib.loads((ROOT/'dataset/trace_completed_instance_validated.toml').read_text())['instance_ids'])
runs={}
for method in ['baseline','dalic_vsm']:
 for run in range(1,4):
  p=ROOT/f'dataset/agent_results/{method}/Qwen35-9B-runpod-run{run}/evaluation.json'
  runs[method,run]={r['instance_id']:r for r in ev.load_run_report(p)['instances']}
excluded=[]
for i in ids:
 p=ROOT/'dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run3'/i/'last_mini_run.traj.json'
 d=json.loads(p.read_text())
 if d['info'].get('exit_status')=='NotFoundError':
  excluded.append({'instance_id':i,'exit_message':d['messages'][-1].get('content')})
excluded_ids={r['instance_id'] for r in excluded}
keep=[i for i in ids if i not in excluded_ids]
out={'definition':'Supplement only: remove the same five API 404 affected instance IDs from all six runs. Main analysis retains all 226 instances.','excluded':excluded,'n':len(keep),'pairs':{}}
for run in range(1,4):
 a,b=runs['dalic_vsm',run],runs['baseline',run]
 d={'vsm':ev.summarize_records([a[i] for i in keep]),'baseline':ev.summarize_records([b[i] for i in keep]),'effects':{}}
 for metric in ['f1','ap']:
  mid=[i for i in keep if metric=='f1' or a[i]['metrics']['avg3']['ranking_eligible']]
  delta=np.array([a[i]['metrics']['avg3'][metric]-b[i]['metrics']['avg3'][metric] for i in mid])
  bb=st.boot(delta,'exclude_api404:'+str(run)+':'+metric)
  d['effects'][metric]={**bb['files'],'seed':bb['seed']}
 out['pairs'][str(run)]=d
(OUT/'repetition_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
print('EXCLUDED',excluded)
for run,d in out['pairs'].items():print(run,'n',d['vsm']['n'],'valid',d['baseline']['valid_n'],d['vsm']['valid_n'],'effects',d['effects'])
# Verify that current raw results still match the evaluation's stored result hash.
checks=0;mismatches=[]
paths=set((ROOT/'dataset/agent_results/dalic_vsm').glob('*/evaluation.json'))|set((ROOT/'dataset/agent_results/dalic_no_runtime').glob('*/evaluation.json'))|set((ROOT/'dataset/agent_results/baseline').glob('*/evaluation.json'))
for p in paths:
 for r in json.loads(p.read_text())['instances']:
  raw=p.parent/r['instance_id']/'results.txt'
  actual=hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None
  if actual!=r.get('result_sha256'):mismatches.append([str(p.relative_to(ROOT)),r['instance_id']])
  checks+=1
out={'checked_records':checks,'mismatches':mismatches}
(OUT/'source_freshness.json').write_text(json.dumps(out,indent=2)+'\n');print('SOURCE_FRESHNESS',out)
