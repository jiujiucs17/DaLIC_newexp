"""Audit saved VSM trajectories for observed effects of the BM25 field mismatch.
Does not execute agent commands or modify experiment outputs/prompts.
"""
from pathlib import Path
from collections import Counter
import sys,json,csv,re,tomllib,hashlib,statistics
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts import evaluate as ev
from scripts.analyze_trajectory_audit import extract_bash_calls
OUT=Path(__file__).parent
BASE=ROOT/'dataset/agent_results/dalic_vsm'
IDS=set(tomllib.loads((ROOT/'dataset/trace_completed_instance_validated.toml').read_text())['instance_ids'])
TRUTH={i:x for i,x in ev.load_ground_truth_map(ROOT/'dataset/swebench_feature_with_truth.jsonl').items() if i in IDS}
KEYERR=re.compile(r'''KeyError:\s*['"]bm25_score['"]''')
GET=re.compile(r'''\.get\s*\(\s*['"]bm25_score['"]''')
SUBSCRIPT=re.compile(r'''\[\s*['"]bm25_score['"]\s*\]''')
JQ=re.compile(r'''\.bm25_score\b''')
rows=[];events=[];all_summary=[];source_files={}
for folder in sorted(p for p in BASE.iterdir() if p.is_dir() and p.name!='logs'):
    files=sorted(p for p in folder.glob('*/last_mini_run.traj.json') if p.parent.name in IDS)
    if not files:continue
    ep=folder/'evaluation.json'
    if ep.exists():
        report=ev.load_run_report(ep);records={r['instance_id']:r for r in report['instances']}
        scope='full_cohort' if set(records)==IDS else 'subset_evaluation'
        source_files[str(ep.relative_to(ROOT))]=hashlib.sha256(ep.read_bytes()).hexdigest()
    else:
        subset={p.parent.name:TRUTH[p.parent.name] for p in files}
        evaluations,_=ev.load_instance_evaluations(folder,subset,ROOT/'dataset/agent_workbench/dalic_vsm')
        records={e.instance_id:ev.instance_record(e) for e in evaluations};scope='test_or_partial'
    group=[]
    for p in files:
        raw=p.read_bytes();d=json.loads(raw);info=d.get('info',{});config=info.get('config',{})
        prompt=config.get('agent',{}).get('instance_template','')
        calls=extract_bash_calls(d);wrong=[c for c in calls if 'bm25_score' in c.command]
        direct=[c for c in calls if KEYERR.search(c.output or '')]
        guarded=[c for c in calls if GET.search(c.command) and 'vsm_score' not in c.command]
        jq=[c for c in calls if JQ.search(c.command) and 'vsm_score' not in c.command]
        silent=[c for c in guarded+jq if c.returncode==0 and not KEYERR.search(c.output or '')]
        rec=records[p.parent.name]
        changed_after=[]
        if direct:
            last=max(c.command_index for c in direct)
            changed_after=[c for c in calls if c.command_index>last and 'vsm_score' in c.command and c.returncode==0 and not KEYERR.search(c.output or '')]
        row={'config':folder.name,'scope':scope,'instance_id':p.parent.name,'valid':rec['valid'],'failure_reason':rec['failure_reason'],'exit_status':info.get('exit_status'),
             'api_calls':info.get('model_stats',{}).get('api_calls'),'step_limit':config.get('agent',{}).get('step_limit'),
             'prompt_has_bm25_score':'bm25_score' in prompt,'prompt_has_vsm_score':'vsm_score' in prompt,
             'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
             'wrong_field_command_n':len(wrong),'explicit_keyerror_command_n':len(direct),
             'explicit_keyerror_turn_n':len({c.model_turn_index for c in direct}),
             'first_keyerror_command_index':min((c.command_index for c in direct),default=None),
             'last_keyerror_command_index':max((c.command_index for c in direct),default=None),
             'subscript_command_n':sum(bool(SUBSCRIPT.search(c.command)) for c in wrong),
             'wrong_only_get_command_n':len(guarded),'wrong_only_jq_command_n':len(jq),
             'potential_silent_command_n':len(silent),'explicit_vsm_read_after_error':bool(changed_after),
             'f1':rec['metrics']['avg3']['f1'],'ap':rec['metrics']['avg3']['ap'],
             'source':str(p.relative_to(ROOT)),'source_sha256':hashlib.sha256(raw).hexdigest()}
        rows.append(row);group.append(row)
        for call in calls:
            kinds=[]
            if KEYERR.search(call.output or ''):kinds.append('explicit_keyerror')
            if call in silent:kinds.append('potential_silent_wrong_field')
            if not kinds and 'bm25_score' not in call.command:continue
            events.append({'config':folder.name,'instance_id':p.parent.name,'valid':rec['valid'],'exit_status':info.get('exit_status'),
                           'kinds':kinds or ['wrong_field_mention'],'command_index':call.command_index,'model_turn_index':call.model_turn_index,'tool_call_id':call.tool_call_id,
                           'returncode':call.returncode,'command':call.command,'output':call.output,'source':str(p.relative_to(ROOT))})
    errors=[r for r in group if r['explicit_keyerror_command_n']]
    clean=[r for r in group if not r['explicit_keyerror_command_n']]
    summary={'config':folder.name,'scope':scope,'n':len(group),'evaluated_n':len(records),'missing_trajectory_ids':sorted(set(records)-{r['instance_id'] for r in group}),
             'valid_n':sum(r['valid'] for r in group),'wrong_prompt_n':sum(r['prompt_has_bm25_score'] for r in group),
             'wrong_field_command_instances':sum(r['wrong_field_command_n']>0 for r in group),
             'explicit_error_instances':len(errors),'explicit_error_commands':sum(r['explicit_keyerror_command_n'] for r in group),
             'explicit_error_turns':sum(r['explicit_keyerror_turn_n'] for r in group),
             'error_then_valid':sum(r['valid'] for r in errors),'error_then_invalid':sum(not r['valid'] for r in errors),
             'error_then_explicit_vsm_read':sum(r['explicit_vsm_read_after_error'] for r in errors),
             'potential_silent_instances':sum(r['potential_silent_command_n']>0 for r in group),
             'error_invalid_exit_status':dict(Counter(r['exit_status'] for r in errors if not r['valid'])),
             'error_total_exit_status':dict(Counter(r['exit_status'] for r in errors)),
             'no_error_invalid_n':sum(not r['valid'] for r in clean),
             'error_valid_rate':sum(r['valid'] for r in errors)/len(errors) if errors else None,
             'no_error_valid_rate':sum(r['valid'] for r in clean)/len(clean) if clean else None,
             'error_api_calls_mean':statistics.mean(r['api_calls'] for r in errors) if errors else None,
             'no_error_api_calls_mean':statistics.mean(r['api_calls'] for r in clean) if clean else None,
             'error_avg3_f1':statistics.mean(r['f1'] for r in errors) if errors else None,
             'no_error_avg3_f1':statistics.mean(r['f1'] for r in clean) if clean else None,
             'max_keyerror_commands_one_trajectory':max((r['explicit_keyerror_command_n'] for r in group),default=0)}
    all_summary.append(summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

# Schema check against the current actual VSM workbench, restricted to eligible IDs.
schema=Counter();schema_files=0
for i in sorted(IDS):
 for g in ('file','class','function'):
  p=ROOT/'dataset/agent_workbench/dalic_vsm'/i/'dynamic_evidence'/(g+'.json')
  data=json.loads(p.read_text());schema_files+=1
  for rec in data:
   schema['records']+=1;schema['with_vsm_score']+=int('vsm_score' in rec);schema['with_bm25_score']+=int('bm25_score' in rec)
meta={'date':'2026-10-05','status':'ANALYZED','schema_files':schema_files,'evidence_schema':dict(schema),'evaluation_sources':source_files,
      'counting':'Counts are instance-run observations, not unique feature instances. Test/partial runs kept separate.',
      'explicit_error_definition':'A matched tool output contains KeyError: bm25_score. Excludes system/user prompt text and commands echoed in docker logs.',
      'potential_silent_definition':'Command uses only bm25_score via .get or jq-style attribute, exits 0 without explicit KeyError; manually review event outputs before concluding actual information loss.',
      'causal_limit':'No corrected-prompt paired reruns are present in this snapshot. Co-occurring invalid outputs and error/no-error score differences do not estimate causal harm.'}
with (OUT/'instances.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
(OUT/'summary.json').write_text(json.dumps({'metadata':meta,'configs':all_summary},ensure_ascii=False,indent=2)+'\n')
print('TOTAL',len(rows),'events',len(events),'schema',dict(schema),flush=True)
