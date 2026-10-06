"""Recompute requested VSM replacement, repetition and runtime ablation comparisons."""
from pathlib import Path
import sys, json, gzip, hashlib, tomllib, re
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts import evaluate as ev
from scripts import compute_requested_statistical_tests as st
from scripts.trad_methods.vsm_variable_cutoff import is_test_path
OUT=Path(__file__).parent
IDS=sorted(tomllib.loads((ROOT/'dataset/trace_completed_instance_validated.toml').read_text())['instance_ids'])
assert len(IDS)==226
TRUTH={i:t for i,t in ev.load_ground_truth_map(ROOT/'dataset/swebench_feature_with_truth.jsonl').items() if i in IDS}
G=ev.GRANULARITIES
RID=[i for i in IDS if all(TRUTH[i]['total'][g] for g in G)]
assert len(RID)==170
models=['Qwen3.5-4B','Qwen3.5-9B','Qwen3.5-27B','GPT-5.4','DeepSeek-v4-Flash','DeepSeek-v4-Pro','DeepSeek-v4-Pro-32K']
old_names=['Qwen_Qwen3.5-4B','Qwen_Qwen3.5-9B','Qwen_Qwen3.5-27B','GPT_5-4','DeepSeek-v4-flash','DeepSeek-v4-pro','DS-v4-pro_32k_think_full']
vsm_names=['Qwen35-4B-runpod-run1','Qwen35-9B-runpod-run1','Qwen35-27B-runpod-run1','gpt54-0305-openai','DeepSeek-v4-flash-0423-openrouter-alibaba','DeepSeek-v4-pro-0423-openrouter-alibaba-run2','DeepSeek-v4-pro-0423-32K-openrouter-alibaba']
new_bnames=['Qwen_Qwen3.5-4B','Qwen35-9B-runpod-run1','Qwen_Qwen3.5-27B','GPT_5-4','DeepSeek-v4-flash','DeepSeek-v4-pro-0423-openrouter-alibaba','DeepSeek-v4-pro-0423-32K-openrouter-baseten']
sources={}; reports={}; folders={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(label,p):
    d=ev.load_run_report(p)
    records={r['instance_id']:r for r in d['instances']}
    assert set(records)==set(IDS)
    for i in IDS:assert records[i]['ground_truth']=={c:{g:sorted(TRUTH[i][c][g]) for g in G} for c in ev.CHANGE_TYPES}
    sources[label]={'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
    reports[label]=d;folders[label]=p.parent
    return records

def point(a,ids=IDS):
    rr=[a[i] for i in ids];return ev.summarize_records(rr)
def values(a,metric,ids):
    return np.array([a[i]['metrics']['avg3']['ap' if metric=='map' else metric] for i in ids],float)
def boot(x,key):
    r=st.boot(np.asarray(x,float),key);return {**r['files'],'seed':r['seed']}
def effect(a,b,key,ids=IDS,tests=False):
    ri=[i for i in ids if i in RID]
    out={'n':len(ids),'ranking_n':len(ri),'a':point(a,ids),'b':point(b,ids)}
    for metric,mid in [('f1',ids),('map',ri)]:
        delta=values(a,metric,mid)-values(b,metric,mid)
        out[metric]=boot(delta,key+':'+metric)
        if tests:
            p,seed=st.flip(delta,key+':'+metric);out[metric].update(raw_p=p,test_seed=seed)
        out[metric]['lopo']={project:float(np.mean([d for d,i in zip(delta,mid) if i.split('-')[0]!=project])) for project in sorted({i.split('-')[0] for i in mid})}
    dv=np.array([int(a[i]['valid'])-int(b[i]['valid']) for i in ids]);out['completion']=boot(dv,key+':completion')
    out['completion']['candidate_only']=int((dv==1).sum());out['completion']['baseline_only']=int((dv==-1).sum())
    if tests:out['completion']['raw_p']=st.mcnemar(int((dv==-1).sum()),int((dv==1).sum()))
    return out

def paired(a,b,key):
    common=[i for i in IDS if a[i]['valid'] and b[i]['valid']]
    return {'all':effect(a,b,key+':all',tests=True),'common':effect(a,b,key+':common',common)}
def correct(family):
    for metric in ['completion','f1','map']:
        pp=st.holm({k:r['all'][metric]['raw_p'] for k,r in family.items()})
        for k,r in family.items():r['all'][metric]['holm_p']=pp[k]

D={};B={};NB={};OLD={}
for model,old,new,bnew in zip(models,old_names,vsm_names,new_bnames):
    D[model]=load('vsm:'+model,ROOT/'dataset/agent_results/dalic_vsm'/new/'evaluation.json')
    B[model]=load('paper_baseline:'+model,ROOT/'dataset/results_archive/agent_results/baseline'/old/'evaluation.json')
    NB[model]=load('current_baseline:'+model,ROOT/'dataset/agent_results/baseline'/bnew/'evaluation.json')
    oldfolder=ROOT/'dataset/results_archive/agent_results/dalic'/old
    evals,_=ev.load_instance_evaluations(oldfolder,TRUTH,ROOT/'dataset/agent_workbench/dalic')
    OLD[model]={e.instance_id:ev.instance_record(e) for e in evals}
    sources['bm25:'+model]={'path':str(oldfolder.relative_to(ROOT)),'records_sha256':ev.record_fingerprints(list(OLD[model].values()))['records_sha256']}
    print('Loaded',model,flush=True)

result={'metadata':{'instances':226,'avg3_map_instances':170,'bootstrap_draws':st.BD,'sign_flip_draws':st.FD,'holm':'Separate families per analysis and outcome: 7 models for baseline comparisons; 3 Qwen models for ablation; 3 repetitions for repeated-run comparisons.','matched_invalid_k':3,'primary_baseline':'Original manuscript archived baseline; current baseline is an alternative complete setup, including run1 for 9B.','status':'ANALYZED; no model experiment rerun','sources':sources},'replacement':{},'current_baseline':{},'bm25_to_vsm':{},'retrieval':{},'repetitions':{},'ablation':{}}
for model in models:
    result['replacement'][model]=paired(D[model],B[model],'replace:'+model)
    result['current_baseline'][model]=paired(D[model],NB[model],'current:'+model)
    result['bm25_to_vsm'][model]=effect(D[model],OLD[model],'bm25_to_vsm:'+model)
    print('Compared agents',model,flush=True)
correct(result['replacement']);correct(result['current_baseline'])

ranks={}
for method in ['static_vsm','static_lsi','static_bm25','trace_vsm','trace_bm25']:
    ranks[method]={}
    for i in IDS:
        p=ROOT/'dataset/trad_methods_results/rankings'/method/(i+'.json.gz')
        d=json.load(gzip.open(p,'rt'));assert d['failure_reason'] is None
        ranks[method][i]=d['rankings']


# Use actual workbench order for the Trace VSM to agent comparison.
# Keep the original ranking cache as a separate sensitivity comparator.
ranks['trace_vsm_cached']=ranks['trace_vsm']
ranks['trace_vsm']={}
rank_mismatches=[]
for i in IDS:
    ranks['trace_vsm'][i]={}
    for g,f in zip(G,['file','class','function']):
        rr=json.loads((ROOT/'dataset/agent_workbench/dalic_vsm'/i/'dynamic_evidence'/(f+'.json')).read_text())
        names=[row['qualified_name'] for row in rr if not is_test_path(row['qualified_name'].split('::')[0])]
        cached=ranks['trace_vsm_cached'][i][g]
        assert set(names)==set(cached)
        ranks['trace_vsm'][i][g]=names
        if names!=cached:
            rank_mismatches.append({'instance_id':i,'granularity':g,'top3_differs':names[:3]!=cached[:3]})
result['ranking_cache_audit']={'definition':'trace_vsm uses actual workbench order excluding test entities; trace_vsm_cached uses saved ranking cache.','set_mismatches':0,'order_mismatches':rank_mismatches}

def retrieval(method,k,match=None):
    records={}
    for i in IDS:
        pred={g:ranks[method][i][g][:len(match[i]['predictions'][g]) if match is not None and match[i]['valid'] else k] for g in G}
        e=ev.InstanceEvaluation(i,{g:set(pred[g]) for g in G},TRUTH[i],False,None,pred,None)
        records[i]=ev.instance_record(e)
    return records
fixed={(method,k):retrieval(method,k) for method in ranks for k in (3,5,10,20)}
result['retrieval']['static_trace_vsm']={str(k):effect(fixed['trace_vsm',k],fixed['static_vsm',k],'static_trace_vsm:'+str(k)) for k in (3,5,10,20)}
result['retrieval']['static_trace_vsm_cached']={str(k):effect(fixed['trace_vsm_cached',k],fixed['static_vsm',k],'static_trace_vsm_cached:'+str(k)) for k in (3,5,10,20)}
result['retrieval']['dalic_vs_ir']={}
for model in models:
    one={}
    for method in ['trace_vsm','trace_vsm_cached','static_vsm','static_lsi','static_bm25']:
        one[method]={'fixed_k3':effect(D[model],fixed[method,3],model+':'+method+':fixed'),
                     'matched':effect(D[model],retrieval(method,3,D[model]),model+':'+method+':matched')}
    result['retrieval']['dalic_vs_ir'][model]=one
    print('Compared retrieval',model,flush=True)

DB=[];BB=[]
for run in range(1,4):
    a=load('repeat_vsm:'+str(run),ROOT/f'dataset/agent_results/dalic_vsm/Qwen35-9B-runpod-run{run}/evaluation.json')
    b=load('repeat_baseline:'+str(run),ROOT/f'dataset/agent_results/baseline/Qwen35-9B-runpod-run{run}/evaluation.json')
    DB.append(a);BB.append(b)
    result['repetitions'][str(run)]=paired(a,b,'repeat:'+str(run))
correct(result['repetitions'])
result['repetition_summary']={}
for name,seq in [('baseline',BB),('vsm',DB)]:
    points=[point(a) for a in seq]
    vals={'completion':[p['valid_rate'] for p in points],'f1':[p['metrics']['avg3']['f1'] for p in points],'map':[p['metrics']['avg3']['map'] for p in points]}
    result['repetition_summary'][name]={k:{'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)),'min':min(v),'max':max(v)} for k,v in vals.items()}
result['repetition_summary']['nine_combinations']=[]
for i,a in enumerate(DB):
 for j,b in enumerate(BB):
    result['repetition_summary']['nine_combinations'].append({'vsm_run':i+1,'baseline_run':j+1,'completion':point(a)['valid_rate']-point(b)['valid_rate'],'f1':float((values(a,'f1',IDS)-values(b,'f1',IDS)).mean()),'map':float((values(a,'map',RID)-values(b,'map',RID)).mean())})
common6=[i for i in IDS if all(a[i]['valid'] for a in DB+BB)]
result['repetition_summary']['common_all_six']={'n':len(common6),'runs':{str(k+1):effect(a,b,'common6:'+str(k),common6) for k,(a,b) in enumerate(zip(DB,BB))}}
result['repetition_summary']['mean_per_instance_effect']={}
for metric,mid in [('f1',IDS),('map',RID)]:
    delta=np.array([values(a,metric,mid)-values(b,metric,mid) for a,b in zip(DB,BB)]).mean(axis=0)
    result['repetition_summary']['mean_per_instance_effect'][metric]=boot(delta,'mean_repeats:'+metric)

for size in ('4B','9B','27B'):
    model='Qwen3.5-'+size
    n=load('no_runtime:'+model,ROOT/f'dataset/agent_results/dalic_no_runtime/Qwen35-{size}-no-runtime-run1/evaluation.json')
    result['ablation'][model]=paired(D[model],n,'ablation:'+model)
correct(result['ablation'])

# Evidence membership is unaffected by ranking order. Check predictions against actual VSM workbench.
evidence={i:{g:{r['qualified_name'] for r in json.loads((ROOT/'dataset/agent_workbench/dalic_vsm'/i/'dynamic_evidence'/(filename+'.json')).read_text())} for g,filename in zip(G,['file','class','function'])} for i in IDS}
result['runtime_scope']={}
for model,a in D.items():
    counts=Counter()
    for i in IDS:
      if not a[i]['valid']:continue
      for g in G:
        pred=set(a[i]['predictions'][g]);gt=TRUTH[i]['total'][g];inside=pred&evidence[i][g];outside=pred-evidence[i][g]
        counts.update({'pred_inside':len(inside),'pred_outside':len(outside),'correct_inside':len(inside&gt),'correct_outside':len(outside&gt),'missing_gt':len(gt-evidence[i][g])})
    result['runtime_scope'][model]=dict(counts)
result['runtime_scope']['coverage']={'covered':sum(len(TRUTH[i]['total'][g]&evidence[i][g]) for i in IDS for g in G),'total':sum(len(TRUTH[i]['total'][g]) for i in IDS for g in G)}

# Read observed configuration snapshots, never credentials or current environment files.
audit={}
for label,folder in folders.items():
    if label.startswith(('paper_baseline:','current_baseline:')) and label.split(':',1)[1].startswith('Qwen'):continue
    counts=Counter();example={};exits=Counter();failed_exits=Counter();runtime_reads=0;loaded=0
    for i in IDS:
        p=folder/i/'last_mini_run.traj.json'
        if not p.exists():continue
        d=json.loads(p.read_text());info=d.get('info',{});c=info.get('config',{});loaded+=1
        ag=c.get('agent',{});m=c.get('model',{})
        safe={'model_name':m.get('model_name'),'model_type':c.get('model_type'),'model_kwargs':{k:v for k,v in m.get('model_kwargs',{}).items() if k in ('temperature','seed','top_p','top_k','max_tokens','max_completion_tokens','reasoning_effort','extra_body','drop_params','tool_choice')},'context_length':m.get('context_length'),'agent_limits':{k:ag.get(k) for k in ('step_limit','cost_limit','wall_time_limit_seconds','max_consecutive_format_errors')},'prompt_sha256':hashlib.sha256((ag.get('system_template','')+ag.get('instance_template','')).encode()).hexdigest(),'observation_sha256':hashlib.sha256(m.get('observation_template','').encode()).hexdigest(),'mini_version':info.get('mini_version')}
        s=json.dumps(safe,sort_keys=True);h=hashlib.sha256(s.encode()).hexdigest();counts[h]+=1;example[h]=safe
        exits[info.get('exit_status','unknown')]+=1
        if not next(r for r in reports[label]['instances'] if r['instance_id']==i)['valid']:failed_exits[info.get('exit_status','unknown')]+=1
        commands=[]
        for msg in d.get('messages',[]):
            for t in msg.get('tool_calls',[]) or []:
                fun=t.get('function',{});arguments=fun.get('arguments','')
                commands.append(arguments if isinstance(arguments,str) else json.dumps(arguments))
        if any('dynamic_evidence' in cmd and any(s in cmd for s in ('file.json','class.json','function.json')) for cmd in commands):runtime_reads+=1
    audit[label]={'trajectories':loaded,'configurations':[{'count':n,'config':example[h]} for h,n in counts.items()],'exit_status':dict(exits),'invalid_exit_status':dict(failed_exits),'commands_reference_runtime_evidence':runtime_reads}
result['configuration_audit']=audit
result['metadata']['sources']=sources
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('SAVED',OUT/'results.json',flush=True)
