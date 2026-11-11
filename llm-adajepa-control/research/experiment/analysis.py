"""Paired, scene-clustered inference; model-seed results remain separately visible."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .spec import CONFIG,ARMS,MODEL_SEEDS,scenes,fingerprint
from .storage import save_json

CONTRASTS={
 'joint_vs_fixed':{'joint':1,'fixed':-1},
 'joint_vs_error':{'joint':1,'error':-1},
 'joint_vs_stagnation':{'joint':1,'stagnation':-1},
 'adaptation_interaction':{'joint':1,'fixed':-1,'joint_frozen':-1,'fixed_frozen':1},
 'async_vs_sync':{'joint_async':1,'joint':-1},
 'enhanced_vs_joint':{'enhanced':1,'joint':-1}}
METRICS=['success','capped_steps','wall_seconds','llm_calls','llm_seconds','tokens','invalid_calls',
         'stale_calls','waiting_steps','deadline_misses','violation','final_distance_m','applied_calls']

def analyze(root):
    root=Path(root);rows=[json.loads(p.read_text()) for p in (root/'rows').glob('*.json')]
    expected={(s['id'],seed,a.name) for s in scenes() for seed in MODEL_SEEDS for a in ARMS}
    found={}
    for r in rows:
        key=(r['scene']['id'],r['model_seed'],r['method'])
        if key in found or key not in expected or r['protocol']!=fingerprint():raise ValueError('Mismatched or duplicate rows')
        found[key]=r
    out=root/'analysis';out.mkdir(exist_ok=True)
    status=dict(expected=len(expected),complete=len(found),missing=len(expected-found.keys()),
        confirmatory_allowed=found.keys()==expected,protocol=fingerprint())
    save_json(out/'completeness.json',status)
    if not status['confirmatory_allowed']: return status
    # Treat a scene, with all seeds/methods, as one bootstrap cluster.
    ids=[s['id'] for s in scenes()]; strata=[(t,c) for t in CONFIG['tasks'] for c in CONFIG['conditions']]
    groups=[np.array([i for i,s in enumerate(scenes()) if (s['task'],s['condition'])==g]) for g in strata]
    rng=np.random.default_rng(20261023)
    draws=np.concatenate([rng.choice(g,(10000,len(g)),replace=True) for g in groups],axis=1)
    means=[]; tables={}
    for a in ARMS:
        tables[a.name]={}
        for metric in METRICS:
            values=np.array([[found[(sid,seed,a.name)][metric] for seed in MODEL_SEEDS] for sid in ids],float)
            tables[a.name][metric]=values.mean(1)
            boot=values.mean(1)[draws].mean(1)
            means.append(dict(method=a.name,metric=metric,mean=float(values.mean()),
                low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975)),
                seed_means=json.dumps(values.mean(0).tolist())))
    with (out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
    comparisons={}
    for name,weights in CONTRASTS.items():
        metric='wall_seconds' if name=='async_vs_sync' else 'success'
        v=sum(w*tables[m][metric] for m,w in weights.items());dist=v[draws].mean(1)
        alpha=.05/len(CONTRASTS);lo,hi=np.quantile(dist,[alpha/2,1-alpha/2])
        comparisons[name]=dict(metric=metric,effect=float(v.mean()),ci=[float(lo),float(hi)],
            family_size=len(CONTRASTS),ci_level=1-alpha,nondegenerate=bool(np.std(dist)>0),
            supported=bool(np.std(dist)>0 and (hi<0 if metric=='wall_seconds' else lo>0)))
    # Additional ablations remain exploratory with paired 95% intervals, not extra primary claims.
    exploratory={}
    for a in ARMS:
        if a.name=='joint':continue
        exploratory[a.name]={}
        for metric in ('success','llm_calls','wall_seconds'):
            v=tables[a.name][metric]-tables['joint'][metric];d=v[draws].mean(1)
            exploratory[a.name][metric]=dict(effect=float(v.mean()),ci95=np.quantile(d,[.025,.975]).tolist())
    save_json(out/'contrasts.json',dict(confirmatory=comparisons,exploratory=exploratory,
        inference='Conditional on these three training seeds; scene-clustered stratified bootstrap. No equivalence/noninferiority claim.'))
    # Perturbation-conditional success is descriptive: exposure can depend on the policy.
    save_json(out/'recovery_descriptive.json',{a.name:{'exposed':sum(r['perturbation_occurred'] for r in rows if r['method']==a.name),
        'successful_exposed':sum(r['success'] and r['perturbation_occurred'] for r in rows if r['method']==a.name)} for a in ARMS})
    summary=['# Results (generated from complete paired data)','',
        'State-assisted 3D manipulation. No claims about pure vision, real robots, consciousness, or hard real-time safety.','',
        '| Method | Success | Wall seconds | VLM calls | Invalid calls |','|---|---:|---:|---:|---:|']
    for a in ARMS: summary.append('| '+a.name+' | '+' | '.join(f"{tables[a.name][m].mean():.3f}" for m in ['success','wall_seconds','llm_calls','invalid_calls'])+' |')
    summary+=['','Six prespecified contrasts use Bonferroni-adjusted intervals. All negative and inconclusive results are retained.']
    (out/'RESULTS.md').write_text('\n'.join(summary))
    return status
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(analyze(a.root))
