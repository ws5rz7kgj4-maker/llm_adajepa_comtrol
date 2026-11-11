import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from .spec import CONFIG,ARMS,MODEL_SEEDS,scenes,fingerprint
from .env import RobotEnv,SkillPolicy,skill_action
from .model import WorldModel,VisualEncoder
from .core import Trigger,CognitiveBus
from .proposer import Proposer,rule_proposer
from .storage import Store,save_json,sha

def episode(scene,arm,model,encoder,proposer,seed,root,store):
    model.reset(); model.temporal=arm.temporal; env=RobotEnv(scene['task'],scene['seed'],scene['condition']); policy=SkillPolicy(env.initial_z)
    rng=np.random.default_rng(scene['seed']+seed*100000); trigger=Trigger(arm.trigger,model.threshold)
    bus=CognitiveBus(rule_proposer if arm.proposer=='rule' else proposer)
    ident=f"{scene['id']}__seed{seed}__{arm.name}"
    history=[]; decisions=[]; transitions=[]; frames=[]; requests=[]; request_images=[]; latents=[]; command=None; command_until=-1
    started=time.monotonic(); start_utc=time.time(); deadline=started;last_chosen=None;skill_start_step=0; misses=0; wait_steps=0; invalid=0; stale=0; tokens=0; llm_s=0.
    predicted=[]; gains=[]; success=False; violation=False; segment_start=env.distance(); pending_at_end=False
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    try:
        s=env.state();z=model.encode(encoder(env.image()),s);latents.append(z.detach().cpu().numpy())
        for step in range(CONFIG['steps']):
            tick=time.monotonic(); phase=policy.skill(s)
            if command and step>=command_until: command=None
            delivered=bus.poll(ident,step,s[3:6])
            if delivered is not None:
                result=delivered['result']; tokens+=result['tokens'];llm_s+=result['seconds']
                invalid+=int(result['proposal'] is None); stale+=int(delivered['stale'])
                delivered['applied']=False
                if not delivered['stale'] and result['proposal'] is not None:
                    p=result['proposal']
                    if arm.compare:
                        command,delivered['candidate_scores']=model.select_skill(z,s,p['skills'],arm.risk)
                    else: command=p['skills'][p['preferred']]
                    command_until=step+20;delivered['selected']=command;delivered['applied']=True
                decisions.append(dict(step=step,**delivered))
            if step%CONFIG['segment']==0 and trigger.ready(step,history,bus.future is not None):
                req=dict(episode=ident,step=step,sent=time.monotonic(),observed_sim_seconds=step/CONFIG['hz'],
                    predictor_update_version=model.updates,task=scene['task'],
                    image=env.image().copy(),state=s.tolist(),current_skill=command or phase,
                    history=history[-4:] if arm.history else [],
                    temporal=({'sim_seconds':step/CONFIG['hz'],'wall_seconds':time.monotonic()-started,
                    'state_velocity':(model.velocity*model.scale[512:520]).tolist(),
                    'current_skill_elapsed_seconds':(step-skill_start_step)/CONFIG['hz']} if arm.temporal else None))
                requests.append({k:v for k,v in req.items() if k!='image'});request_images.append(req['image'])
                bus.submit(req);trigger.record(step)
            pending=bus.future is not None
            if pending and (arm.execution=='sync' or time.monotonic()-bus.request['sent']>CONFIG['timeout_seconds']):
                a=env.hold();wait_steps+=1;chosen='hold';score=None
            else:
                chosen=command or phase
                if arm.name=='skill_only': a,score=skill_action(chosen,s),None
                else: a,score=model.plan(z,s,chosen,rng,arm.risk)
            if chosen!=last_chosen:skill_start_step=step;last_chosen=chosen
            executed_monotonic=time.monotonic()
            ok,violation=env.step(a);success=bool(ok)
            ns=env.state();nz=model.encode(encoder(env.image()),ns);latents.append(nz.detach().cpu().numpy())
            at=torch.as_tensor(a,device=model.device)
            # One optimizer step at segment boundaries, using only earlier transitions.
            err,gain=model.observe(z,at,nz,arm.adapt and (step+1)%CONFIG['segment']==0,arm.trust_update)
            predicted.append(err)
            if (step+1)%CONFIG['segment']==0:
                distance=env.distance()
                history.append(dict(step=step+1,sim_seconds=(step+1)/CONFIG['hz'],wall_seconds=time.monotonic()-started,progress=segment_start-distance,
                    prediction_error=float(np.mean(predicted)),audit_gain=gain,skill=chosen))
                predicted=[];segment_start=distance
            transitions.append(dict(step=step,observed_sim_seconds=step/CONFIG['hz'],executed_monotonic=executed_monotonic,
                physical_dt_seconds=1/CONFIG['hz'],state=s.tolist(),next_state=ns.tolist(),action=a.tolist(),
                skill=chosen,prediction_error=err,audit_gain=gain,pending=pending,cost=score,
                seconds=time.monotonic()-tick,success=success,violation=violation))
            if step%20==0:frames.append(env.image())
            z=nz;s=ns
            if CONFIG['realtime']:
                deadline+=1/CONFIG['hz']; remaining=deadline-time.monotonic()
                if remaining>0: time.sleep(remaining)
                else: misses+=1
            if success or violation: break
        pending_at_end=bus.future is not None
        # Do not apply late decisions to the next episode.
        if pending_at_end:
            bus.future.result(timeout=120)
            late=bus.poll(ident,env.step_count,s[3:6]);result=late['result']
            tokens+=result['tokens'];llm_s+=result['seconds'];invalid+=int(result['proposal'] is None)
            late['applied']=False;late['episode_ended']=True;decisions.append(dict(step=env.step_count,**late))
        elapsed=time.monotonic()-started
        row=dict(id=ident,scene=scene,model_seed=seed,method=arm.name,protocol=fingerprint(),
            success=success,started_unix_seconds=start_utc,steps=env.step_count,capped_steps=env.step_count if success else CONFIG['steps'],
            wall_seconds=elapsed,sim_seconds=env.step_count/CONFIG['hz'],proposal_calls=trigger.calls,llm_calls=trigger.calls if arm.proposer=='vlm' else 0,
            llm_seconds=llm_s,tokens=int(tokens),invalid_calls=invalid,stale_calls=stale,
            applied_calls=sum(d.get('applied',False) for d in decisions),waiting_steps=wait_steps,
            deadline_misses=misses,violation=violation,final_distance_m=env.distance(),
            adaptation_updates=model.updates,rolled_back_updates=model.rollbacks,
            perturbation_occurred=env.perturbed,pending_at_end=pending_at_end,
            gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        folder=Path(root)/'episodes';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(ident+'.npz')
        np.savez_compressed(path,transitions=json.dumps(transitions,allow_nan=False),
            feedback=json.dumps(history,allow_nan=False),decisions=json.dumps(decisions,allow_nan=False),
            frames=np.array(frames),latents=np.array(latents),requests=json.dumps(requests,allow_nan=False),
            request_images=np.array(request_images),row=json.dumps(row,allow_nan=False))
        receipt=store.push(path)
        store.record('rows/'+ident+'.json',dict(**row,trajectory_receipt=receipt))
        print('EPISODE_COMPLETE',ident,success,flush=True)
        return row
    finally: bus.close();env.close()

def run(root,worker,workers,store):
    encoder=VisualEncoder(); proposer=Proposer(); models={}
    work=[(seed,s) for seed in MODEL_SEEDS for s in scenes()]
    for index,(seed,scene) in enumerate(work):
        if index%workers!=worker: continue
        if seed not in models:models[seed]=WorldModel(Path(root)/'models'/f'model-{seed}.pt')
        arms=ARMS[index%len(ARMS):]+ARMS[:index%len(ARMS)]
        for arm in arms:
            ident=f"{scene['id']}__seed{seed}__{arm.name}"
            row_path=Path(root)/'rows'/(ident+'.json')
            if row_path.exists():
                row=json.loads(row_path.read_text());assert row['protocol']==fingerprint()
                assert sha(Path(root)/'episodes'/(ident+'.npz'))==row['trajectory_receipt']['sha256']
                continue
            episode(scene,arm,models[seed],encoder,proposer,seed,root,store)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--repo');p.add_argument('--run')
    a=p.parse_args();torch.set_num_threads(2);run(a.root,a.worker,a.workers,Store(a.root,a.repo,a.run))
if __name__=='__main__':main()
