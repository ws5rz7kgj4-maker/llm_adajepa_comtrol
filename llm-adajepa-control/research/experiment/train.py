import argparse,json,os,random,tempfile
from pathlib import Path
import numpy as np
import torch
from .spec import CONFIG,scenes,MODEL_SEEDS,fingerprint
from .env import RobotEnv,SkillPolicy,skill_action
from .model import VisualEncoder,Predictor,loss
from .storage import Store,save_json,sha

def collection_scenes(worker=0,workers=1):
    if not isinstance(workers,int) or not isinstance(worker,int) or workers<1 or not 0<=worker<workers:
        raise ValueError('Invalid collection worker assignment')
    ordered=scenes('train')+scenes('val')
    return [scene for index,scene in enumerate(ordered) if index%workers==worker]

def validate_data_file(path,scene):
    """Reject partial, non-finite, wrong-scene or wrong-shape training episodes."""
    try:
        with np.load(path,allow_pickle=False) as d:
            if not {'z','a','y','success','scene'}.issubset(d.files):
                raise ValueError('Missing trajectory fields')
            z,a,y=d['z'],d['a'],d['y']
            if (a.ndim!=2 or a.shape[1]!=4 or not 1<=len(a)<=CONFIG['steps']
                or z.shape!=(len(a),524) or y.shape!=z.shape):
                raise ValueError('Invalid trajectory shapes or step count')
            if any(not np.issubdtype(v.dtype,np.number) or not np.isfinite(v).all() for v in (z,a,y)):
                raise ValueError('Non-finite or nonnumeric trajectory')
            if d['scene'].shape!=() or json.loads(str(d['scene'].item()))!=scene:
                raise ValueError('Trajectory scene identity mismatch')
            success=d['success']
            if success.shape!=() or success.dtype.kind!='b':
                raise ValueError('Invalid trajectory success flag')
            return dict(steps=len(a),success=bool(success.item()),scene=scene)
    except (OSError,KeyError,TypeError,ValueError) as exc:
        raise ValueError('Invalid data episode '+str(path)+': '+str(exc)) from exc

def save_data_episode(path,scene,z,a,y,success):
    """Publish a complete NPZ atomically so upload never observes a partial file."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,z=np.array(z,dtype='float32'),a=np.array(a,dtype='float32'),
                y=np.array(y,dtype='float32'),success=bool(success),scene=json.dumps(scene))
            stream.flush();os.fsync(stream.fileno())
        validate_data_file(temp,scene)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def collect(root,store,worker=0,workers=1):
    assigned=collection_scenes(worker,workers)
    encoder=VisualEncoder(); root=Path(root)
    for scene in assigned:
        path=root/'data'/f"{scene['id']}.npz"
        if path.exists():
            validate_data_file(path,scene)
            store.push(path)
            continue
        env=RobotEnv(scene['task'],scene['seed']); policy=SkillPolicy(env.initial_z)
        rng=np.random.default_rng(scene['seed']); zz=[]; aa=[]; yy=[]; success=False
        try:
            z=np.r_[encoder(env.image()),env.state()]
            for step in range(CONFIG['steps']):
                a=skill_action(policy.skill(env.state()),env.state())
                # Broad but bounded behavior coverage; no OOD test conditions here.
                a[:3]=np.clip(a[:3]+rng.normal(0,.08,3),-1,1)
                if step%40==35: a[3]=-1
                ok,violation=env.step(a); nz=np.r_[encoder(env.image()),env.state()]
                zz.append(z);aa.append(a);yy.append(nz);z=nz;success|=ok
                if violation or ok: break
            save_data_episode(path,scene,zz,aa,yy,success)
            store.push(path)
            print('DATA_EPISODE',scene['id'],len(aa),flush=True)
        finally: env.close()

def check_data(root,store):
    """Run only after all collection workers finish; validates every planned file."""
    root=Path(root);validated={scene['id']:validate_data_file(root/'data'/f"{scene['id']}.npz",scene)
        for scene in collection_scenes()}
    quality={}
    for task in CONFIG['tasks']:
        outcomes=[]
        for scene in scenes('train'):
            if scene['task']==task:
                outcomes.append(validated[scene['id']]['success'])
        quality[task]={'episodes':len(outcomes),'successes':sum(outcomes)}
    store.record('data_quality.json',quality)
    if any(x['successes']==0 for x in quality.values()):raise RuntimeError('Training data lacks successful examples; formal test blocked')
    return quality

def arrays(root,split):
    batches=[]
    for scene in scenes(split):
        with np.load(Path(root)/'data'/f"{scene['id']}.npz") as d:
            z,a,y=[d[k] for k in ('z','a','y')]
            vel=np.zeros((len(z),8),dtype='float32');vel[1:]=(z[1:,512:520]-z[:-1,512:520])*CONFIG['hz']/.1
            batches.append([z,np.concatenate([a,vel],-1),y])
    return [torch.from_numpy(np.concatenate([d[i] for d in batches])).float() for i in range(3)]

def train_seed(root,seed,store,device="cuda"):
    root=Path(root); path=root/'models'/f'model-{seed}.pt'
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['protocol']==fingerprint() and ck['validation']['passed']
        return
    tr=arrays(root,'train'); va=arrays(root,'val')
    mean=tr[0].mean(0); scale=tr[0].std(0).clamp_min(.05)
    # Meter scales fixed to make small physical displacements meaningful.
    scale[512:520]=.1; scale[520:523]=1.;scale[523]=1.
    for data in (tr,va): data[0]=(data[0]-mean)/scale;data[2]=(data[2]-mean)/scale
    va_gpu=[a.to(device) for a in va]; states=[]; logs=[]
    gen=torch.Generator().manual_seed(seed)
    for member in range(CONFIG['ensemble_members']):
        torch.manual_seed(seed*101+member);m=Predictor().to(device); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
        best=float('inf'); weights=None
        for epoch in range(CONFIG['train_epochs']):
            # Bootstrap members for epistemic disagreement; common data across outer seeds.
            order=torch.randint(len(tr[0]),(len(tr[0]),),generator=gen)
            for idx in order.split(CONFIG['batch_size']):
                z,a,y=[t[idx].to(device) for t in tr]; opt.zero_grad()
                # Shared model trained to support both temporal and no-temporal arms.
                a=a.clone();a[:,4:]*=(torch.rand(len(a),1,device=device)>.5)
                value=loss(m(z,a),y).mean();value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                opt.step()
            with torch.no_grad():
                errs=torch.cat([loss(m(z,a),y).cpu() for z,a,y in zip(*[x.split(512) for x in va_gpu])])
                v=float(errs.mean())
            logs.append(dict(seed=seed,member=member,epoch=epoch,val_loss=v))
            print('TRAIN',logs[-1],flush=True)
            if v<best: best=v;weights={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        states.append(weights)
    models=[Predictor().to(device) for _ in states]
    for m,w in zip(models,states): m.load_state_dict(w)
    with torch.no_grad():
        errors=[]; physical=[]; no_time_errors=[];no_time_physical=[]
        for z,a,y in zip(*[x.split(512) for x in va_gpu]):
            p=torch.stack([m(z,a) for m in models]).mean(0)
            errors.append(loss(p,y).cpu());physical.append(((p[:,512:520]-y[:,512:520])**2).mean(-1).cpu())
            aa=a.clone();aa[:,4:]=0;pn=torch.stack([m(z,aa) for m in models]).mean(0)
            no_time_errors.append(loss(pn,y).cpu());no_time_physical.append(((pn[:,512:520]-y[:,512:520])**2).mean(-1).cpu())
        errors=torch.cat(errors);physical=torch.cat(physical)
        no_time_errors=torch.cat(no_time_errors);no_time_physical=torch.cat(no_time_physical)
        persistence=float(loss(va[0],va[2]).mean()); physical_persistence=float(((va[0][:,512:520]-va[2][:,512:520])**2).mean())
    validation=dict(persistence_loss=persistence,model_loss=float(errors.mean()),
        physical_persistence=physical_persistence,physical_model_loss=float(physical.mean()),no_time_model_loss=float(no_time_errors.mean()),no_time_physical_loss=float(no_time_physical.mean()))
    validation['passed']=bool(np.isfinite(float(errors.mean())) and errors.mean()<persistence*.95 and physical.mean()<physical_persistence*.95 and no_time_errors.mean()<persistence*.95 and no_time_physical.mean()<physical_persistence*.95)
    store.record(f'models/validation-{seed}.json',dict(validation=validation,epochs=logs))
    if not validation['passed']: raise RuntimeError('World model did not beat persistence on validation; formal test blocked')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(models=states,mean=mean,scale=scale,error_threshold=float(torch.quantile(torch.cat([errors,no_time_errors]),.95)),
        seed=seed,protocol=fingerprint(),validation=validation,
        data_hashes={p.name:sha(p) for p in sorted((root/'data').glob('*.npz'))}),path)
    store.push(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['collect','train','check-data'],required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--repo');p.add_argument('--run');a=p.parse_args();torch.set_num_threads(2)
    st=Store(a.root,a.repo,a.run)
    if a.phase=='collect': collect(a.root,st,a.worker,a.workers)
    elif a.phase=='check-data':check_data(a.root,st)
    else: train_seed(a.root,a.seed,st)
if __name__=='__main__':main()
