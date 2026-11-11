"""JEPA-style frozen image embeddings + observed state; three residual predictors.
This is a NEW state-assisted adaptation, NOT the official AdaJEPA checkpoint.
"""
import copy
from collections import deque
import numpy as np
import torch
from torch import nn
from .spec import CONFIG
from .env import skill_action,target

class VisualEncoder:
    def __init__(self,device='cuda'):
        from torchvision.models import resnet18,ResNet18_Weights
        self.device=device
        m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc=nn.Identity()
        self.model=m.to(device).eval().requires_grad_(False)
    @torch.inference_mode()
    def __call__(self,image):
        x=torch.as_tensor(image.copy(),device=self.device).permute(2,0,1).float()[None]/255
        x=torch.nn.functional.interpolate(x,(224,224),mode='bilinear',align_corners=False)
        x=(x-torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]
        return self.model(x)[0].cpu().numpy()

class Predictor(nn.Module):
    def __init__(self):
        super().__init__(); d=524
        self.net=nn.Sequential(nn.Linear(d+4+8,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,d))
    def forward(self,z,a):
        if a.shape[-1]==4: a=torch.cat([a,torch.zeros((*a.shape[:-1],8),device=a.device)],-1)
        delta=self.net(torch.cat([z,a],dim=-1))
        # task and goal are static observations, never learned dynamics.
        delta=delta*torch.cat([torch.ones(520,device=z.device),torch.zeros(4,device=z.device)])
        return z+delta

def loss(pred,truth):
    return ((pred[...,:512]-truth[...,:512])**2).mean(-1)+((pred[...,512:520]-truth[...,512:520])**2).mean(-1)

class WorldModel:
    def __init__(self,checkpoint,device='cuda'):
        self.device=device
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        self.models=nn.ModuleList([Predictor() for _ in ck['models']]).to(device)
        for m,w in zip(self.models,ck['models']): m.load_state_dict(w)
        self.mean=ck['mean'].to(device); self.scale=ck['scale'].to(device)
        self.threshold=float(ck['error_threshold']); self.base=copy.deepcopy(self.models.state_dict())
        self.reset()
    def reset(self):
        self.models.load_state_dict(self.base); self.models.eval()
        self.optim=torch.optim.Adam(self.models.parameters(),lr=CONFIG['online_lr'])
        self.velocity=torch.zeros(8,device=self.device);self.temporal=False;self.replay=deque(maxlen=CONFIG['replay']); self.updates=0; self.rollbacks=0
    def encode(self,v,s): return (torch.as_tensor(np.r_[v,s],device=self.device)-self.mean)/self.scale
    def state(self,z): return (z*self.scale+self.mean)[...,512:]
    @torch.no_grad()
    def predict(self,z,a): return torch.stack([m(z,a) for m in self.models])
    def observe(self,z,a,nz,adapt,trust=False):
        a=torch.cat([a,self.velocity if self.temporal else torch.zeros_like(self.velocity)])
        self.replay.append((z.detach().clone(),a.detach().clone(),nz.detach().clone()))
        self.velocity=(nz[512:520]-z[512:520]).detach()*CONFIG['hz']
        err=float(loss(self.predict(z,a).mean(0),nz))
        if len(self.replay)<CONFIG['train_min']+CONFIG['audit']: return err,None
        data=list(self.replay); train=data[:-CONFIG['audit']]; audit=data[-CONFIG['audit']:]
        tr=[torch.stack([r[i] for r in train]) for i in range(3)]
        au=[torch.stack([r[i] for r in audit]) for i in range(3)]
        with torch.no_grad(): before=float(loss(self.predict(au[0],au[1]).mean(0),au[2]).mean())
        if not adapt: return err,0.
        previous=copy.deepcopy(self.models.state_dict()) if trust else None
        opt_previous=copy.deepcopy(self.optim.state_dict()) if trust else None
        self.optim.zero_grad()
        objective=sum(loss(m(tr[0],tr[1]),tr[2]).mean() for m in self.models)/len(self.models)
        if trust:
            anchor=sum((p-self.base[n]).square().mean() for n,p in self.models.named_parameters())
            objective=objective+CONFIG['anchor_weight']*anchor
        objective.backward(); nn.utils.clip_grad_norm_(self.models.parameters(),1.)
        self.optim.step(); self.updates+=1
        with torch.no_grad(): after=float(loss(self.predict(au[0],au[1]).mean(0),au[2]).mean())
        gain=(before-after)/max(before,1e-8)
        if trust and (not np.isfinite(after) or after>before):
            self.models.load_state_dict(previous); self.optim.load_state_dict(opt_previous); self.rollbacks+=1
        return err,float(gain)
    @torch.no_grad()
    def score(self,z,sequences,goal_xyz,risk):
        # All arms evaluate all three ensemble members, so risk ablation has equal rollout count.
        a=torch.as_tensor(sequences,device=self.device,dtype=torch.float32)
        costs=[]
        goal=torch.as_tensor(goal_xyz,device=self.device,dtype=torch.float32)
        for m in self.models:
            zz=z.repeat(a.shape[0],1)
            velocity=self.velocity.repeat(a.shape[0],1) if self.temporal else torch.zeros(a.shape[0],8,device=z.device)
            for h in range(a.shape[1]):
                nxt=m(zz,torch.cat([a[:,h],velocity],-1))
                velocity=(nxt[:,512:520]-zz[:,512:520])*CONFIG['hz'] if self.temporal else torch.zeros_like(velocity)
                zz=nxt
            s=self.state(zz)
            costs.append(((s[:,:3]-goal)**2).sum(-1)+.001*(a**2).mean((1,2)))
        c=torch.stack(costs)
        return (c.mean(0)+(CONFIG['risk_weight']*c.std(0,unbiased=False) if risk else 0)).cpu().numpy()
    def plan(self,z,s,skill,rng,risk=False):
        base=skill_action(skill,s); n=CONFIG['candidates']; h=CONFIG['horizon']
        actions=np.tile(base,(n,h,1)); actions[1:,:,:3]+=rng.normal(0,.12,(n-1,h,3))
        actions=np.clip(actions,-1,1).astype('float32')
        cost=self.score(z,actions,target(skill,s)[0],risk)
        return actions[np.argmin(cost),0],float(np.min(cost))
    def select_skill(self,z,s,skills,risk=False):
        # Common cost across candidates. Stage-aware goal is supplied by the common controller.
        from .env import target
        sequences=np.array([np.tile(skill_action(k,s),(CONFIG['horizon'],1)) for k in skills])
        if np.linalg.norm(s[:3]-s[3:6])>.035: common=s[3:6]+[0,0,.02]
        elif s[-1]<.5: common=s[8:11]
        else: common=np.r_[s[8:10],max(s[2],1.10)]
        scores=self.score(z,sequences,common,risk)
        return skills[int(np.argmin(scores))],scores.tolist()
