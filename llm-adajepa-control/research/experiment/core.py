import json, math, time
from concurrent.futures import ThreadPoolExecutor
from .spec import CONFIG

SKILLS=('approach','descend','grasp','lift','carry','place','release','regrasp','retreat')

def parse_proposal(raw):
    """Allow prose/fences around ONE object; reject malformed/range-invalid decisions."""
    dec=json.JSONDecoder(); found=[]
    for i,ch in enumerate(raw):
        if ch!='{': continue
        try:
            obj,end=dec.raw_decode(raw[i:])
        except ValueError: continue
        if isinstance(obj,dict) and 'skills' in obj: found.append(obj)
    if len(found)!=1: raise ValueError('Expected exactly one proposal object')
    x=found[0]
    if set(x)!= {'skills','preferred'}: raise ValueError('Unexpected schema')
    if not isinstance(x['skills'],list) or len(x['skills'])!=3: raise ValueError('Three skills required')
    if any(s not in SKILLS for s in x['skills']): raise ValueError('Unknown skill')
    if type(x['preferred']) is not int or not 0<=x['preferred']<3: raise ValueError('Bad preferred index')
    return x

class Trigger:
    def __init__(self,mode,error_threshold):
        self.mode=mode; self.error_threshold=error_threshold; self.calls=0; self.last=-999
    def ready(self,step,history,pending=False):
        if pending or self.mode=='none' or self.calls>=CONFIG['max_calls']: return False
        if step==0: return True
        if step-self.last<CONFIG['cooldown']: return False
        if self.mode=='fixed': return step in CONFIG['fixed_steps']
        if len(history)<4: return False
        h=history[-4:]
        stagnant=sum(x['progress'] for x in h)<CONFIG['progress_threshold_m']
        high=sum(x['prediction_error'] for x in h)/len(h)>self.error_threshold
        gains=[x['audit_gain'] for x in h if x['audit_gain'] is not None]
        ineffective=len(gains)>=3 and sum(gains)/len(gains)<=CONFIG['efficacy_threshold']
        if self.mode=='error': return high
        if self.mode=='stagnation': return stagnant
        if self.mode=='joint': return stagnant and (ineffective or not high)
        raise ValueError(self.mode)
    def record(self,step): self.calls+=1; self.last=step

class CognitiveBus:
    """One immutable request at a time. Only main thread touches physics/model weights."""
    def __init__(self,proposer):
        self.pool=ThreadPoolExecutor(max_workers=1); self.proposer=proposer
        self.future=None; self.request=None
    def submit(self,request):
        if self.future is not None: raise RuntimeError('Duplicate pending request')
        self.request=request; self.future=self.pool.submit(self.proposer, request)
    def poll(self,episode,step,object_xyz):
        if self.future is None or not self.future.done(): return None
        request=self.request; result=self.future.result()
        self.future=None; self.request=None
        age=step-request['step']
        drift=math.sqrt(sum((a-b)**2 for a,b in zip(object_xyz,request['state'][3:6])))
        stale=(episode!=request['episode'] or age>CONFIG['max_response_age_steps'] or
               drift>CONFIG['max_object_drift_m'] or time.monotonic()-request['sent']>CONFIG['timeout_seconds'])
        return dict(result=result, stale=stale, age_steps=age, drift_m=drift)
    def close(self):
        # A running CUDA operation cannot be forcibly killed safely by a thread.
        # Episode results record pending work; process supervisor is final timeout.
        try:
            if self.future is not None: self.future.result(timeout=120)
        finally:
            self.pool.shutdown(wait=False,cancel_futures=True)
