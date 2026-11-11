"""Immutable experimental choices. No thresholds are fitted on test results."""
from dataclasses import dataclass, asdict
import hashlib, json

@dataclass(frozen=True)
class Arm:
    name: str
    trigger: str = 'none'
    adapt: bool = True
    compare: bool = True
    history: bool = True
    proposer: str = 'vlm'
    execution: str = 'sync'
    trust_update: bool = False
    risk: bool = False
    temporal: bool = False

ARMS = [Arm('skill_only', adapt=False), Arm('adaptive_only'), Arm('frozen_only', adapt=False),
    Arm('fixed', 'fixed'), Arm('error', 'error'), Arm('stagnation', 'stagnation'),
    Arm('joint', 'joint'), Arm('joint_frozen', 'joint', adapt=False),
    Arm('fixed_frozen', 'fixed', adapt=False),
    Arm('joint_no_compare', 'joint', compare=False),
    Arm('joint_no_history', 'joint', history=False),
    Arm('joint_rule', 'joint', proposer='rule'),
    Arm('joint_async', 'joint', execution='async'),
    Arm('joint_trust', 'joint', trust_update=True),
    Arm('joint_risk', 'joint', risk=True),
    Arm('enhanced', 'joint', trust_update=True, risk=True, temporal=True),
    Arm('enhanced_async', 'joint', trust_update=True, risk=True, temporal=True, execution='async'),
    Arm('joint_temporal', 'joint', temporal=True),
    Arm('enhanced_no_time', 'joint', trust_update=True, risk=True),
    Arm('enhanced_no_trust', 'joint', risk=True, temporal=True),
    Arm('enhanced_no_risk', 'joint', trust_update=True, temporal=True)]
TASKS = ['Lift', 'PickPlaceCan']
CONDITIONS = ['nominal', 'heavy', 'slippery', 'impulse']
MODEL_SEEDS = [11, 29, 47]
VLM_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
VLM_REV = '66285546d2b821cf421d4f5eb2576359d3770cd3'
CONFIG = dict(version='3d-v1-20261023', tasks=TASKS, conditions=CONDITIONS,
    model_seeds=MODEL_SEEDS, scenes_per_cell=25, train_per_task=120, val_per_task=20,
    steps=240, hz=20, segment=5, horizon=5, candidates=32, train_epochs=30, ensemble_members=3, risk_weight=1.0, anchor_weight=0.001,
    batch_size=256, latent_visual_dim=512, state_dim=12, max_calls=4,
    fixed_steps=[0,60,120,180], cooldown=30, min_window=20,
    progress_threshold_m=0.005, efficacy_threshold=0.02, error_quantile=0.95,
    online_lr=0.0001, replay=80, audit=5, train_min=20,
    max_response_age_steps=60, max_object_drift_m=0.08, timeout_seconds=15,
    image_size=128, realtime=True, observation='rgb+simulator_object_xyz+proprioception',
    arms=[asdict(a) for a in ARMS], vlm_id=VLM_ID, vlm_revision=VLM_REV)

def fingerprint():
    return hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()

def scenes(split='test'):
    out=[]
    n=CONFIG['scenes_per_cell'] if split=='test' else CONFIG['train_per_task' if split=='train' else 'val_per_task']
    offset={'train':10000,'val':20000,'test':30000}[split]
    for t,task in enumerate(TASKS):
        for c,condition in enumerate(CONDITIONS if split=='test' else ['nominal']):
            for i in range(n):
                out.append(dict(id=f'{split}-{task}-{condition}-{i:03}', task=task,
                    condition=condition, seed=offset+1000*t+100*c+i, split=split))
    return out
