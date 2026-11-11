import json,time,os
from pathlib import Path
import torch,numpy as np
from .env import RobotEnv,SkillPolicy,skill_action
from .model import VisualEncoder
from .proposer import Proposer
from .spec import CONFIG,fingerprint
from .storage import sha

def check(root,store):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for formal run')
    report=dict(protocol=fingerprint(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda,render_backend=os.environ.get('MUJOCO_GL'),checks=[])
    encoder=VisualEncoder();vlm=Proposer()
    for i,task in enumerate(CONFIG['tasks']):
        env=RobotEnv(task,90000+i)
        try:
            before=env.state();im=env.image();feat=encoder(im)
            assert feat.shape==(512,) and np.isfinite(feat).all()
            # Engineering sentinel: never imported into the test manifest.
            for _ in range(3):env.step(np.array([0,0,.2,-1]))
            assert np.isfinite(env.state()).all()
            req=dict(task=task,state=env.state().tolist(),image=env.image(),history=[],current_skill='approach')
            r=vlm(req)
            if r['proposal'] is None:
                store.record('preflight_failure.json',dict(task=task,raw=r['raw'],invalid=r['invalid']))
                raise RuntimeError('Real VLM proposal failed schema check; formal evaluation blocked')
            report['checks'].append(dict(task=task,visual_shape=list(im.shape),state_dim=len(before),
                vlm_proposal=r['proposal'],raw=r['raw'],vlm_seconds=r['seconds']))
        finally:env.close()
    import importlib.metadata
    report['versions']={k:importlib.metadata.version(k) for k in ['robosuite','mujoco','numpy','torchvision','transformers','huggingface_hub']}
    store.record('preflight.json',report)
