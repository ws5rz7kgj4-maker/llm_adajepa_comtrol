"""Real MuJoCo physics via robosuite. State-assisted, not a vision-only claim."""
import numpy as np
from .spec import CONFIG

class RobotEnv:
    def __init__(self,task,seed,condition='nominal',camera=True):
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config
        np.random.seed(seed)
        config=load_composite_controller_config(robot='Panda')
        # World aligned deltas; zero rotation delta preserves the initial orientation.
        config['body_parts']['right']['input_ref_frame']='world'
        self.task=task; self.condition=condition; self.step_count=0
        self.env=suite.make(task,robots='Panda',controller_configs=config,
            has_renderer=False,has_offscreen_renderer=camera,use_camera_obs=camera,
            camera_names='agentview',camera_heights=128,camera_widths=128,
            use_object_obs=True,control_freq=CONFIG['hz'],horizon=CONFIG['steps'],
            ignore_done=True,reward_shaping=False,hard_reset=False)
        self.obs=self.env.reset(); self.last_grip=-1.
        if task=='Lift':
            self.body=self.env.cube_body_id; self.obj=self.env.cube
            self.key='cube_pos'; self.goal=np.array(self.obs[self.key])+[0,0,.16]
        else:
            self.obj=next(o for o in self.env.objects if o.name.lower()=='can')
            self.body=self.env.obj_body_id[self.obj.name]; self.key=self.obj.name+'_pos'
            self.goal=self.env.target_bin_placements[self.env.object_id].copy()+[0,0,.04]
        self.initial_z=float(self.obs[self.key][2]); self.camera=camera
        self.geom_ids=[self.env.sim.model.geom_name2id(n) for n in self.obj.contact_geoms]
        self.perturbed=False
        if condition=='heavy':
            self.env.sim.model.body_mass[self.body]*=4
            self.env.sim.model.body_inertia[self.body]*=4
            self.perturbed=True
        elif condition=='slippery':
            self.env.sim.model.geom_friction[self.geom_ids,0]*=.15; self.perturbed=True
        elif condition not in ('nominal','impulse'): raise ValueError(condition)
        self.env.sim.forward()
    def state(self):
        # eef xyz, observed object xyz, gripper joint positions, goal xyz, task bit.
        return np.r_[self.obs['robot0_eef_pos'], self.obs[self.key],
                     self.obs['robot0_gripper_qpos'][:2],self.goal,float(self.task!='Lift')].astype('float32')
    def image(self): return np.ascontiguousarray(self.obs['agentview_image'][::-1,:,:3])
    def hold(self): return np.array([0,0,0,self.last_grip],dtype='float32')
    def step(self,a):
        a=np.asarray(a,dtype='float32'); assert a.shape==(4,) and np.isfinite(a).all()
        a=np.clip(a,-1,1); self.last_grip=float(a[3])
        # A reproducible, finite external force at the same simulation step for every arm.
        force=self.condition=='impulse' and 50<=self.step_count<55
        self.env.sim.data.xfrc_applied[self.body,:]=0
        if force: self.env.sim.data.xfrc_applied[self.body,0]=2.; self.perturbed=True
        self.obs,_,_,_=self.env.step(np.r_[a[:3],np.zeros(3),a[3]])
        self.step_count+=1
        s=self.state()
        # This is a workspace monitor, NOT a certified collision/safety controller.
        violation=bool(not np.isfinite(s).all() or s[2]<.72 or s[2]>1.5 or np.linalg.norm(s[:2])>.85)
        return bool(self.env._check_success()),violation
    def distance(self): return float(np.linalg.norm(self.state()[3:6]-self.goal))
    def close(self): self.env.close()

class SkillPolicy:
    def __init__(self,initial_z):
        self.initial_z=initial_z; self.phase='approach'; self.dwell=0
    def skill(self,s):
        e,o=s[:3],s[3:6]; d=np.linalg.norm(e-o)
        if self.phase=='approach' and np.linalg.norm(e[:2]-o[:2])<.012: self.phase='descend'
        if self.phase=='descend' and d<.018: self.phase='grasp'; self.dwell=0
        if self.phase=='grasp':
            self.dwell+=1
            if self.dwell>=12: self.phase='lift'
        if self.phase=='lift' and o[2]>self.initial_z+.13 and s[-1]>.5: self.phase='carry'
        if self.phase=='carry' and np.linalg.norm(o[:2]-s[8:10])<.025: self.phase='place'
        if self.phase=='place' and o[2]<s[10]+.025: self.phase='release';self.dwell=0
        if self.phase=='release':
            self.dwell+=1
            if self.dwell>=12:self.phase='retreat'
        if self.phase in ('lift','carry') and d>.10 and o[2]<self.initial_z+.04: self.phase='approach'
        return self.phase

def target(skill,s):
    o=s[3:6]; g=s[8:11]
    offsets={'approach':np.array([0,0,.10]),'descend':np.zeros(3),
             'grasp':np.zeros(3),'regrasp':np.array([.015,0,.025]),'retreat':np.array([0,0,.18])}
    if skill in offsets: xyz=o+offsets[skill]
    elif skill=='lift': xyz=np.r_[o[:2],max(g[2],o[2]+.07)] if s[-1]<.5 else np.r_[o[:2],1.12]
    elif skill=='carry': xyz=np.r_[g[:2],1.12]
    elif skill=='place': xyz=g+[0,0,.015]
    elif skill=='release': xyz=s[:3]
    else: raise ValueError(skill)
    grip=1. if skill in ('grasp','lift','carry','place') else -1.
    return xyz,grip

def skill_action(skill,s):
    xyz,grip=target(skill,s)
    return np.r_[np.clip((xyz-s[:3])*12.,-.6,.6),grip].astype('float32')
