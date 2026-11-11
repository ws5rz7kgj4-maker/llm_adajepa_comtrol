import json,time
import torch
from .spec import VLM_ID,VLM_REV
from .core import SKILLS,parse_proposal

class Proposer:
    def __init__(self,device='cuda'):
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.device=device
        self.processor=AutoProcessor.from_pretrained(VLM_ID,revision=VLM_REV,trust_remote_code=False,min_pixels=224*224,max_pixels=224*224)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_ID,revision=VLM_REV,trust_remote_code=False,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    def __call__(self,req):
        from PIL import Image
        payload={k:req[k] for k in ('task','state','history','current_skill')}
        if req.get('temporal') is not None: payload['temporal_context']=req['temporal']
        prompt='You control a Panda robot with fixed wrist orientation. Coordinates are meters in world frame. State order: end-effector xyz, object xyz, two gripper joint positions, goal xyz, task bit (0 lift, 1 pick-place). Propose 3 next skills to recover/progress. Available: '+', '.join(SKILLS)+'. Skills: approach above object; descend to object; grasp close; lift; carry above target; place lower; release open; regrasp open and offset; retreat up. Use observation and feedback, do not infer hidden dynamics with certainty. Return exactly {"skills":["approach","descend","grasp"],"preferred":0} with your choices. No other keys.\n'+json.dumps(payload,allow_nan=False)
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        image=Image.fromarray(req['image']).convert('RGB').resize((224,224))
        inputs=self.processor(text=[text],images=[image],return_tensors='pt').to(self.device)
        started=time.monotonic()
        with torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=192,do_sample=False,use_cache=True)
        ids=out[0,inputs['input_ids'].shape[-1]:]
        raw=self.processor.decode(ids,skip_special_tokens=True).strip()
        result=dict(raw=raw,seconds=time.monotonic()-req.get('sent',started),tokens=len(ids),proposal=None,invalid=None,
            requested_monotonic=req.get('sent'),inference_started_monotonic=started,returned_monotonic=time.monotonic(),
            observed_step=req.get('step'))
        try: result['proposal']=parse_proposal(raw)
        except (ValueError,TypeError) as exc: result['invalid']=str(exc)
        return result

def rule_proposer(req):
    current=req['current_skill']; skills=[current,'regrasp','retreat']
    if current in ('approach','descend'): skills=[current,'descend','approach']
    return dict(raw='rule',seconds=0.,tokens=0,invalid=None,proposal={'skills':skills,'preferred':0})
