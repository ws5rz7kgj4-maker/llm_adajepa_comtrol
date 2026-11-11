import json,time
import numpy as np
import pytest
from experiment.spec import CONFIG,ARMS,MODEL_SEEDS,scenes
from experiment.core import parse_proposal,Trigger,CognitiveBus

@pytest.mark.parametrize('raw',[
 '{"skills":["lift","grasp","retreat"],"preferred":0}',
 '```json\n{"skills":["lift","grasp","retreat"],"preferred":2}\n```',
 'Answer: {"skills":["lift","grasp","retreat"],"preferred":1}'])
def test_parse(raw):assert parse_proposal(raw)['skills'][0]=='lift'

@pytest.mark.parametrize('raw',[
 '{"skills":["fly","grasp","retreat"],"preferred":0}',
 '{"skills":["lift"],"preferred":0}',
 '{"skills":["lift","grasp","retreat"],"preferred":true}',
 '{"skills":["lift","grasp","retreat"],"preferred":3}',
 '{"skills":["lift","grasp","retreat"],"preferred":0,"torque":999}',
 'not json'])
def test_reject(raw):
    with pytest.raises(ValueError):parse_proposal(raw)

def test_manifest():
    splits=[{x['seed'] for x in scenes(s)} for s in ['train','val','test']]
    assert all(not(splits[i]&splits[j]) for i in range(3) for j in range(i))
    assert len(scenes())==200 and len(ARMS)==21 and len(MODEL_SEEDS)==3
    assert len({a.name for a in ARMS})==21

def hist(progress=0,error=2,gain=0):
    return [dict(progress=progress,prediction_error=error,audit_gain=gain)]*4

def test_trigger_distinguishes_adapting_from_stuck():
    t=Trigger('joint',1);assert t.ready(0,[]);t.record(0)
    assert not t.ready(30,hist(gain=.5))
    assert t.ready(30,hist(gain=0))
    assert t.ready(30,hist(error=.1,gain=.5)) # accurate yet ineffective strategy
    assert not t.ready(30,hist(progress=.1))
    assert not t.ready(30,hist(),True)

def test_call_budget_and_cooldown():
    t=Trigger('joint',1)
    for i in range(4):
        assert t.ready(i*30,hist());t.record(i*30)
        assert not t.ready(i*30+1,hist())
    assert not t.ready(200,hist())

def test_bus_delivers_when_low_level_is_holding():
    bus=CognitiveBus(lambda req: {'proposal':{'skills':['lift']*3,'preferred':0}})
    req=dict(episode='a',step=0,state=[0]*12,sent=time.monotonic())
    bus.submit(req);bus.future.result(timeout=1)
    r=bus.poll('a',2,[0,0,0]);assert r is not None and not r['stale'];assert bus.future is None;bus.close()

@pytest.mark.parametrize('episode,step,xyz',[('b',0,[0,0,0]),('a',61,[0,0,0]),('a',1,[1,0,0])])
def test_stale_rejected(episode,step,xyz):
    b=CognitiveBus(lambda req:{'proposal':None});b.submit(dict(episode='a',step=0,state=[0]*12,sent=time.monotonic()));b.future.result(timeout=1)
    assert b.poll(episode,step,xyz)['stale'];b.close()

def test_thread_error_propagates():
    def bad(req):raise RuntimeError('backend failure')
    b=CognitiveBus(bad);b.submit({})
    with pytest.raises(RuntimeError):b.future.result(timeout=1)
    with pytest.raises(RuntimeError):b.close()
