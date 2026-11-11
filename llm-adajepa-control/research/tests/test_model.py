import copy
import numpy as np
import torch
from experiment.model import Predictor,WorldModel,loss

def make(tmp_path):
    torch.manual_seed(2);models=[Predictor() for _ in range(3)]
    p=tmp_path/'model.pt';torch.save(dict(models=[m.state_dict() for m in models],mean=torch.zeros(524),scale=torch.ones(524),error_threshold=1.),p)
    return WorldModel(p,'cpu')

def test_parameters_update_and_reset(tmp_path):
    m=make(tmp_path);initial=copy.deepcopy(m.models.state_dict())
    for _ in range(26):m.observe(torch.ones(524),torch.ones(4),torch.zeros(524),True)
    assert m.updates==2
    assert any(not torch.equal(initial[k],v) for k,v in m.models.state_dict().items())
    m.reset();assert all(torch.equal(initial[k],v) for k,v in m.models.state_dict().items())

def test_frozen_does_not_update(tmp_path):
    m=make(tmp_path);initial=copy.deepcopy(m.models.state_dict())
    for _ in range(26):err,gain=m.observe(torch.ones(524),torch.zeros(4),torch.zeros(524),False)
    assert m.updates==0 and gain==0
    assert all(torch.equal(initial[k],v) for k,v in m.models.state_dict().items())

def test_audit_excluded_from_gradient(tmp_path):
    m=make(tmp_path);data=[]
    for i in range(25):
        z=torch.full((524,),float(i));a=torch.zeros(4);y=z+.1
        m.observe(z,a,y,False)
    assert [float(r[0][0]) for r in list(m.replay)[:-5]]==list(map(float,range(20)))
    assert [float(r[0][0]) for r in list(m.replay)[-5:]]==list(map(float,range(20,25)))

def test_predictor_keeps_goal_and_task():
    m=Predictor();z=torch.randn(4,524);a=torch.randn(4,4)
    assert torch.equal(m(z,a)[:,-4:],z[:,-4:])

def test_risk_adds_nonnegative_penalty(tmp_path):
    m=make(tmp_path);z=torch.zeros(524);a=np.zeros((4,5,4),dtype='float32')
    plain=m.score(z,a,[0,0,0],False);risk=m.score(z,a,[0,0,0],True)
    assert np.all(risk>=plain)

def test_trust_rejects_worsening_update(tmp_path):
    m=make(tmp_path)
    for _ in range(24):m.observe(torch.ones(524),torch.ones(4),torch.zeros(524),False)
    # Force a huge optimizer jump to verify both weights and optimizer are restored.
    for group in m.optim.param_groups:group['lr']=1e5
    before=copy.deepcopy(m.models.state_dict())
    m.observe(torch.ones(524),torch.ones(4),torch.zeros(524),True,True)
    assert m.rollbacks==1
    assert all(torch.equal(before[k],v) for k,v in m.models.state_dict().items())

def test_temporal_observation_is_causal(tmp_path):
    m=make(tmp_path);m.temporal=True
    z=torch.zeros(524);nz=z.clone();nz[512]=.05
    m.observe(z,torch.zeros(4),nz,False)
    first=list(m.replay)[0]
    assert torch.equal(first[1][4:],torch.zeros(8))
    assert torch.isclose(m.velocity[0],torch.tensor(1.))
    m.observe(nz,torch.zeros(4),nz,False)
    second=list(m.replay)[1]
    assert torch.isclose(second[1][4],torch.tensor(1.))
    assert torch.isclose(m.velocity[0],torch.tensor(0.))

def test_training_validation_gate_stops_untrained_models(tmp_path,monkeypatch):
    import experiment.train as tr
    from experiment.storage import Store
    from experiment.spec import CONFIG
    torch.manual_seed(7)
    z=torch.randn(32,524);a=torch.zeros(32,12);y=z.clone()
    monkeypatch.setattr(tr,'arrays',lambda *args:[z.clone(),a.clone(),y.clone()])
    monkeypatch.setitem(CONFIG,'train_epochs',1)
    import pytest
    with pytest.raises(RuntimeError,match='persistence'):
        tr.train_seed(tmp_path,11,Store(tmp_path),device='cpu')
    assert (tmp_path/'models'/'validation-11.json').exists()
    assert not (tmp_path/'models'/'model-11.pt').exists()
