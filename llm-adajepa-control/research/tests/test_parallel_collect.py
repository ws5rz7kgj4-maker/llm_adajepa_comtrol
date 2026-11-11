import json
from pathlib import Path
import numpy as np
import pytest
from experiment import train


class Store:
    def __init__(self):self.pushed=[];self.records=[]
    def push(self,path):self.pushed.append(Path(path))
    def record(self,name,value):self.records.append((name,value))


def small_scenes(monkeypatch):
    original=train.scenes
    selected={split:[scene for scene in original(split) if scene['id'].endswith('-000')]
              for split in ('train','val')}
    monkeypatch.setattr(train,'scenes',lambda split:selected[split])
    return selected


class FakeEnv:
    def __init__(self,task,seed):
        self.initial_z=.8;self.seed=seed;self.s=np.zeros(12,dtype='float32');self.n=0
        self.s[0]=seed/100000
    def state(self):return self.s.copy()
    def image(self):return np.full((2,2,3),self.seed%251,dtype='uint8')
    def step(self,a):
        self.s[:3]+=a[:3];self.n+=1
        return self.n==3,False
    def close(self):pass


def install_fake_physics(monkeypatch):
    monkeypatch.setattr(train,'RobotEnv',FakeEnv)
    monkeypatch.setattr(train,'VisualEncoder',lambda:lambda image:np.full(512,image[0,0,0]/255,dtype='float32'))
    class Policy:
        def __init__(self,z):pass
        def skill(self,s):return 'approach'
    monkeypatch.setattr(train,'SkillPolicy',Policy)
    monkeypatch.setattr(train,'skill_action',lambda skill,state:np.array([.1,.2,.3,-1.],dtype='float32'))


def test_four_collection_shards_cover_once_in_canonical_order():
    combined=train.scenes('train')+train.scenes('val')
    shards=[train.collection_scenes(i,4) for i in range(4)]
    ids=[scene['id'] for shard in shards for scene in shard]
    assert len(ids)==len(set(ids))==280
    assert set(ids)=={scene['id'] for scene in combined}
    assert all(shard==combined[i::4] for i,shard in enumerate(shards))


@pytest.mark.parametrize('worker,workers',[(0,0),(-1,4),(4,4)])
def test_invalid_shard_rejected(worker,workers):
    with pytest.raises(ValueError):train.collection_scenes(worker,workers)


def test_sharding_preserves_per_scene_rng_and_arrays(tmp_path,monkeypatch):
    small_scenes(monkeypatch);install_fake_physics(monkeypatch)
    sequential=tmp_path/'sequential';parallel=tmp_path/'parallel'
    train.collect(sequential,Store())
    stores=[]
    for worker in range(4):
        store=Store();stores.append(store)
        train.collect(parallel,store,worker,4)
    assert all(not store.records for store in stores)
    for scene in train.collection_scenes():
        name=scene['id']+'.npz'
        with np.load(sequential/'data'/name) as first,np.load(parallel/'data'/name) as second:
            for key in ('z','a','y','success','scene'):np.testing.assert_array_equal(first[key],second[key])
    store=Store();quality=train.check_data(parallel,store)
    assert all(q['episodes']==1 and q['successes']==1 for q in quality.values())
    assert store.records[0][0]=='data_quality.json'


def test_recovered_complete_files_are_validated_requeued_and_not_simulated(tmp_path,monkeypatch):
    small_scenes(monkeypatch);install_fake_physics(monkeypatch)
    train.collect(tmp_path,Store())
    monkeypatch.setattr(train,'RobotEnv',lambda *a:pytest.fail('Existing scene must not be simulated'))
    store=Store();train.collect(tmp_path,store)
    assert len(store.pushed)==len(train.collection_scenes())


@pytest.mark.parametrize('corruption',['wrong_shape','nonfinite','wrong_scene','empty','success_type'])
def test_corrupt_existing_episode_cannot_be_reused(tmp_path,monkeypatch,corruption):
    small_scenes(monkeypatch);install_fake_physics(monkeypatch)
    scene=train.collection_scenes()[0]
    data=dict(z=np.zeros((2,524),dtype='float32'),a=np.zeros((2,4),dtype='float32'),
              y=np.zeros((2,524),dtype='float32'),success=True,scene=json.dumps(scene))
    if corruption=='wrong_shape':data['y']=np.zeros((2,523))
    elif corruption=='nonfinite':data['z'][0,0]=np.nan
    elif corruption=='wrong_scene':data['scene']=json.dumps(dict(scene,seed=-1))
    elif corruption=='empty':data['a']=np.zeros((0,4))
    else:data['success']=1
    path=tmp_path/'data'/(scene['id']+'.npz');path.parent.mkdir();np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='Invalid data episode'):train.collect(tmp_path,Store())


def test_partial_npz_is_not_published(tmp_path,monkeypatch):
    scene=train.collection_scenes()[0];path=tmp_path/(scene['id']+'.npz')
    def fail(stream,**kwargs):stream.write(b'partial');raise OSError('simulated disk failure')
    monkeypatch.setattr(np,'savez_compressed',fail)
    with pytest.raises(OSError):train.save_data_episode(path,scene,[],[],[],False)
    assert not path.exists() and not list(tmp_path.glob('*.tmp'))


def test_quality_check_requires_every_worker_file(tmp_path,monkeypatch):
    small_scenes(monkeypatch);install_fake_physics(monkeypatch)
    train.collect(tmp_path,Store(),0,4)
    with pytest.raises(ValueError,match='Invalid data episode'):train.check_data(tmp_path,Store())
