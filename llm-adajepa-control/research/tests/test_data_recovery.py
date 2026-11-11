"""Legacy data audit and replay checks with an immutable fake Hub, no network."""
import copy,hashlib,json,zipfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiment import recovery
from experiment.spec import CONFIG,scenes,fingerprint

RUN='3d-8e34f1a77be6';REPO='test/legacy';REV='a'*40

@pytest.fixture
def hub(tmp_path,monkeypatch):
    current=Path(__file__).resolve().parents[1]
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as z:
        for name in (*recovery.SCIENTIFIC_FILES,'experiment/train.py'):z.writestr(name,(current/name).read_bytes())
    source_sha=recovery.digest(source)
    monkeypatch.setattr(recovery,'LEGACY_TRAIN_SHA',recovery.digest(current/'experiment/train.py'))
    prefix='runs/'+RUN+'/'
    paths={'source/'+source_sha+'.zip':source}
    values={
        'protocol.json':dict(fingerprint=fingerprint(),config=CONFIG,source_sha256=source_sha,
                             **{split:scenes(split) for split in ('train','val','test')}),
        'rendering.json':dict(backend='egl',attempts=[]),
        'cloud_bootstrap_runtime.json':dict(after=dict(cuda_available=True,torch='2.14.1'))}
    for name,value in values.items():
        path=tmp_path/name;path.write_text(json.dumps(value));paths[prefix+name]=path
    for scene in [scenes('train')[0],scenes('val')[0]]:
        path=tmp_path/(scene['id']+'.npz')
        z=np.zeros((2,524),dtype='float32');a=np.zeros((2,4),dtype='float32')
        np.savez_compressed(path,z=z,a=a,y=z.copy(),success=np.array(True),scene=json.dumps(scene))
        paths[prefix+'data/'+path.name]=path
    entries={remote:NS(path=remote,size=p.stat().st_size,lfs={'sha256':recovery.digest(p)})
             for remote,p in paths.items() if '/data/' in remote}
    requests=[]
    class Api:
        def repo_info(self,repo,**kwargs):return NS(sha=REV)
        def list_repo_files(self,repo,**kwargs):
            assert kwargs['revision']==REV;return list(paths)
        def list_repo_tree(self,repo,**kwargs):
            assert kwargs['revision']==REV and kwargs['recursive'];return list(entries.values())
    def download(repo,remote,**kwargs):
        assert repo==REPO and kwargs==dict(repo_type='dataset',revision=REV)
        requests.append(remote);return str(paths[remote])
    def inspect():return recovery.inspect_recovery(REPO,RUN,source_sha,api=Api(),download=download,current_source_root=current)
    return NS(inspect=inspect,download=download,paths=paths,entries=entries,source=source,
              source_sha=source_sha,current=current,requests=requests,prefix=prefix,tmp=tmp_path)

def test_audit_and_restore_only_complete_data_at_fixed_revision(hub):
    manifest=hub.inspect()
    assert manifest['revision']==REV and manifest['recovered_episodes']==2
    assert len(manifest['missing_scene_ids'])==278
    target=hub.tmp/'restored';report=recovery.restore_legacy_data(manifest,target,download=hub.download)
    assert report['recovered_episodes']==2 and report['rendering_backend']=='egl'
    assert len(list((target/'data').glob('*.npz')))==2 and (target/'rendering.json').exists()
    assert not (target/'models').exists() and not (target/'rows').exists()
    assert recovery.restore_legacy_data(manifest,target,download=hub.download)==report

@pytest.mark.parametrize('path',['models/model.pt','rows/episode.json','episodes/one.npz','checkpoints/index.json','COMPLETED.json'])
def test_parent_training_or_evaluation_artifacts_block_migration(hub,path):
    hub.paths[hub.prefix+path]=hub.source
    with pytest.raises(RuntimeError,match='artifacts'):hub.inspect()

def test_hub_lfs_hash_mismatch_rejected(hub):
    next(iter(hub.entries.values())).lfs={'sha256':'f'*64}
    with pytest.raises(RuntimeError,match='LFS checksum'):hub.inspect()

def test_unknown_scene_path_rejected(hub):
    hub.paths[hub.prefix+'data/test-Lift-nominal-000.npz']=hub.source
    with pytest.raises(RuntimeError,match='Unexpected parent data path'):hub.inspect()

def test_source_identity_mismatch_rejected(hub):
    path=hub.paths[hub.prefix+'protocol.json'];obj=json.loads(path.read_text());obj['source_sha256']='f'*64;path.write_text(json.dumps(obj))
    with pytest.raises(RuntimeError,match='identity'):hub.inspect()

def test_source_changes_block_reuse(hub,tmp_path):
    root=tmp_path/'changed'
    for name in (*recovery.SCIENTIFIC_FILES,'experiment/train.py'):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((hub.current/name).read_bytes())
    (root/'experiment/model.py').write_text('changed')
    with pytest.raises(RuntimeError,match='Scientific source changed'):recovery.verify_compatible_source(hub.source,root)

@pytest.mark.parametrize('kind',['nan','wrong_scene','discontinuous','wrong_dtype'])
def test_invalid_npz_rejected(hub,kind):
    scene=scenes('train')[0];z=np.zeros((2,524),dtype='float32');y=z.copy();a=np.zeros((2,4),dtype='float32')
    meta=scene
    if kind=='nan':a[0,0]=np.nan
    elif kind=='wrong_scene':meta=scenes('train')[1]
    elif kind=='discontinuous':y[0,0]=1
    elif kind=='wrong_dtype':z=z.astype('float64')
    path=hub.tmp/'invalid.npz';np.savez_compressed(path,z=z,y=y,a=a,success=True,scene=json.dumps(meta))
    with pytest.raises(RuntimeError):recovery.validate_legacy_npz(path,scene)

def test_restoration_refuses_tampered_download(hub):
    manifest=hub.inspect();entry=manifest['data'][0]
    hub.paths[entry['remote']].write_bytes(b'corrupted')
    with pytest.raises(RuntimeError,match='checksum'):recovery.restore_legacy_data(manifest,hub.tmp/'restored',download=hub.download)

def test_restoration_refuses_traversal_and_duplicate_entries(hub):
    manifest=hub.inspect();bad=copy.deepcopy(manifest);bad['data'][0]['path']='../outside.npz'
    with pytest.raises(RuntimeError,match='path'):recovery.restore_legacy_data(bad,hub.tmp/'restored',download=hub.download)
    bad=copy.deepcopy(manifest);bad['data'].append(bad['data'][0])
    with pytest.raises(RuntimeError,match='duplicate'):recovery.restore_legacy_data(bad,hub.tmp/'restored2',download=hub.download)

def test_missing_metadata_and_network_errors_do_not_look_like_empty_data(hub):
    hub.entries.clear()
    with pytest.raises(RuntimeError,match='metadata'):hub.inspect()
