import json,sys,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from experiment import driver
from experiment.spec import scenes,MODEL_SEEDS,ARMS,fingerprint
from experiment.storage import Store,save_json,sha


def pair(root):
    scene=scenes()[0];seed=MODEL_SEEDS[0];arm=ARMS[0].name
    ident=f"{scene['id']}__seed{seed}__{arm}"
    episode=root/'episodes'/(ident+'.npz');episode.parent.mkdir(parents=True,exist_ok=True);episode.write_bytes(b'complete test trajectory')
    row=root/'rows'/(ident+'.json')
    save_json(row,dict(id=ident,scene=scene,model_seed=seed,method=arm,protocol=fingerprint(),trajectory_receipt={'sha256':sha(episode)}))
    return row,episode


def test_orphan_pair_quarantined_before_reenqueue(tmp_path):
    row,episode=pair(tmp_path);episode.unlink()
    result=driver.validate_episode_pairs(tmp_path,quarantine=True)
    assert result==dict(complete_pairs=0,quarantined_orphans=1)
    assert not row.exists() and (tmp_path/'quarantine'/'rows'/row.name).exists()


def test_orphan_trajectory_quarantined(tmp_path):
    row,episode=pair(tmp_path);row.unlink()
    assert driver.validate_episode_pairs(tmp_path,quarantine=True)['quarantined_orphans']==1
    assert not episode.exists()


def test_complete_pair_requires_matching_checksum(tmp_path):
    row,episode=pair(tmp_path)
    assert driver.validate_episode_pairs(tmp_path)['complete_pairs']==1
    episode.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='checksum'):driver.validate_episode_pairs(tmp_path,quarantine=True)


def test_analysis_cannot_use_orphan_row(tmp_path):
    row,episode=pair(tmp_path);episode.unlink()
    with pytest.raises(RuntimeError,match='analysis blocked'):driver.validate_episode_pairs(tmp_path)


def test_resume_verifies_source_before_copying_artifacts(tmp_path,monkeypatch):
    root=tmp_path/'new';root.mkdir();monkeypatch.setenv('EXPERIMENT_SOURCE_SHA256','a'*64)
    module=types.ModuleType('huggingface_hub');module.HfApi=lambda:NS(repo_info=lambda *a,**k:NS(sha='b'*40))
    monkeypatch.setitem(sys.modules,'huggingface_hub',module)
    def restore(repo,run,revision,dest):
        save_json(dest/'protocol.json',dict(fingerprint=fingerprint(),source_sha256='c'*64))
        (dest/'models').mkdir();(dest/'models'/'bad.pt').write_bytes(b'old')
        return {}
    monkeypatch.setattr(driver,'restore_checkpoints',restore)
    with pytest.raises(RuntimeError,match='source mismatch'):driver.restore(root,'test/repo','3d-test')
    assert not (root/'models').exists()


def test_resume_copy_filters_orphans_and_queue(tmp_path,monkeypatch):
    root=tmp_path/'new';root.mkdir();monkeypatch.setenv('EXPERIMENT_SOURCE_SHA256','a'*64)
    module=types.ModuleType('huggingface_hub');module.HfApi=lambda:NS(repo_info=lambda *a,**k:NS(sha='b'*40))
    monkeypatch.setitem(sys.modules,'huggingface_hub',module)
    def restore(repo,run,revision,dest):
        save_json(dest/'protocol.json',dict(fingerprint=fingerprint(),source_sha256='a'*64))
        save_json(dest/'rendering.json',{'backend':'egl'})
        save_json(dest/'cloud_bootstrap_runtime.json',{'after':{}})
        row,episode=pair(dest);episode.unlink()
        (dest/'.checkpoint_queue').mkdir();(dest/'.checkpoint_queue'/'old.json').write_text('{}')
        return {'packs':1}
    monkeypatch.setattr(driver,'restore_checkpoints',restore)
    result=driver.restore(root,'test/repo','3d-test')
    assert result['pairs']['quarantined_orphans']==1
    assert not list((root/'rows').glob('*.json')) and not (root/'.checkpoint_queue').exists()


def test_runtime_identity_rejects_changed_stack():
    report={'after':dict(python=[3,12],torch='2.14.1',torchvision='0.29.1',cuda='13.0',protected={'torch':'2.14.1'})}
    driver.validate_runtime_identity(report,report)
    changed=json.loads(json.dumps(report));changed['after']['cuda']='12.0'
    with pytest.raises(RuntimeError,match='runtime identity'):driver.validate_runtime_identity(report,changed)


def test_renderer_identity_is_stricter_than_backend():
    report={'backend':'egl','attempts':[{'backend':'egl','ok':True,'result':{'checks':[dict(vendor='NVIDIA',renderer='L4',version='4.6')]}}]}
    driver.validate_renderer_identity(report,report)
    changed=json.loads(json.dumps(report));changed['attempts'][0]['result']['checks'][0]['renderer']='llvmpipe'
    with pytest.raises(RuntimeError,match='renderer identity'):driver.validate_renderer_identity(report,changed)


def test_completion_marker_waits_for_verified_prior_artifacts(tmp_path,monkeypatch):
    store=Store(tmp_path);(tmp_path/'logs').mkdir();calls=[]
    publisher=NS(stats={'commits':2})
    def drain(publisher,root,**kwargs):
        calls.append((root/'COMPLETED.json').exists())
        return True
    monkeypatch.setattr(driver,'drain_checkpoints',drain)
    driver.finalize_success(tmp_path,store,publisher,{'confirmatory_allowed':True})
    assert calls==[False,True]
    assert json.loads((tmp_path/'COMPLETED.json').read_text())['all_prior_artifacts_remotely_verified']


def test_unverified_prior_artifacts_block_completion_marker(tmp_path,monkeypatch):
    store=Store(tmp_path);(tmp_path/'logs').mkdir()
    monkeypatch.setattr(driver,'drain_checkpoints',lambda *a,**k:False)
    with pytest.raises(RuntimeError,match='remain local'):driver.finalize_success(tmp_path,store,NS(stats={}),{})
    assert not (tmp_path/'COMPLETED.json').exists()


def test_non429_upload_error_backs_off_without_losing_queue(tmp_path,monkeypatch):
    clock=[100.];monkeypatch.setattr(driver.time,'time',lambda:clock[0]);calls=[]
    def flush(**kwargs):calls.append(kwargs);raise ConnectionError('offline')
    publisher=NS(flush=flush,pending_count=lambda:3)
    assert driver.checkpoint_tick(publisher,tmp_path)['status']=='error'
    assert driver.checkpoint_tick(publisher,tmp_path)['reason']=='network_backoff'
    assert len(calls)==1
    clock[0]=161.;driver.checkpoint_tick(publisher,tmp_path)
    assert len(calls)==2


def test_final_drain_is_bounded_and_preserves_mutable_log(tmp_path,monkeypatch):
    clock=[100.];monkeypatch.setattr(driver.time,'time',lambda:clock[0]);monkeypatch.setattr(driver.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(driver.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    publisher=NS(flush=lambda **k:{'status':'deferred','reason':'HTTP 429'},has_pending=lambda:True,pending_count=lambda:5,next_attempt_at=3700.,stats={})
    assert not driver.drain_checkpoints(publisher,tmp_path,max_seconds=12)
    assert clock[0]==112 and not (tmp_path/'logs'/'checkpoint_activity.jsonl').exists()


def test_durable_counts_use_pairs_and_unique_paths():
    publisher=NS(state={'packs':[{'files':[{'path':'data/a.npz','sha256':'1'}, {'path':'rows/a.json','sha256':'2'}]},
       {'files':[{'path':'data/a.npz','sha256':'3'}, {'path':'episodes/a.npz','sha256':'4'}, {'path':'rows/b.json','sha256':'5'}]}]})
    assert driver.durable_counts(publisher)==dict(durable_data_episodes=1,durable_episode_pairs=1)


def test_log_snapshot_is_bounded_stable_and_does_not_enqueue_live_file(tmp_path):
    logs=tmp_path/'logs';logs.mkdir();active=logs/'collect-0.log';active.write_bytes(b'0123456789')
    store=Store(tmp_path,'test/repo','3d-000000000000')
    driver.snapshot_logs(tmp_path,store,'collect',limit=4)
    tail=logs/'tails'/'collect-0.tail.log'
    assert tail.read_bytes()==b'6789'
    with active.open('ab') as handle:handle.write(b'next')
    assert tail.read_bytes()==b'6789'
    queued={json.loads(p.read_text())['path'] for p in (tmp_path/'.checkpoint_queue').glob('*.json')}
    assert queued=={'logs/tails/collect-0.tail.log','logs/tails/collect-capture.json'}
    metadata=json.loads((logs/'tails'/'collect-capture.json').read_text())['files'][0]
    assert metadata['source_bytes']==10 and metadata['captured_start_byte']==6
    driver.snapshot_logs(tmp_path,store,'collect',limit=4)
    assert tail.read_bytes()==b'next'
