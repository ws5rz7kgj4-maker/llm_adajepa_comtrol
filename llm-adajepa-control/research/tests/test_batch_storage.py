"""Local checkpoint packs + mocked HF commits; no network or paid jobs."""
import hashlib,json,sys,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from experiment import storage as st

class RateLimited(Exception):
    def __init__(self,headers=None):self.response=NS(status_code=429,headers=headers or {})

@pytest.fixture
def hub(tmp_path,monkeypatch):
    state=NS(commits=[],remote={},snapshots={},downloads=[],fail=None,on_download=None)
    def commit(**kwargs):
        if state.fail:raise state.fail
        state.commits.append(kwargs)
        for operation in kwargs['operations']:
            state.remote[operation.path_in_repo]=Path(operation.path_or_fileobj).read_bytes()
        oid=str(len(state.commits));state.snapshots[oid]=dict(state.remote)
        return NS(oid=oid)
    def download(repo,path,repo_type,revision):
        state.downloads.append((path,revision))
        if state.on_download:state.on_download()
        data=state.snapshots[revision][path]
        local=tmp_path/'downloads'/revision/path;local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(data)
        return str(local)
    module=types.ModuleType('huggingface_hub')
    module.HfApi=lambda:NS(create_commit=commit)
    module.CommitOperationAdd=lambda **kwargs:NS(**kwargs)
    module.hf_hub_download=download
    monkeypatch.setitem(sys.modules,'huggingface_hub',module)
    state.download=download
    return state

@pytest.fixture
def clock(monkeypatch):
    value=[100000.]
    monkeypatch.setattr(st.time,'time',lambda:value[0])
    return value


def test_store_only_enqueues_and_never_contacts_hub(tmp_path,hub):
    store=st.Store(tmp_path,'owner/repo','run1')
    receipt=store.record('rows/test.json',{'success':True})
    assert receipt['local'] and receipt['verified'] and receipt['pending_remote']
    assert receipt['path']=='rows/test.json' and not hub.commits and not hub.downloads
    queue=list((tmp_path/'.checkpoint_queue').glob('*.json'))
    assert len(queue)==1 and json.loads(queue[0].read_text())['sha256']==receipt['sha256']
    store.record('rows/test.json',{'success':False})
    assert len(list((tmp_path/'.checkpoint_queue').glob('*.json')))==1
    assert not list(tmp_path.rglob('*.tmp'))


def test_batch_single_commit_roundtrip_and_fixed_revision_verification(tmp_path,hub,clock):
    root=tmp_path/'source';store=st.Store(root,'owner/repo','run1')
    store.record('protocol.json',{'version':1})
    store.record('rows/test.json',{'success':True})
    (root/'episodes').mkdir();(root/'episodes'/'test.npz').write_bytes(b'actual binary payload')
    store.push(root/'episodes'/'test.npz')
    pub=st.BatchPublisher(root,'owner/repo','run1')
    report=pub.flush()
    assert report['status']=='published' and report['files']==3
    assert len(hub.commits)==1 and len(hub.commits[0]['operations'])==2
    assert len(hub.downloads)==2 and {revision for _,revision in hub.downloads}=={'1'}
    assert not pub.has_pending() and pub.stats['verified_files']==3
    restored=st.restore_checkpoints('owner/repo','run1','1',tmp_path/'restored')
    assert restored['files']==3
    assert (tmp_path/'restored'/'episodes'/'test.npz').read_bytes()==b'actual binary payload'
    assert json.loads((tmp_path/'restored'/'rows'/'test.json').read_text())['success']


@pytest.mark.parametrize('headers,delay',[({},3600),({'Retry-After':'600'},600),({'RateLimit':'"api";r=0;t=240'},240)])
def test_429_defers_without_dropping_queue_or_immediate_retry(tmp_path,hub,clock,headers,delay):
    store=st.Store(tmp_path,'owner/repo','run1');store.record('one.json',{'a':1})
    pub=st.BatchPublisher(tmp_path,'owner/repo','run1');hub.fail=RateLimited(headers)
    result=pub.flush(force=True)
    assert result['status']=='deferred' and result['reason']=='HTTP 429'
    assert result['next_attempt_at']==clock[0]+delay and pub.pending_count()==1
    hub.fail=None
    assert pub.flush(force=True)['status']=='deferred' and not hub.commits
    # A newly constructed publisher honors the persisted cooldown.
    pub=st.BatchPublisher(tmp_path,'owner/repo','run1')
    clock[0]+=delay-1
    assert pub.flush(force=True)['status']=='deferred'
    clock[0]+=1
    assert pub.flush(force=True)['status']=='published' and len(hub.commits)==1


def test_force_cannot_bypass_minimum_commit_spacing(tmp_path,hub,clock):
    store=st.Store(tmp_path,'owner/repo','run1');pub=st.BatchPublisher(tmp_path,'owner/repo','run1')
    store.record('one.json',{});pub.flush(force=True)
    store.record('two.json',{});clock[0]+=29
    assert pub.flush(force=True)['status']=='deferred' and len(hub.commits)==1
    clock[0]+=1
    assert pub.flush(force=True)['status']=='published' and len(hub.commits)==2


def test_failed_readback_keeps_queue_and_inflight(tmp_path,hub,clock,monkeypatch):
    store=st.Store(tmp_path,'owner/repo','run1');store.record('one.json',{})
    pub=st.BatchPublisher(tmp_path,'owner/repo','run1')
    module=sys.modules['huggingface_hub'];original=module.hf_hub_download
    def corrupted(*a,**kw):
        path=Path(original(*a,**kw));path.write_bytes(b'corrupted');return str(path)
    monkeypatch.setattr(module,'hf_hub_download',corrupted)
    with pytest.raises(RuntimeError,match='checksum'):pub.flush()
    assert pub.has_pending() and (tmp_path/'.checkpoint_publish'/'inflight.json').exists()
    assert pub.flush(force=True)['status']=='deferred'


def test_concurrent_new_version_survives_acknowledgement(tmp_path,hub,clock):
    store=st.Store(tmp_path,'owner/repo','run1');store.record('mutable.json',{'v':1})
    pub=st.BatchPublisher(tmp_path,'owner/repo','run1')
    def update():
        hub.on_download=None;store.record('mutable.json',{'v':2})
    hub.on_download=update
    assert pub.flush()['status']=='published' and pub.pending_count()==1
    clock[0]+=120;pub.flush()
    st.restore_checkpoints('owner/repo','run1','2',tmp_path/'restored')
    assert json.loads((tmp_path/'restored'/'mutable.json').read_text())=={'v':2}


def test_file_count_bound_splits_batches(tmp_path,hub,clock,monkeypatch):
    monkeypatch.setattr(st,'MAX_FILES',2)
    store=st.Store(tmp_path,'owner/repo','run1');pub=st.BatchPublisher(tmp_path,'owner/repo','run1')
    for i in range(3):store.record(str(i)+'.json',{'i':i})
    assert pub.flush()['files']==2 and pub.pending_count()==1
    clock[0]+=120
    assert pub.flush()['files']==1 and not pub.has_pending()


def test_restore_refuses_tampered_pack(tmp_path,hub,clock):
    store=st.Store(tmp_path,'owner/repo','run1');store.record('one.json',{'a':1})
    st.BatchPublisher(tmp_path,'owner/repo','run1').flush()
    name=next(path for path in hub.snapshots['1'] if path.endswith('.zip'))
    hub.snapshots['1'][name]=b'corrupt'
    with pytest.raises(RuntimeError,match='checksum'):st.restore_checkpoints('owner/repo','run1','1',tmp_path/'restored')


def test_restore_rejects_traversal_manifest(tmp_path,hub,clock):
    store=st.Store(tmp_path,'owner/repo','run1');store.record('one.json',{'a':1})
    st.BatchPublisher(tmp_path,'owner/repo','run1').flush()
    path='runs/run1/checkpoints/index.json';index=json.loads(hub.snapshots['1'][path])
    index['packs'][0]['files'][0]['path']='../escape.json'
    hub.snapshots['1'][path]=json.dumps(index).encode()
    with pytest.raises(ValueError,match='Unsafe'):st.restore_checkpoints('owner/repo','run1','1',tmp_path/'restored')


def test_startup_metadata_precedes_data_when_batch_is_full(tmp_path,hub,clock,monkeypatch):
    monkeypatch.setattr(st,'MAX_FILES',3)
    store=st.Store(tmp_path,'owner/repo','run1')
    for i in range(6):store.record('data/'+str(i)+'.json',{'i':i})
    store.record('rendering.json',{'backend':'egl'})
    store.record('protocol.json',{'version':1})
    store.record('cloud_bootstrap_runtime.json',{'runtime':'test'})
    pub=st.BatchPublisher(tmp_path,'owner/repo','run1');pub.flush()
    packed=pub.state['packs'][0]['files']
    assert [rec['path'] for rec in packed]==['protocol.json','rendering.json','cloud_bootstrap_runtime.json']
    assert pub.pending_count()==6
