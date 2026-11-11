"""Read-only legacy-data audit before a paid run; no model/result migration."""
import ast,hashlib,json,os,re,shutil,tempfile,zipfile
from pathlib import Path
from .spec import CONFIG,scenes,fingerprint

LEGACY_TRAIN_SHA='5dbb7e6443f71dda02071b057f433ba4bcc53f79c72c75016b39bbfcf8ceeace'
SCIENTIFIC_FILES=('experiment/spec.py','experiment/env.py','experiment/model.py',
                  'experiment/proposer.py','experiment/core.py','experiment/evaluate.py',
                  'requirements.txt','cloud_entry.sh')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def _json(path,limit=2*1024*1024):
    path=Path(path)
    if path.stat().st_size>limit:raise RuntimeError('Recovery metadata is too large')
    return json.loads(path.read_text())

def _sdk(api,download):
    if api is None or download is None:
        from huggingface_hub import HfApi,hf_hub_download
        if api is None:api=HfApi()
        if download is None:download=hf_hub_download
    return api,download

def _ast_node(source,name):
    found=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(found)!=1:raise RuntimeError('Expected one scientific function: '+name)
    return found[0]

def _collection_loop(source):
    collect=_ast_node(source,'collect')
    found=[n for n in ast.walk(collect) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step']
    if len(found)!=1:raise RuntimeError('Expected one per-episode collection loop')
    return ast.dump(found[0],include_attributes=False)

def verify_compatible_source(source_zip,current_source_root):
    root=Path(current_source_root)
    with zipfile.ZipFile(source_zip) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise RuntimeError('Duplicate paths in original source archive')
        checked={}
        for name in SCIENTIFIC_FILES:
            info=z.getinfo(name)
            if info.file_size>2*1024*1024:raise RuntimeError('Unexpected original source size')
            original=z.read(name);current=(root/name).read_bytes()
            if original!=current:raise RuntimeError('Scientific source changed: '+name)
            checked[name]=hashlib.sha256(original).hexdigest()
        original=z.read('experiment/train.py')
        if hashlib.sha256(original).hexdigest()!=LEGACY_TRAIN_SHA:
            raise RuntimeError('Unrecognized original collection implementation')
        old=original.decode();new=(root/'experiment/train.py').read_text()
        for name in ('arrays','train_seed'):
            if ast.dump(_ast_node(old,name),include_attributes=False)!=ast.dump(_ast_node(new,name),include_attributes=False):
                raise RuntimeError('Training algorithm changed: '+name)
        if _collection_loop(old)!=_collection_loop(new):raise RuntimeError('Per-episode collection algorithm changed')
        # Sharding and atomic persistence may change; per-scene initial conditions may not.
        new_nodes={ast.dump(n,include_attributes=False) for n in ast.walk(_ast_node(new,'collect')) if isinstance(n,ast.Assign)}
        for name in ('env','policy','rng','zz','aa','yy','success','z'):
            assignments=[n for n in ast.walk(_ast_node(old,'collect')) if isinstance(n,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
            if any(ast.dump(n,include_attributes=False) not in new_nodes for n in assignments):
                raise RuntimeError('Per-scene collection state changed: '+name)
    return dict(exact_files=checked,legacy_train_sha256=LEGACY_TRAIN_SHA,
                collection_loop_unchanged=True,training_functions_unchanged=True,
                permitted_changes='worker sharding, atomic persistence, upload batching, restart plumbing')

def validate_legacy_npz(path,scene):
    import numpy as np
    path=Path(path)
    if path.stat().st_size>16*1024*1024:raise RuntimeError('Unexpected trajectory file size')
    with zipfile.ZipFile(path) as z:
        if sum(n.file_size for n in z.infolist())>16*1024*1024:raise RuntimeError('Unexpected expanded trajectory size')
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'z','a','y','success','scene'}:raise RuntimeError('Unexpected trajectory schema')
        z,a,y=data['z'],data['a'],data['y']
        n=len(a)
        if not 1<=n<=CONFIG['steps'] or z.shape!=(n,524) or y.shape!=(n,524) or a.shape!=(n,4):
            raise RuntimeError('Invalid trajectory dimensions')
        if any(x.dtype!=np.dtype('float32') or not np.isfinite(x).all() for x in (z,a,y)):
            raise RuntimeError('Invalid trajectory values or dtype')
        if np.any(np.abs(a)>1.000001):raise RuntimeError('Action outside protocol bounds')
        meta=data['scene'];success=data['success']
        if meta.shape!=() or meta.dtype.kind not in ('U','S') or json.loads(str(meta.item()))!=scene:
            raise RuntimeError('Trajectory scene metadata mismatch')
        if success.shape!=() or success.dtype.kind!='b':raise RuntimeError('Invalid success annotation')
        if n>1 and not np.array_equal(z[1:],y[:-1]):raise RuntimeError('Trajectory continuity mismatch')
        task=float(scene['task']!='Lift')
        if np.any(z[:,-1]!=task) or np.any(y[:,-1]!=task):raise RuntimeError('Trajectory task bit mismatch')
        if not np.array_equal(z[:,520:524],y[:,520:524]):raise RuntimeError('Static task/goal changed')
        return dict(steps=n,success=bool(success.item()))

def _metadata_hash(entry):
    lfs=getattr(entry,'lfs',None)
    value=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
    if value is not None and not re.fullmatch(r'[a-f0-9]{64}',value):raise RuntimeError('Invalid Hub LFS digest')
    return value

def inspect_recovery(repo,parent_run,expected_source_sha,*,api=None,download=None,current_source_root=None,revision=None):
    """Download and validate complete legacy training/validation data on the caller's CPU.

    All Hub operations are read-only. Network/metadata errors propagate, never
    become an empty-directory success. Returns a small JSON-serializable manifest.
    """
    if not re.fullmatch(r'3d-[a-f0-9]{12}',parent_run):raise ValueError('Invalid parent run')
    if not re.fullmatch(r'[a-f0-9]{64}',expected_source_sha):raise ValueError('Invalid expected source digest')
    api,download=_sdk(api,download)
    revision=revision or api.repo_info(repo,repo_type='dataset').sha
    if not re.fullmatch(r'[a-f0-9]{40}',revision):raise RuntimeError('A fixed repository revision is required')
    prefix='runs/'+parent_run+'/'
    files=api.list_repo_files(repo,repo_type='dataset',revision=revision)
    parent_files=[p[len(prefix):] for p in files if p.startswith(prefix)]
    if any(p.startswith(('models/','rows/','episodes/','checkpoints/')) or p=='COMPLETED.json' for p in parent_files):
        raise RuntimeError('Parent contains training/evaluation artifacts; data-only migration refused')
    if any(p not in parent_files for p in ('protocol.json','rendering.json','cloud_bootstrap_runtime.json')):
        raise RuntimeError('Parent protocol/renderer/runtime provenance missing')
    def get(remote):return Path(download(repo,remote,repo_type='dataset',revision=revision))
    protocol_path=get(prefix+'protocol.json');protocol=_json(protocol_path)
    if protocol.get('fingerprint')!=fingerprint() or protocol.get('source_sha256')!=expected_source_sha:
        raise RuntimeError('Parent source/protocol identity mismatch')
    for split in ('train','val','test'):
        if protocol.get(split)!=scenes(split):raise RuntimeError('Parent scene split differs: '+split)
    if protocol.get('config')!=CONFIG:raise RuntimeError('Parent configuration differs')
    source=get('source/'+expected_source_sha+'.zip')
    if digest(source)!=expected_source_sha:raise RuntimeError('Original source archive hash mismatch')
    compatibility=verify_compatible_source(source,current_source_root or Path(__file__).resolve().parents[1])
    rendering_path=get(prefix+'rendering.json');rendering=_json(rendering_path)
    if rendering.get('backend') not in ('egl','osmesa'):raise RuntimeError('Unknown parent rendering backend')
    runtime_path=get(prefix+'cloud_bootstrap_runtime.json');runtime=_json(runtime_path)
    if not isinstance(runtime.get('after'),dict) or not runtime['after'].get('cuda_available'):
        raise RuntimeError('Parent CUDA runtime identity missing')
    expected={s['id']:s for split in ('train','val') for s in scenes(split)}
    data_files=sorted(p for p in parent_files if p.startswith('data/'))
    if not data_files:raise RuntimeError('No archived training/validation trajectories to reuse')
    entries={getattr(e,'path',getattr(e,'rfilename','')):e for e in
             api.list_repo_tree(repo,path_in_repo=prefix+'data',recursive=True,repo_type='dataset',revision=revision)}
    verified=[]
    for relative in data_files:
        ident=Path(relative).stem
        if relative!='data/'+ident+'.npz' or ident not in expected:raise RuntimeError('Unexpected parent data path: '+relative)
        remote=prefix+relative;path=get(remote);value=digest(path);entry=entries.get(remote)
        if entry is None:raise RuntimeError('Missing pinned Hub file metadata: '+remote)
        lfs_sha=_metadata_hash(entry)
        if lfs_sha and value!=lfs_sha:raise RuntimeError('Hub LFS checksum mismatch: '+relative)
        size=getattr(entry,'size',None)
        if size is not None and path.stat().st_size!=size:raise RuntimeError('Hub file size mismatch: '+relative)
        stats=validate_legacy_npz(path,expected[ident])
        verified.append(dict(path=relative,remote=remote,scene_id=ident,sha256=value,size_bytes=path.stat().st_size,
                             hub_lfs_sha256=lfs_sha,**stats))
    return dict(version=1,repo=repo,parent_run=parent_run,revision=revision,protocol=fingerprint(),
                source_sha256=expected_source_sha,source_compatibility=compatibility,
                protocol_file_sha256=digest(protocol_path),rendering_file=dict(remote=prefix+'rendering.json',sha256=digest(rendering_path)),
                rendering_metadata=rendering,bootstrap_runtime=runtime,bootstrap_runtime_sha256=digest(runtime_path),
                rendering_backend=rendering['backend'],data=verified,recovered_episodes=len(verified),
                expected_data_episodes=len(expected),missing_scene_ids=sorted(set(expected)-{v['scene_id'] for v in verified}))

def restore_legacy_data(manifest,dest,*,download=None):
    """Restore only audited data and renderer; never models or evaluation outcomes."""
    if manifest.get('version')!=1 or manifest.get('protocol')!=fingerprint():raise RuntimeError('Recovery manifest protocol mismatch')
    run=manifest.get('parent_run','');revision=manifest.get('revision','');repo=manifest.get('repo')
    if not re.fullmatch(r'3d-[a-f0-9]{12}',run) or not re.fullmatch(r'[a-f0-9]{40}',revision) or not isinstance(repo,str) or not repo:
        raise RuntimeError('Invalid recovery origin')
    if download is None:
        from huggingface_hub import hf_hub_download
        download=hf_hub_download
    root=Path(dest);root.mkdir(parents=True,exist_ok=True)
    expected={s['id']:s for split in ('train','val') for s in scenes(split)};seen=set();restored=[]
    def install(remote,relative,sha256,scene=None):
        if not re.fullmatch(r'[a-f0-9]{64}',sha256):raise RuntimeError('Invalid manifest checksum')
        source=Path(download(repo,remote,repo_type='dataset',revision=revision))
        if digest(source)!=sha256:raise RuntimeError('Recovery download checksum mismatch: '+relative)
        if scene is not None:validate_legacy_npz(source,scene)
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=sha256:raise RuntimeError('Existing local artifact differs: '+relative)
            return
        fd,name=tempfile.mkstemp(prefix='.'+target.name+'.',dir=target.parent)
        try:
            with os.fdopen(fd,'wb') as out,source.open('rb') as src:shutil.copyfileobj(src,out);out.flush();os.fsync(out.fileno())
            os.replace(name,target)
        finally:
            if os.path.exists(name):os.unlink(name)
    for entry in manifest.get('data',[]):
        ident=entry.get('scene_id');relative=entry.get('path');remote=entry.get('remote')
        if ident not in expected or ident in seen or relative!='data/'+ident+'.npz' or remote!='runs/'+run+'/'+relative:
            raise RuntimeError('Invalid or duplicate recovery data path')
        seen.add(ident);install(remote,relative,entry.get('sha256',''),expected[ident]);restored.append(relative)
    if not restored or len(restored)!=manifest.get('recovered_episodes'):raise RuntimeError('Recovery trajectory count mismatch')
    render=manifest.get('rendering_file',{})
    if render.get('remote')!='runs/'+run+'/rendering.json':raise RuntimeError('Invalid rendering provenance path')
    install(render['remote'],'rendering.json',render.get('sha256',''))
    actual=_json(root/'rendering.json')
    if actual.get('backend')!=manifest.get('rendering_backend') or actual.get('backend') not in ('egl','osmesa'):
        raise RuntimeError('Recovered renderer identity mismatch')
    return dict(parent_run=run,revision=revision,recovered_episodes=len(restored),rendering_backend=actual['backend'])
