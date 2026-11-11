"""Local-first artifacts with one bounded, rate-aware checkpoint publisher."""
import email.utils,fcntl,hashlib,io,json,os,re,time,uuid,zipfile
from pathlib import Path,PurePosixPath

MAX_FILES=256
MAX_BYTES=512*1024*1024

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def _atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()

def save_json(path,obj):
    _atomic_bytes(path,(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

def _relative(value):
    if not isinstance(value,str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Invalid checkpoint path')
    p=PurePosixPath(value)
    if p.is_absolute() or any(part in ('.','..') or part.startswith('.') for part in p.parts) or p.as_posix()!=value:
        raise ValueError('Unsafe checkpoint path')
    return value

def _inside(root,relative):
    relative=_relative(relative);target=Path(root)/relative
    if not target.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Checkpoint escapes destination')
    return target

def _digest(data):return hashlib.sha256(data).hexdigest()

class Store:
    def __init__(self,root,repo=None,run=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.repo=repo;self.run=run
    def push(self,path):
        path=Path(path);relative=_relative(path.relative_to(self.root).as_posix());digest=sha(path)
        if self.repo:
            # Each process writes a distinct temporary file, then atomically replaces
            # the latest descriptor for this logical path. No network in workers.
            key=_digest(relative.encode())
            queue=self.root/'.checkpoint_queue';queue.mkdir(exist_ok=True)
            with (queue/(key+'.lock')).open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                save_json(queue/(key+'.json'),dict(path=relative,sha256=digest,size=path.stat().st_size))
        return dict(path=relative,sha256=digest,verified=True,local=True,pending_remote=bool(self.repo))
    def record(self,name,obj):
        p=_inside(self.root,name);save_json(p,obj);return self.push(p)

def _retry_after(exc,now):
    response=getattr(exc,'response',None)
    if getattr(response,'status_code',None)!=429:return None
    headers=getattr(response,'headers',{}) or {}
    value=headers.get('Retry-After') or headers.get('retry-after')
    if value:
        try:return max(30.,float(value))
        except (ValueError,TypeError):
            try:return max(30.,email.utils.parsedate_to_datetime(value).timestamp()-now)
            except (ValueError,TypeError,OverflowError):pass
    # Hugging Face uses RateLimit reset durations, e.g. "api";r=0;t=120.
    value=headers.get('RateLimit') or headers.get('ratelimit') or ''
    match=re.search(r'(?:^|[;,])\s*t\s*=\s*(\d+)',value)
    if match:return max(30.,float(match.group(1)))
    return 3600.

class BatchPublisher:
    """The driver alone owns this object; workers only enqueue local files.

    force bypasses batching delay, never the minimum commit spacing or cooldown.
    Each remote checkpoint is one commit containing an immutable ZIP and index.
    """
    def __init__(self,root,repo,run,interval=120,min_interval=30):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.repo=repo;self.run=run;self.interval=max(30.,float(interval));self.min_interval=max(30.,float(min_interval))
        self.directory=self.root/'.checkpoint_publish';self.directory.mkdir(exist_ok=True)
        self.state_path=self.directory/'state.json';self.queue=self.root/'.checkpoint_queue'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(packs=[],last_attempt=0.,next_attempt_at=0.,stats=dict(commits=0,verified_files=0,bytes=0))
        if self.state.get('run',run)!=run:raise RuntimeError('Publisher state belongs to another run')
        self.state['run']=run
    @property
    def next_attempt_at(self):return self.state.get('next_attempt_at',0.)
    @property
    def stats(self):return dict(self.state['stats'])
    def pending_count(self):return len(list(self.queue.glob('*.json'))) if self.queue.exists() else 0
    def has_pending(self):return self.pending_count()>0
    def _status(self,status,**kwargs):
        result=dict(status=status,pending=self.pending_count(),next_attempt_at=self.next_attempt_at,stats=self.stats)
        result.update(kwargs);return result
    def _persist(self):save_json(self.state_path,self.state)
    def _snapshot(self):
        selected=[];content=[];total=0
        metadata=('protocol.json','rendering.json','cloud_bootstrap_runtime.json','data_recovery.json','resume.json')
        descriptors=[]
        for q in self.queue.glob('*.json'):
            try:rec=json.loads(q.read_text())
            except FileNotFoundError:continue
            priority=metadata.index(rec['path']) if rec['path'] in metadata else len(metadata)
            descriptors.append((priority,rec['path'],q,rec))
        for _,_,q,rec in sorted(descriptors,key=lambda item:(item[0],item[1])):
            path=_inside(self.root,rec['path'])
            if rec['size']>MAX_BYTES:raise RuntimeError('Individual checkpoint file exceeds size bound')
            if len(selected)>=MAX_FILES or total+rec['size']>MAX_BYTES:break
            try:data=path.read_bytes()
            except FileNotFoundError:continue
            if len(data)!=rec['size'] or _digest(data)!=rec['sha256']:continue
            selected.append(dict(rec,queue_file=q.name));content.append((rec['path'],data));total+=len(data)
        if not selected:return None
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_STORED) as z:
            for relative,data in content:
                info=zipfile.ZipInfo(relative,(2026,10,23,0,0,0));z.writestr(info,data)
        data=output.getvalue();digest=_digest(data);path='checkpoints/packs/'+digest+'.zip'
        pack=dict(path=path,sha256=digest,bytes=len(data),files=[{k:r[k] for k in ('path','sha256','size')} for r in selected])
        packs=list(self.state['packs'])
        if not any(p['path']==path for p in packs):packs.append(pack)
        index=dict(format='adajepa-checkpoints-v1',run=self.run,packs=packs)
        _atomic_bytes(self.directory/'pack.zip',data);save_json(self.directory/'index.json',index)
        inflight=dict(pack=pack,selected=selected,index=index,index_sha256=sha(self.directory/'index.json'))
        save_json(self.directory/'inflight.json',inflight)
        return inflight
    def _acknowledge(self,inflight,revision):
        self.state['packs']=inflight['index']['packs'];self.state['revision']=revision
        self.state['stats']['verified_files']+=len(inflight['selected'])
        self.state['stats']['bytes']+=inflight['pack']['bytes']
        self.state['next_attempt_at']=max(self.state['next_attempt_at'],time.time()+self.min_interval)
        self._persist()
        for rec in inflight['selected']:
            path=self.queue/rec['queue_file']
            with path.with_suffix('.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                try:current=json.loads(path.read_text())
                except FileNotFoundError:continue
                if all(current.get(k)==rec[k] for k in ('path','sha256','size')):path.unlink(missing_ok=True)
        (self.directory/'inflight.json').unlink(missing_ok=True)
    def flush(self,force=False):
        with (self.directory/'publisher.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if self.state_path.exists():self.state=json.loads(self.state_path.read_text())
            if not self.has_pending() and not (self.directory/'inflight.json').exists():return self._status('empty')
            now=time.time();due=max(self.next_attempt_at,self.state['last_attempt']+(self.min_interval if force else self.interval))
            if now<due:return self._status('deferred',reason='cadence_or_rate_limit',next_attempt_at=due)
            inflight=json.loads((self.directory/'inflight.json').read_text()) if (self.directory/'inflight.json').exists() else self._snapshot()
            if inflight is None:return self._status('deferred',reason='files_changing')
            self.state['last_attempt']=now;self.state['next_attempt_at']=now+self.min_interval;self._persist()
            from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
            prefix='runs/'+self.run+'/'
            try:
                api=HfApi()
                result=api.create_commit(repo_id=self.repo,repo_type='dataset',
                    operations=[CommitOperationAdd(path_in_repo=prefix+inflight['pack']['path'],path_or_fileobj=str(self.directory/'pack.zip')),
                                CommitOperationAdd(path_in_repo=prefix+'checkpoints/index.json',path_or_fileobj=str(self.directory/'index.json'))],
                    commit_message='Checkpoint '+inflight['pack']['sha256'][:12])
                self.state['stats']['commits']+=1;self._persist()
                downloaded=hf_hub_download(self.repo,prefix+inflight['pack']['path'],repo_type='dataset',revision=result.oid)
                if sha(downloaded)!=inflight['pack']['sha256']:raise RuntimeError('Remote checkpoint checksum mismatch')
                downloaded_index=hf_hub_download(self.repo,prefix+'checkpoints/index.json',repo_type='dataset',revision=result.oid)
                if sha(downloaded_index)!=inflight['index_sha256']:raise RuntimeError('Remote checkpoint index mismatch')
                self._acknowledge(inflight,result.oid)
                return self._status('published',commit=result.oid,files=len(inflight['selected']))
            except Exception as exc:
                delay=_retry_after(exc,time.time())
                if delay is not None:
                    self.state['next_attempt_at']=time.time()+delay;self.state['last_error']='HTTP 429';self._persist()
                    return self._status('deferred',reason='HTTP 429',retry_after_seconds=delay)
                # Never discard local queue after unknown HTTP outcomes.
                raise

def restore_checkpoints(repo,run,revision,dest):
    """Restore only hashed checkpoint packs from one fixed repository revision."""
    from huggingface_hub import hf_hub_download
    prefix='runs/'+run+'/';dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    index_path=hf_hub_download(repo,prefix+'checkpoints/index.json',repo_type='dataset',revision=revision)
    if Path(index_path).stat().st_size>50*1024*1024:raise RuntimeError('Checkpoint index exceeds bound')
    index=json.loads(Path(index_path).read_text())
    if index.get('format')!='adajepa-checkpoints-v1' or index.get('run')!=run or not isinstance(index.get('packs'),list):
        raise RuntimeError('Invalid checkpoint index identity')
    if len(index['packs'])>20000:raise RuntimeError('Too many checkpoint packs')
    restored={};seen_packs=set()
    for pack in index['packs']:
        pack_path=_relative(pack['path'])
        if not re.fullmatch(r'checkpoints/packs/[a-f0-9]{64}\.zip',pack_path) or pack_path in seen_packs:
            raise RuntimeError('Invalid or duplicate checkpoint pack path')
        seen_packs.add(pack_path)
        if pack_path!='checkpoints/packs/'+pack['sha256']+'.zip':raise RuntimeError('Checkpoint pack identity mismatch')
        files=pack['files']
        if not isinstance(files,list) or len(files)>MAX_FILES:raise RuntimeError('Checkpoint file count exceeds bound')
        if any(not isinstance(f['size'],int) or f['size']<0 for f in files) or sum(f['size'] for f in files)>MAX_BYTES:
            raise RuntimeError('Checkpoint payload exceeds bound')
        if not isinstance(pack['bytes'],int) or pack['bytes']>MAX_BYTES+2*1024*1024 or pack['bytes']<0:
            raise RuntimeError('Checkpoint archive exceeds bound')
        expected={_relative(f['path']):f for f in files}
        if len(expected)!=len(files):raise RuntimeError('Duplicate file in checkpoint manifest')
        downloaded=hf_hub_download(repo,prefix+pack_path,repo_type='dataset',revision=revision)
        if Path(downloaded).stat().st_size!=pack['bytes'] or sha(downloaded)!=pack['sha256']:
            raise RuntimeError('Restored checkpoint checksum mismatch')
        with zipfile.ZipFile(downloaded) as z:
            infos=z.infolist()
            if len(infos)!=len(expected) or {i.filename for i in infos}!=set(expected):raise RuntimeError('Checkpoint members disagree with manifest')
            for info in infos:
                target=_inside(dest,info.filename);rec=expected[info.filename]
                if info.file_size!=rec['size'] or (info.external_attr>>16)&0o170000==0o120000:
                    raise RuntimeError('Unsafe checkpoint member')
                data=z.read(info)
                if _digest(data)!=rec['sha256']:raise RuntimeError('Checkpoint member checksum mismatch')
                _atomic_bytes(target,data);restored[info.filename]=rec['sha256']
    return dict(run=run,revision=revision,packs=len(index['packs']),files=len(restored),sha256=restored)
