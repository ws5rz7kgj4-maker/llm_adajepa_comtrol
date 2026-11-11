"""Bounded compute workers and one batched, verified checkpoint publisher."""
import argparse,json,os,re,shutil,subprocess,sys,tempfile,time,traceback
from pathlib import Path
from .spec import CONFIG,MODEL_SEEDS,scenes,ARMS,fingerprint
from .storage import Store,BatchPublisher,restore_checkpoints,sha,_atomic_bytes


def scrub(value):
    text=str(value);token=os.environ.get('HF_TOKEN','')
    return text.replace(token,'[REDACTED]') if token else text


def checkpoint_tick(publisher,root,force=False,record_activity=True):
    """One owner; network failures retain the queue and cannot mask worker errors."""
    if time.time()<getattr(publisher,'_driver_retry_at',0):return {'status':'deferred','reason':'network_backoff'}
    started=time.time()
    try:
        result=publisher.flush(force=force)
    except Exception as exc:
        publisher._driver_retry_at=time.time()+60
        result=dict(status='error',error_type=type(exc).__name__,message=scrub(exc),pending=publisher.pending_count())
    if result.get('status') in ('published','error') or result.get('reason')=='HTTP 429':
        event=dict(started_unix=started,finished_unix=time.time(),**result)
        print('CHECKPOINT',json.dumps(event),flush=True)
        if record_activity:
            path=Path(root)/'logs'/'checkpoint_activity.jsonl';path.parent.mkdir(exist_ok=True,parents=True)
            with path.open('a') as f:f.write(json.dumps(event)+'\n')
    return result


def drain_checkpoints(publisher,root,max_seconds=3900):
    """Bounded final drain. A rate limit never becomes a per-file retry storm."""
    deadline=time.monotonic()+max_seconds;last_progress=0.
    while publisher.has_pending():
        if time.monotonic()>=deadline:return False
        # Logs have already been enqueued: keep their bytes stable during drain.
        checkpoint_tick(publisher,root,force=True,record_activity=False)
        if not publisher.has_pending():return True
        now=time.monotonic()
        if now-last_progress>=30:
            print('CHECKPOINT_WAIT',json.dumps(dict(pending_files=publisher.pending_count(),
                retry_after_seconds=max(0,publisher.next_attempt_at-time.time()),stats=publisher.stats)),flush=True)
            last_progress=now
        time.sleep(min(5,max(0,deadline-now)))
    return True


def durable_counts(publisher):
    latest={}
    for pack in getattr(publisher,'state',{}).get('packs',[]):
        for item in pack.get('files',[]):latest[item['path']]=item['sha256']
    rows={Path(name).stem for name in latest if name.startswith('rows/') and name.endswith('.json')}
    trajectories={Path(name).stem for name in latest if name.startswith('episodes/') and name.endswith('.npz')}
    return dict(durable_data_episodes=sum(name.startswith('data/') and name.endswith('.npz') for name in latest),
                durable_episode_pairs=len(rows&trajectories))


def snapshot_logs(root,store,phase,limit=128*1024):
    """Stable bounded snapshots of active logs; full logs follow phase completion."""
    metadata=[]
    for path in sorted((root/'logs').glob(phase+'-*.log')):
        with path.open('rb') as handle:
            end=handle.seek(0,os.SEEK_END);start=max(0,end-limit);handle.seek(start)
            data=handle.read(end-start)
        target=root/'logs'/'tails'/(path.stem+'.tail.log')
        _atomic_bytes(target,data);store.push(target)
        metadata.append(dict(source=path.relative_to(root).as_posix(),source_bytes=end,
            captured_start_byte=start,captured_bytes=len(data),tail=target.relative_to(root).as_posix()))
    store.record('logs/tails/'+phase+'-capture.json',dict(captured_unix=time.time(),files=metadata))


def run_group(commands,root,phase,devices,publisher=None,store=None):
    jobs=[]
    try:
        for i,cmd in enumerate(commands):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(devices[i]);env['MUJOCO_EGL_DEVICE_ID']=str(devices[i])
            env['OMP_NUM_THREADS']='2';env['NUMBA_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
            env['PYTHONUNBUFFERED']='1'
            log=(root/'logs'/f'{phase}-{i}.log').open('a');print('START',phase,i,flush=True)
            jobs.append((subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        last_progress=0.;last_log_snapshot=float('-inf')
        while jobs and any(p.poll() is None for p,l in jobs):
            if any(p.poll() not in (None,0) for p,l in jobs):raise RuntimeError('Worker failed: '+phase+'; inspect saved logs')
            if store and time.monotonic()-last_log_snapshot>=120:
                snapshot_logs(root,store,phase);last_log_snapshot=time.monotonic()
            if publisher:checkpoint_tick(publisher,root)
            if time.monotonic()-last_progress>=30:
                print('PROGRESS',json.dumps(dict(phase=phase,
                    local_completed_episodes=len(list((root/'rows').glob('*.json'))),
                    local_data_episodes=len(list((root/'data').glob('*.npz'))),
                    pending_upload_files=publisher.pending_count() if publisher else 0,
                    remote_checkpoint_stats=publisher.stats if publisher else {},
                    **(durable_counts(publisher) if publisher else {}))),flush=True)
                last_progress=time.monotonic()
            time.sleep(2)
        if any(p.returncode for p,l in jobs):raise RuntimeError('Worker failed: '+phase)
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate()
        for p,l in jobs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            l.close()
    if store:queue_logs(root,store)
    if publisher:checkpoint_tick(publisher,root,force=True)


def validate_episode_pairs(root,quarantine=False):
    """Only complete matching rows/trajectories are resumable or analyzable."""
    root=Path(root);rows={p.stem:p for p in (root/'rows').glob('*.json')}
    episodes={p.stem:p for p in (root/'episodes').glob('*.npz')}
    allowed={f"{s['id']}__seed{seed}__{a.name}" for s in scenes() for seed in MODEL_SEEDS for a in ARMS}
    unknown=(set(rows)|set(episodes))-allowed
    if unknown:raise RuntimeError('Unexpected formal artifact identities')
    orphans=set(rows)^set(episodes)
    if orphans and not quarantine:raise RuntimeError('Unpaired formal artifacts; analysis blocked')
    for ident in sorted(orphans):
        path=rows.get(ident) or episodes[ident]
        target=root/'quarantine'/path.parent.name/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise RuntimeError('Quarantine target already exists')
        os.replace(path,target)
    paired=set(rows)&set(episodes)
    for ident in paired:
        row=json.loads(rows[ident].read_text())
        if row.get('id')!=ident or row.get('protocol')!=fingerprint():raise RuntimeError('Formal row identity mismatch')
        expected=f"{row['scene']['id']}__seed{row['model_seed']}__{row['method']}"
        if expected!=ident or sha(episodes[ident])!=row.get('trajectory_receipt',{}).get('sha256'):
            raise RuntimeError('Formal trajectory/row checksum mismatch')
    return dict(complete_pairs=len(paired),quarantined_orphans=len(orphans))


def restore(root,repo,run):
    from huggingface_hub import HfApi
    root=Path(root);revision=HfApi().repo_info(repo,repo_type='dataset').sha
    expected=os.environ.get('EXPERIMENT_SOURCE_SHA256','')
    if not re.fullmatch(r'[a-f0-9]{64}',expected):raise RuntimeError('Current source identity missing')
    with tempfile.TemporaryDirectory(prefix='adajepa-restore-',dir=root.parent) as directory:
        source=Path(directory);restored=restore_checkpoints(repo,run,revision,source)
        lock=json.loads((source/'protocol.json').read_text())
        if lock['fingerprint']!=fingerprint() or lock.get('source_sha256')!=expected:
            raise RuntimeError('Resume protocol/source mismatch; do not mix versions')
        if not (source/'rendering.json').exists():raise RuntimeError('Resume lacks renderer identity')
        pairs=validate_episode_pairs(source,quarantine=True)
        parent_runtime=json.loads((source/'cloud_bootstrap_runtime.json').read_text())
        parent_rendering=json.loads((source/'rendering.json').read_text())
        for name in ('data','models','episodes','rows'):
            if (source/name).exists():shutil.copytree(source/name,root/name,dirs_exist_ok=True)
        shutil.copyfile(source/'rendering.json',root/'rendering.json')
    return dict(revision=revision,checkpoint=restored,pairs=pairs,
                parent_runtime=parent_runtime,parent_rendering=parent_rendering)


def validate_runtime_identity(previous,current):
    old=previous.get('after',{});new=current.get('after',{})
    keys=('python','torch','torchvision','cuda','protected')
    if any(key not in old or key not in new or old[key]!=new[key] for key in keys):
        raise RuntimeError('Recovered data runtime identity differs from current runtime')


def validate_renderer_identity(previous,current):
    def signatures(report):
        found=set()
        for attempt in report.get('attempts',[]):
            if attempt.get('ok') and attempt.get('backend')==report.get('backend'):
                for check in (attempt.get('result') or {}).get('checks',[]):
                    values=tuple(check.get(k) for k in ('vendor','renderer','version'))
                    if any(v is None or v=='None' for v in values):raise RuntimeError('Missing renderer identity')
                    found.add(values)
        return found
    if previous.get('backend')!=current.get('backend') or not signatures(previous) or signatures(previous)!=signatures(current):
        raise RuntimeError('Recovered data renderer identity differs from current renderer')


def queue_logs(root,store):
    for path in sorted((root/'logs').glob('*')):
        if path.is_file():store.push(path)


def finalize_success(root,store,publisher,result):
    queue_logs(root,store)
    if not drain_checkpoints(publisher,root):raise RuntimeError('Results remain local; remote finalization incomplete')
    # No completion marker can precede the verified results and trajectory packs.
    store.record('checkpoint_summary.json',dict(stats=publisher.stats,interval_seconds=120,min_interval_seconds=30,
        local_integrity_is_not_remote_durability=True))
    store.record('COMPLETED.json',dict(result,all_prior_artifacts_remotely_verified=True))
    if not drain_checkpoints(publisher,root):raise RuntimeError('Completion marker remote verification incomplete')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/work/results');p.add_argument('--repo',required=True);p.add_argument('--run',required=True)
    group=p.add_mutually_exclusive_group();group.add_argument('--resume');group.add_argument('--recovery-manifest')
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    store=Store(root,a.repo,a.run);publisher=BatchPublisher(root,a.repo,a.run,interval=120,min_interval=30);failed=None
    try:
        runtime_path=Path('cloud_bootstrap_runtime.json')
        runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else None
        parent_rendering=None
        if a.resume:
            resumed=restore(root,a.repo,a.resume)
            validate_runtime_identity(resumed['parent_runtime'],runtime or {})
            parent_rendering=resumed['parent_rendering']
            store.record('resume.json',dict(parent=a.resume,**resumed))
        elif a.recovery_manifest:
            from .recovery import restore_legacy_data
            manifest=json.loads(Path(a.recovery_manifest).read_text())
            validate_runtime_identity(manifest['bootstrap_runtime'],runtime or {})
            summary=restore_legacy_data(manifest,root)
            parent_rendering=manifest['rendering_metadata']
            store.record('data_recovery.json',dict(manifest=manifest,summary=summary))
        if a.resume or a.recovery_manifest:
            for directory in ('data','models','episodes','rows'):
                for path in sorted((root/directory).glob('*')):
                    if path.is_file():store.push(path)
        if runtime:store.record('cloud_bootstrap_runtime.json',runtime)
        store.record('protocol.json',dict(fingerprint=fingerprint(),config=CONFIG,source_sha256=os.environ.get('EXPERIMENT_SOURCE_SHA256'),
            train=scenes('train'),val=scenes('val'),test=scenes(),planned_test_episodes=len(scenes())*len(MODEL_SEEDS)*len(ARMS),
            checkpoint_policy=dict(interval_seconds=120,min_interval_seconds=30,one_publisher=True)))
        import torch
        if torch.cuda.device_count()!=4:raise RuntimeError('Expected exactly four GPUs')
        from .rendering import select_renderer
        rendering=select_renderer(root,required_backend=parent_rendering['backend'] if parent_rendering else None)
        if parent_rendering:validate_renderer_identity(parent_rendering,rendering)
        store.record('rendering.json',rendering);checkpoint_tick(publisher,root,force=True)
        common=['--root',str(root),'--repo',a.repo,'--run',a.run]
        run_group([[sys.executable,'-m','experiment.check_cloud',*common]],root,'preflight',[0],publisher,store)
        run_group([[sys.executable,'-m','experiment.train','--phase','collect','--worker',str(i),'--workers','4',*common] for i in range(4)],root,'collect',[0,1,2,3],publisher,store)
        run_group([[sys.executable,'-m','experiment.train','--phase','check-data',*common]],root,'data-quality',[0],publisher,store)
        run_group([[sys.executable,'-m','experiment.train','--phase','train','--seed',str(s),*common] for s in MODEL_SEEDS],root,'train',[0,1,2],publisher,store)
        run_group([[sys.executable,'-m','experiment.evaluate','--worker',str(i),'--workers','4',*common] for i in range(4)],root,'evaluate',[0,1,2,3],publisher,store)
        validate_episode_pairs(root)
        from .analysis import analyze
        result=analyze(root)
        if not result['confirmatory_allowed']:raise RuntimeError('Incomplete formal results')
        for path in sorted((root/'analysis').glob('*')):store.push(path)
        finalize_success(root,store,publisher,result)
    except Exception as exc:
        failed=exc;detail=scrub(traceback.format_exc());print(detail,flush=True)
        try:
            store.record('FAILED.json',dict(error_type=type(exc).__name__,detail=detail))
            queue_logs(root,store)
            if not drain_checkpoints(publisher,root,max_seconds=90):
                print('Checkpoint recovery incomplete; some files remain local. No success claim.',flush=True)
        except Exception as archive_exc:print('Failure archive error:',type(archive_exc).__name__,scrub(archive_exc),flush=True)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
