"""Choose one tested renderer before importing physics in any long-lived worker."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

RESULT_PREFIX='RENDER_PROBE_RESULT '

def _run_probe(backend,device,log_path):
    env=os.environ.copy()
    env.update(MUJOCO_GL=backend,PYOPENGL_PLATFORM=backend,
               CUDA_VISIBLE_DEVICES=str(device),MUJOCO_EGL_DEVICE_ID=str(device),
               OMP_NUM_THREADS='2',NUMBA_NUM_THREADS='2',MKL_NUM_THREADS='2',LP_NUM_THREADS='2')
    started=time.monotonic()
    command=[sys.executable,'-m','experiment.rendering','--probe',backend,'--device',str(device)]
    try:
        p=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         text=True,timeout=120,check=False)
        output=p.stdout;code=p.returncode;reason=None
    except subprocess.TimeoutExpired as exc:
        output=exc.stdout or ''
        if isinstance(output,bytes):output=output.decode(errors='replace')
        code=None;reason='render probe exceeded 120 seconds'
    token=os.environ.get('HF_TOKEN','')
    if token:output=output.replace(token,'[REDACTED]')
    log_path.write_text(output)
    result=None
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:result=json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:pass
    ok=code==0 and isinstance(result,dict) and result.get('backend')==backend and result.get('device')==device
    return dict(backend=backend,device=device,ok=ok,returncode=code,
                seconds=time.monotonic()-started,log=log_path.name,
                reason=reason or (None if ok else 'probe failed; see saved log'),result=result)

def select_renderer(root,required_backend=None):
    """Prefer EGL on every GPU, else use OSMesa uniformly; pin resumes explicitly.

    Model computation still uses CUDA. This function only chooses rasterization.
    It never changes the backend after returning or after a required backend fails.
    """
    if required_backend not in (None,'egl','osmesa'):
        raise ValueError('Unsupported required rendering backend')
    root=Path(root);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    attempts=[]
    for backend in ([required_backend] if required_backend else ['egl','osmesa']):
        passed=True
        # EGL is device-specific; software rendering has no GPU dependency.
        for device in (range(4) if backend=='egl' else [0]):
            probe=_run_probe(backend,device,logs/f'rendering-{backend}-{device}.log')
            attempts.append(probe)
            if not probe['ok']:
                passed=False;break
        if passed:
            os.environ['MUJOCO_GL']=backend
            os.environ['PYOPENGL_PLATFORM']=backend
            os.environ['LP_NUM_THREADS']='2'
            report=dict(backend=backend,required_backend=required_backend,attempts=attempts,
                        model_compute='cuda',selection_scope='all stages, methods and workers',
                        fallback=required_backend is None and backend=='osmesa')
            print('RENDERER',backend,'; CUDA model computation unchanged',flush=True)
            return report
    raise RuntimeError('No working renderer'+(' for pinned backend '+required_backend if required_backend else '')+
                       '; inspect rendering-*.log; formal evaluation has not started')

def _probe(backend,device):
    # Executed only in a fresh subprocess, where OpenGL platform selection is safe.
    import numpy as np
    from .env import RobotEnv
    from .spec import CONFIG
    from OpenGL import GL
    results=[]
    for i,task in enumerate(CONFIG['tasks']):
        env=RobotEnv(task,91000+i)
        try:
            frame=env.image()
            if frame.shape!=(128,128,3) or not np.isfinite(frame).all() or float(frame.std())<=1:
                raise RuntimeError('Invalid or blank RGB frame for '+task)
            def gl_text(key):
                value=GL.glGetString(key)
                return value.decode(errors='replace') if isinstance(value,bytes) else str(value)
            results.append(dict(task=task,shape=list(frame.shape),pixel_std=float(frame.std()),
                                vendor=gl_text(GL.GL_VENDOR),renderer=gl_text(GL.GL_RENDERER),
                                version=gl_text(GL.GL_VERSION)))
        finally:env.close()
    print(RESULT_PREFIX+json.dumps(dict(backend=backend,device=device,checks=results)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',choices=['egl','osmesa'],required=True)
    parser.add_argument('--device',type=int,required=True);args=parser.parse_args()
    _probe(args.probe,args.device)
