#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
# This interpreter only orchestrates setup. It never receives pip installs.
python - "$@" <<'PY_BOOTSTRAP'
import importlib.metadata,json,os,pathlib,shutil,subprocess,sys,time
WORK_ROOT=pathlib.Path('/work')
PREFIX='ADAJEPA_RUNTIME '
PROBE=r'''
import importlib.metadata as md,json,pathlib,sys,torch,torchvision
protected={}
for d in md.distributions():
    name=d.metadata.get('Name','').lower().replace('_','-')
    if name in ('torch','torchvision','torchaudio','triton') or name.startswith('nvidia-'):
        protected.setdefault(name,d.version)
print('ADAJEPA_RUNTIME '+json.dumps(dict(executable=sys.executable,python=list(sys.version_info[:2]),
 torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda=torch.version.cuda,
 cuda_available=torch.cuda.is_available(),devices=torch.cuda.device_count(),
 torch_file=torch.__file__,torchvision_file=torchvision.__file__,protected=protected,
 package_paths=[str(pathlib.Path(p).absolute()) for p in sys.path if p and pathlib.Path(p).name in ('site-packages','dist-packages') and pathlib.Path(p).is_dir()])))
'''

def valid_runtime(record):
    return (record.get('python')==[3,12]
        and record.get('torch','').split('+')[0]=='2.14.1'
        and record.get('torchvision','').split('+')[0]=='0.29.1'
        and record.get('cuda')=='13.0' and record.get('cuda_available') is True
        and record.get('devices')==4)

def run(command,phase,timeout=120,env=None):
    print('BOOTSTRAP',phase,flush=True)
    try:
        result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              timeout=timeout,env=env,check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(phase+' exceeded '+str(timeout)+' seconds') from exc
    output=result.stdout or '';token=os.environ.get('HF_TOKEN','')
    if token:output=output.replace(token,'[REDACTED]')
    if result.returncode:
        print(output[-5000:],flush=True)
        raise RuntimeError(phase+' failed with exit code '+str(result.returncode))
    return output

def clean_environment():
    env=os.environ.copy();env['PIP_REQUIRE_VIRTUALENV']='true'
    env.pop('PIP_BREAK_SYSTEM_PACKAGES',None);env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
    return env

def probe(interpreter,phase,env=None):
    output=run([interpreter,'-c',PROBE],phase,timeout=60,env=clean_environment() if env is None else env)
    records=[json.loads(line[len(PREFIX):]) for line in output.splitlines() if line.startswith(PREFIX)]
    if len(records)!=1 or not valid_runtime(records[0]):
        raise RuntimeError(phase+': pinned CUDA torch/torchvision and four GPUs required')
    return records[0]

def select_runtime():
    candidates=['/opt/venv/bin/python','/opt/conda/bin/python','/opt/pytorch/bin/python',sys.executable]
    candidates += [shutil.which('python'),shutil.which('python3')]
    seen=set();errors=[]
    for candidate in candidates:
        if not candidate or not pathlib.Path(candidate).is_file():continue
        identity=os.path.abspath(candidate)
        if identity in seen:continue
        seen.add(identity)
        try:return candidate,probe(candidate,'runtime_probe:'+candidate)
        except Exception as exc:errors.append(str(exc))
    raise RuntimeError('No compatible preinstalled GPU runtime; no package installation attempted. '+'; '.join(errors))

def inherit_runtime_paths(python,record,env):
    # A venv created from another venv otherwise inherits only its base Python.
    # Plain .pth paths are processed in our venv before base site-packages, while
    # packages installed into our venv keep precedence over the inherited paths.
    paths=record.get('package_paths')
    if not isinstance(paths,list) or not paths:
        raise RuntimeError('Selected runtime did not expose its package paths')
    code=r'''import json,pathlib,sys,sysconfig
paths=json.loads(sys.argv[1])
if any(not isinstance(p,str) or not pathlib.Path(p).is_absolute() or '\n' in p or '\r' in p for p in paths):
 raise RuntimeError('Invalid inherited runtime path')
target=pathlib.Path(sysconfig.get_path('purelib'))
if not target.is_relative_to(pathlib.Path(sys.prefix)):
 raise RuntimeError('Expected isolated venv site-packages')
(target/'adajepa_inherited_runtime.pth').write_text('\n'.join(paths)+'\n')
'''
    run([python,'-c',code,json.dumps(paths)],'inherit_runtime_paths',env=env)

def assert_runtime_unchanged(before,after):
    if (before['protected']!=after['protected'] or before['torch_file']!=after['torch_file']
        or before['torchvision_file']!=after['torchvision_file']):
        raise RuntimeError('Protected GPU runtime changed; formal work blocked')

def main(arguments):
    started=time.monotonic()
    env=clean_environment()
    selected,before=select_runtime()
    print('BOOTSTRAP selected_runtime',selected,flush=True)
    # Install OS libraries and venv support, never alter system Python packages.
    apt=shutil.which('apt-get')
    if apt:
        run([apt,'update','-qq'],'apt_update',timeout=180)
        run([apt,'install','-y','-qq','libegl1','libgl1','libglib2.0-0','libosmesa6','python3-venv'],
            'apt_libraries',timeout=300)
    runtime=WORK_ROOT/'adajepa-runtime-venv'
    if runtime.exists():raise RuntimeError('Runtime directory already exists; refusing unknown environment')
    run([selected,'-m','venv','--system-site-packages',str(runtime)],'create_isolated_venv',env=env)
    python=str(runtime/'bin'/'python')
    inherit_runtime_paths(python,before,env)
    isolated_before=probe(python,'isolated_runtime_before_install',env=env)
    assert_runtime_unchanged(before,isolated_before)
    constraint=WORK_ROOT/'adajepa-runtime-constraints.txt'
    constraint.write_text(''.join(name+'=='+version+'\n' for name,version in sorted(before['protected'].items())))
    pip=[python,'-m','pip','--isolated','--disable-pip-version-check','--require-virtualenv',
         'install','--constraint',str(constraint)]
    run(pip+['-r','requirements.txt'],'install_dependencies',timeout=900,env=env)
    run(pip+['--no-deps','robosuite==1.5.1'],'install_robosuite',timeout=180,env=env)
    after=probe(python,'runtime_after_install',env=env)
    assert_runtime_unchanged(before,after)
    report=dict(selected_interpreter=selected,isolated_interpreter=python,before=before,after=after,
                elapsed_seconds=time.monotonic()-started)
    pathlib.Path('cloud_bootstrap_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    run([python,'-m','pytest','-q','tests'],'software_tests',timeout=180,env=env)
    print('BOOTSTRAP ready; starting experiment driver',flush=True)
    # One process owns the experiment after setup; retain signal propagation.
    os.execve(python,[python,'-m','experiment.driver',*arguments],env)

if __name__=='__main__':
    try:main(sys.argv[1:])
    except Exception as exc:
        print('BOOTSTRAP FAILED:',type(exc).__name__,str(exc),flush=True)
        raise SystemExit(1)
PY_BOOTSTRAP
