import json,os,subprocess
from types import SimpleNamespace
import pytest
from experiment import rendering

def fake_probe(calls,fail=()):
    def probe(backend,device,path):
        calls.append((backend,device))
        return dict(backend=backend,device=device,ok=(backend,device) not in fail)
    return probe

def test_egl_requires_every_device_and_sets_shared_backend(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(rendering,'_run_probe',fake_probe(calls))
    monkeypatch.setenv('MUJOCO_GL','old');monkeypatch.setenv('PYOPENGL_PLATFORM','old')
    report=rendering.select_renderer(tmp_path)
    assert calls==[('egl',i) for i in range(4)]
    assert report['backend']=='egl' and not report['fallback']
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl'

def test_one_egl_device_failure_selects_uniform_software(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(rendering,'_run_probe',fake_probe(calls,{('egl',2)}))
    monkeypatch.setenv('MUJOCO_GL','old');monkeypatch.setenv('PYOPENGL_PLATFORM','old')
    report=rendering.select_renderer(tmp_path)
    assert calls==[('egl',0),('egl',1),('egl',2),('osmesa',0)]
    assert report['backend']=='osmesa' and report['fallback']
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='osmesa'

def test_resume_never_changes_pinned_renderer(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(rendering,'_run_probe',fake_probe(calls,{('egl',0)}))
    monkeypatch.setenv('MUJOCO_GL','unchanged')
    with pytest.raises(RuntimeError,match='pinned'):rendering.select_renderer(tmp_path,'egl')
    assert calls==[('egl',0)] and os.environ['MUJOCO_GL']=='unchanged'

def test_software_resume_does_not_probe_egl(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(rendering,'_run_probe',fake_probe(calls))
    monkeypatch.setenv('MUJOCO_GL','old');monkeypatch.setenv('PYOPENGL_PLATFORM','old')
    report=rendering.select_renderer(tmp_path,'osmesa')
    assert calls==[('osmesa',0)] and report['backend']=='osmesa' and not report['fallback']

def test_no_renderer_blocks_formal_work(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(rendering,'_run_probe',fake_probe(calls,{('egl',0),('osmesa',0)}))
    with pytest.raises(RuntimeError,match='formal evaluation has not started'):rendering.select_renderer(tmp_path)
    assert calls==[('egl',0),('osmesa',0)]

def test_probe_uses_fresh_process_and_scrubs_token(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_TOKEN','SENSITIVE_TOKEN')
    def run(command,**kwargs):
        assert command[-4:]==['--probe','egl','--device','3']
        assert kwargs['env']['MUJOCO_EGL_DEVICE_ID']==kwargs['env']['CUDA_VISIBLE_DEVICES']=='3'
        assert kwargs['env']['MUJOCO_GL']==kwargs['env']['PYOPENGL_PLATFORM']=='egl'
        assert kwargs['timeout']==120
        return SimpleNamespace(returncode=0,stdout='SENSITIVE_TOKEN\n'+rendering.RESULT_PREFIX+json.dumps(dict(backend='egl',device=3)))
    monkeypatch.setattr(subprocess,'run',run)
    path=tmp_path/'probe.log';result=rendering._run_probe('egl',3,path)
    assert result['ok'] and 'SENSITIVE_TOKEN' not in path.read_text()

def test_probe_timeout_is_recorded_failure(tmp_path,monkeypatch):
    def run(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],120,output=b'partial')
    monkeypatch.setattr(subprocess,'run',run)
    path=tmp_path/'probe.log';result=rendering._run_probe('egl',0,path)
    assert not result['ok'] and result['returncode'] is None and path.read_text()=='partial'
