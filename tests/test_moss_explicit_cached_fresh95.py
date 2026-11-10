"""Four NEW authored/UNRUN MOSS software nodes; ONE total tiny update.

Original source guards, real tiny original MOSS architecture, original HNDL
modules and actual PG095 game execute. Synthetic complete fields replace only
data collection. No original proof, teacher or pretrained runtime is called.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import pytest
import torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow,FrozenResponseAdapter
from moss_sliders import MODEL_ID,MODEL_REVISION,gan,training as old_training
from moss_sliders.smoke import tiny_model
from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline,MOSSImageScheduler
from moss_sliders.fresh95 import open_training,open_inference,save_prepared_cache,file_pin,Checkpoint
from moss_sliders.fresh95 import api,cache,policy
from moss_sliders.fresh95.util import equal

def component_factory(request):
    assert request['operation']in('fresh','resume','inference')and request['particle_bank']is None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        model=tiny_model()
    return MOSSImagePipeline(model,MOSSImageScheduler())

@pytest.fixture
def staged(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('no original collectors/teachers/pretrained runtime in new software cases')
    for name in('build_run','_build_run','build_cached_run','prepare_cached_rows'):
        monkeypatch.setattr(gan,name,forbidden)
    monkeypatch.setattr(old_training,'sample_training_state',forbidden)
    monkeypatch.setattr(FrozenResponseAdapter,'prepare',forbidden)
    config=dict(objective='pg09_gan_only',pg_version='0.9.5',rank=2,seed=7,resolution=32,
        rollout_steps=2,strength=1.,batch_size=2,steps=3,lr=.001,d_lr_mult=18.,critic_width=4)
    base=component_factory(dict(operation='fresh',particle_bank=None))
    base_state=deepcopy(base.transformer.state_dict());base_path=tmp_path/'synthetic-original-base.pt'
    torch.save(base_state,base_path);bp=file_pin(base_path)
    identity=dict(model=MODEL_ID,revision=MODEL_REVISION,scope='synthetic-software-only',
        whole_files=[dict(path=bp.path,bytes=bp.bytes,sha256=bp.sha256)],constructor_seed=123)
    embedding=torch.linspace(.1,.6,48).reshape(1,3,16);mask=torch.ones(1,3,dtype=torch.bool)
    contexts=tuple((embedding+delta,mask.clone())for delta in(0.,.2,-.1))
    prepared=[]
    for position in range(2):
        z=torch.linspace(-.3,.3,16).reshape(1,4,2,2)+position*.01
        for sign in(1.,-1.):
            row=ResponseRow(*deepcopy(contexts),z.clone(),timestep=torch.tensor(1000.-position*500.),scale=sign,strength=1.,guidance=1.)
            prepared.append(PreparedResponse(row,torch.full_like(z,.2*sign),torch.zeros_like(z)))
    completed_path=tmp_path/'synthetic-completed-four-fields.pt'
    torch.save([cache.dump(x)for x in prepared],completed_path);whole=file_pin(completed_path);source=file_pin(__file__)
    origins=[dict(scope='synthetic-software-only',complete_prepared_output=True,signed_scale=x.row.scale,
        whole_output_files=[dict(path=whole.path,bytes=whole.bytes,sha256=whole.sha256)],
        source_files=[dict(path=source.path,bytes=source.bytes,sha256=source.sha256)],
        real_teacher_calls=0,synthetic_row=index)for index,x in enumerate(prepared)]
    frozen=save_prepared_cache(tmp_path/'completed-cache.pt',config=config,base_identity=identity,
        prepared=prepared,teacher_origins=origins,cache_id='NEW-synthetic-MOSS-paired-four-fields',owner_admission=lambda request:True)
    return dict(config=config,cache=frozen,base_identity=identity,device='cpu',cohort_id='NEW-MOSS-cached-fresh95-software-v1',
        component_factory=component_factory,owner_admission=lambda request:True),tmp_path,base_state

def opened(staged,name='fresh'):
    options,root,_=staged
    return open_training(directory=root/name,mode='fresh',**options)

def test_new_before_optimizer_same_HNDL_Parameters_zeroUP_critic_KEEP_no_bank(staged,monkeypatch):
    original=policy.initialize;game=api.ConditionalPG095;seen=[]
    def initialize(slider,critic,*args):
        ids=[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()]
        before=deepcopy(critic.state_dict());receipt=original(slider,critic,*args)
        assert ids==[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()]
        assert equal(before,critic.state_dict())and all(torch.count_nonzero(x.up.weight)==0 for x in slider.residuals)
        seen.append(receipt);return receipt
    def construct(backend,*args,**kwargs):
        assert seen and backend.metadata()['fresh_initialization']['receipt']==seen[-1]
        assert kwargs['particles']is None
        return game(backend,*args,**kwargs)
    monkeypatch.setattr(policy,'initialize',initialize);monkeypatch.setattr(api,'ConditionalPG095',construct)
    with opened(staged)as owner:
        assert owner.game.completed_steps==0 and owner.game.prior is None
        assert owner.game.state_dict()['config']['backend']['fresh_initialization']['receipt']==seen[0]
        assert (owner.directory/'FULL-000000.pt').is_file()and(owner.directory/'RAW-000000.pt').is_file()
        assert {id(p)for p in owner.native.network.parameters()}=={id(p)for p in owner.slider.parameters()}
        for residual in owner.slider.residuals:
            assert residual._graph['down']is residual.down and residual._graph['up']is residual.up

def test_new_a1_QR_Gram_deterministic_KEEP_caller_RNG_and_frozen_base(staged):
    before=torch.get_rng_state().clone();options,_,base=staged
    with opened(staged,'a')as owner:
        first=deepcopy(owner.native.network.state_dict());receipt=deepcopy(owner.initialization)
        pipe=owner.pipe;flags=owner.attachment.flags
        for residual in owner.slider.residuals:
            down=residual.down.weight.detach();rows,cols=down.shape;variance=1/residual.down.in_features
            gram=down@down.T if rows<=cols else down.T@down
            assert torch.allclose(gram,torch.eye(min(rows,cols))*variance*max(rows,cols),rtol=1e-5,atol=1e-6)
            assert math.isclose(float(down.square().mean()),variance,rel_tol=1e-5,abs_tol=1e-7)
            assert torch.count_nonzero(residual.up.weight)==0
        assert equal(base,pipe.transformer.state_dict())
    assert all(p.requires_grad==flag for p,flag in flags)
    assert all('_moss_slider_attached'not in vars(m)for m in pipe.transformer.modules())
    assert all('_pg09_scale_checkpoint'not in vars(m)for m in pipe.transformer.layers)
    assert torch.equal(before,torch.get_rng_state())
    with opened(staged,'b')as again:
        assert equal(first,again.native.network.state_dict())and receipt==again.initialization
        assert all(x['before_sha256']==x['after_sha256']for x in receipt['parameters']if x['role']!='host-a1-down-QR')
    assert torch.equal(before,torch.get_rng_state())

def test_new_ONE_update_strict_whole_FULL_opts_RNG_and_RAW_native_skip(staged,monkeypatch):
    options,root,base=staged
    with opened(staged)as trained:
        trained.update()  # The ONE total new tiny random-model software update.
        full=trained.save(root/'saved-FULL1.pt');raw=trained.export(root/'saved-RAW1.pt')
        state=deepcopy(trained.game.state_dict());network=deepcopy(trained.native.network.state_dict());row=trained.game.rows[0].row
        pipe=trained.pipe
    assert equal(base,pipe.transformer.state_dict())
    def forbidden(*args,**kwargs):raise AssertionError('resume/inference cannot initialize or collect teachers')
    monkeypatch.setattr(policy,'initialize',forbidden)
    with open_training(directory=root/'resume',mode='resume',resume=full,**options)as resumed:
        assert resumed.inherited_steps==1 and equal(state,resumed.game.state_dict())
    monkeypatch.setattr(api,'load_cache',forbidden);monkeypatch.setattr(api,'ConditionalPG095',forbidden)
    with open_inference(raw,component_factory=component_factory,owner_admission=lambda request:True)as inferred:
        assert not hasattr(inferred,'game')and not hasattr(inferred,'critic')
        assert equal(network,inferred.native.network.state_dict())
        before=torch.get_rng_state().clone();out=inferred.response(row,strength=1.,seed=17)
        assert out.shape==(1,4,2,2)and torch.isfinite(out).all()and torch.equal(before,torch.get_rng_state())
        assert equal(base,inferred.pipe.transformer.state_dict())

def test_new_early_mode_FULL_EMA_refusal_and_retry_only_pending_cleanup(staged,monkeypatch):
    options,root,_=staged;calls=[]
    def forbidden(request):calls.append(request);raise AssertionError('bad intent reached component factory')
    bad=dict(options,component_factory=forbidden)
    for mode,resume,ema in(('fresh',Checkpoint('unused',1,'0'*64),None),('resume',None,None),('fresh',None,.9)):
        with pytest.raises(ValueError):open_training(directory=root/'bad',mode=mode,resume=resume,ema_decay=ema,**bad)
    legacy=root/'old-FULL.pt';torch.save(dict(schema='old-moss-pg09-checkpoint-v2'),legacy)
    with pytest.raises(ValueError):open_training(directory=root/'legacy',mode='resume',resume=file_pin(legacy),**bad)
    with pytest.raises(ValueError):
        open_training(directory=root/'denied',mode='fresh',**dict(options,component_factory=forbidden,owner_admission=lambda request:False))
    assert calls==[]
    owner=opened(staged,'cleanup');handle=owner.attachment.pending_hooks[0];remove=handle.remove;attempts=[]
    def once():
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('deliberate fail-once owned hook removal')
        return remove()
    monkeypatch.setattr(handle,'remove',once)
    with pytest.raises(RuntimeError,match='fail-once'):owner.close()
    assert owner.closing and not owner.closed and owner.attachment.pending_hooks==[handle]
    with pytest.raises(RuntimeError,match='cleanup'):owner.update()
    with pytest.raises(RuntimeError,match='cleanup'):owner.response(None)
    owner.close();assert owner.closed and len(attempts)==2
    # A foreign forward replacement is retained; the original owned wrapper can
    # be restored by its owner and cleanup retried without repeating removals.
    foreign_owner=opened(staged,'foreign');block=foreign_owner.pipe.transformer.layers[0]
    owned=vars(block)['forward'];foreign=lambda *args,**kwargs:None;block.forward=foreign
    with pytest.raises(RuntimeError,match='ownership changed'):foreign_owner.close()
    assert block.forward is foreign and foreign_owner.closing and not foreign_owner.closed
    block.forward=owned;foreign_owner.close();assert foreign_owner.closed
