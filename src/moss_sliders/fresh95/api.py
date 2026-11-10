"""Opt-in cached native MOSS owners; no old builder or teacher collection."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib,math
from pathlib import Path
import torch
from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.pg095 import ConditionalPG095
from particle_sliders.pg09_backends import MossResponseAdapter
from moss_sliders import MODEL_ID,MODEL_REVISION
from moss_sliders.learning_rates import recipe_overrides
from . import PROFILE,ownership,policy
from .cache import load_cache,to_device,validate_base
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json

FULL_SCHEMA='MOSS-native-HNDL-explicit-cached-fresh95-v1-FULL'
RAW_SCHEMA='MOSS-native-HNDL-explicit-cached-fresh95-v1-RAW'

def configuration(value):
    if not isinstance(value,dict):raise ValueError('explicit original configuration required')
    allowed={'objective','pg_version','rank','seed','resolution','rollout_steps','strength','batch_size','steps','lr','critic_width','d_lr_mult'}
    if set(value)-allowed or value.get('objective')!='pg09_gan_only'or value.get('pg_version')!='0.9.5':
        raise ValueError('exact original native PG095 adapter-only recipe required')
    for key,lower,upper in(('rank',1,128),('batch_size',2,512),('rollout_steps',2,50),('steps',1,10000),('seed',0,2**32-1),('resolution',16,2048),('critic_width',1,4096)):
        if type(value.get(key))is not int or not lower<=value[key]<=upper:raise ValueError('invalid recipe: '+key)
    if value['resolution']%16 or type(value.get('strength'))not in(int,float)or not math.isfinite(value['strength'])or value['strength']<=0:
        raise ValueError('original resolution and positive finite strength required')
    recipe_overrides(value)  # Unchanged explicit rate/no-particle-prior policy.
    canonical(value)
    return deepcopy(value)

def intent(mode,resume,cohort,ema_decay):
    if mode not in('fresh','resume')or not isinstance(cohort,str)or not cohort.strip():raise ValueError('explicit fresh|resume and new cohort required')
    if (mode=='fresh'and resume is not None)or(mode=='resume'and resume is None):raise ValueError('fresh refuses state; resume requires typed new FULL')
    if ema_decay is not None:raise ValueError('no weight EMA; RAW current weights only')

def devices_for(device):
    device=torch.device(device)
    if device.type not in('cpu','cuda')or device.type=='cuda'and device.index is None:
        raise ValueError('explicit cpu or indexed CUDA device required; no admission by library')
    return device,[device.index]if device.type=='cuda'else[]

def pipeline_check(pipe,device,cache_identity=None):
    from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline,MOSSImageTransformer
    if type(pipe)is not MOSSImagePipeline or type(pipe.transformer)is not MOSSImageTransformer or pipe.transformer.device!=device:
        raise ValueError('actual original MOSS pipeline/transformer on declared owner device required')
    if cache_identity is not None:
        shape=cache_identity['response_shape']
        if pipe.transformer.config.latent_channels!=shape[1]:raise ValueError('native cached channel geometry differs')

class ReceiptBackend:
    def __init__(self,native,initialization,sources,cache_identity):
        self.backend=native;self.network=native.network
        self.initialization=deepcopy(initialization);self.sources=deepcopy(sources);self.cache=deepcopy(cache_identity)
    def __getattr__(self,name):return getattr(self.backend,name)
    def metadata(self):
        return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,
            receipt=deepcopy(self.initialization),sources=deepcopy(self.sources)),
            completed_native_cache=deepcopy(self.cache),particle_bank=None,
            old_teacher_collector_called=False,old_legacy091_cached_builder_called=False)

@contextmanager
def observed(owner,seed):
    from particle_sliders.lifecycle import _network_observation
    owner.require_open()
    with _network_observation(owner.native.network,seed,devices=owner.devices):yield

class NativeOwner(ownership.Owner):
    def response(self,row,*,strength=1.,seed=7):
        """Native response diagnostic; never a teacher or scoring API."""
        with observed(self,seed),torch.no_grad():return self.native._response(row,scale=strength)
    def render(self,prompt,*,scale=1.,seed=7,**options):
        # Actual original preflight/encoding/scheduler/conditional CFG/VAE route.
        # The resource owner must supply all staged native components.
        from moss_sliders.inference import render
        with observed(self,seed):
            return render(self.pipe,self.slider,prompt,scale=scale,seed=seed,
                response_backend=self.native,**options)

class Training(NativeOwner):
    def __init__(self,config,cache,directory,*,mode,cohort_id,base_identity,device,
                 component_factory,owner_admission,resume=None,checkpointing=True,ema_decay=None):
        super().__init__()
        # Intent, exact FULL/cache and all original guards precede the model
        # factory, adapter, critic, optimizer and fresh initializer.
        intent(mode,resume,cohort_id,ema_decay)
        if type(checkpointing)is not bool or not callable(component_factory)or not callable(owner_admission):
            raise ValueError('explicit owner factory/admission/checkpoint choice required')
        config=configuration(config);validate_base(base_identity)
        self.device,self.devices=devices_for(device);self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable owner directory required')
        self.runtime_identity=ownership.guard();self.sources=ownership.sources()
        self.config,self.cohort_id,self.base_identity=config,cohort_id,deepcopy(base_identity)
        self.options=dict(config=config,base_identity=base_identity,device=str(self.device),checkpointing=checkpointing)
        self.construction=policy.Construction(mode,cohort_id,hashlib.sha256(canonical(self.options).encode()).hexdigest())
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','game','ema_decay'}
            if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA
                    or packet['profile']!=PROFILE or packet['cohort_id']!=cohort_id
                    or packet['options']!=self.options or packet['sources']!=self.sources or packet['ema_decay']is not None):
                raise ValueError('exact new-profile typed FULL required; no old artifact migration')
            policy.validate_receipt(packet['initialization'],self.construction,self.options,self.sources)
        prepared,conditions,self.cache_identity=load_cache(cache,config=config,base_identity=base_identity)
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('whole original prepared cache differs')
        request=dict(operation=mode,profile=PROFILE,cohort_id=cohort_id,options=deepcopy(self.options),
            completed_cache=deepcopy(self.cache_identity),new_teacher_or_trajectory_calls=0,
            fresh_PG_initialization_calls=int(mode=='fresh'),particle_bank=None,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('resource/data owner did not admit this explicit operation')
        try:
            with torch.random.fork_rng(devices=self.devices):
                self.pipe=component_factory(deepcopy(request));pipeline_check(self.pipe,self.device,self.cache_identity)
                if prepared[0].row.neutral[0].shape[-1]!=self.pipe.transformer.config.text_dim:
                    raise ValueError('full cached text-conditioning width differs from native model')
                # Match the original adapter/critic constructor law. This is a
                # NEW cached cohort, not a reconstruction of old teacher RNG.
                seed_initialization_rng(config['seed'],self.devices)
                self.attachment=ownership.Attachment(self.pipe.transformer)
                self.slider=self.attachment.attach(config['rank'])
                identity=dict(base_identity,network_runtime=self.runtime_identity)
                if hasattr(self.pipe,'_moss_training_context_identity'):
                    if base_identity.get('conditioning')!=self.pipe._moss_training_context_identity:
                        raise ValueError('owner-bound original conditioning identity differs')
                    identity['conditioning']=deepcopy(self.pipe._moss_training_context_identity)
                self.native=MossResponseAdapter(self.pipe.transformer,self.slider,model_identity=identity)
                from moss_sliders.native_hndl import MossHNDLFieldCritic
                self.critic=MossHNDLFieldCritic(self.cache_identity['response_shape'],conditions.shape[1],config['critic_width']).to(self.device)
                self.initialization=(policy.initialize(self.slider,self.critic,self.construction,self.options,self.sources)
                    if mode=='fresh'else deepcopy(packet['initialization']))
                if checkpointing:self.attachment.checkpoint(self.native.network)
            prepared=to_device(prepared,self.device);conditions=conditions.to(self.device)
            self.backend=ReceiptBackend(self.native,self.initialization,self.sources,self.cache_identity)
            # All fresh policies/receipt are already in the immutable signature
            # before the shared core constructs the first optimizer.
            self.game=ConditionalPG095(self.backend,prepared,self.critic,conditions=conditions,particles=None,
                response_scale=self.cache_identity['response_scale'],response_domain='continuous_field',
                preset='bispf_optimized',sampling='balanced',batch_size=config['batch_size'],
                total_steps=config['steps'],seed=config['seed'],recipe_overrides=recipe_overrides(config))
            if self.game.prior is not None:raise RuntimeError('MOSS adapter-only lane must not gain a bank')
            if packet is not None:
                self.game.load_state_dict(packet['game'])
                if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('typed whole game/optimizers/cursors/RNG restore differs')
            self.inherited_steps=self.game.completed_steps
            new_directory(self.directory);write_json(self.directory/'initialization.json',self.initialization)
            self.entry_FULL=self.save(self.directory/f'FULL-{self.inherited_steps:06d}.pt')
            self.entry_RAW=self.export(self.directory/f'RAW-{self.inherited_steps:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('MOSS entry and pending cleanup failed',[primary,cleanup])
            raise
    def update(self):self.require_open();return self.game.step()
    def save(self,path):
        self.require_open()
        return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,
            options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open();state=self.game.export_state_dict()
        return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,
            options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),model_identity=deepcopy(self.native.model_identity),
            network=state['network'],completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))

class Inference(NativeOwner):
    def __init__(self,raw,*,component_factory,owner_admission):
        super().__init__()
        # RAW metadata is authenticated before any added/native component.
        packet=read_packet(raw)
        keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','model_identity','network','completed_steps','selection','ema_decay'}
        if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=RAW_SCHEMA
                or packet['profile']!=PROFILE or packet['selection']!='raw'or packet['ema_decay']is not None
                or type(packet['completed_steps'])is not int or packet['completed_steps']<0
                or not callable(component_factory)or not callable(owner_admission)):
            raise ValueError('exact typed new-profile RAW and external owner required; no old migration')
        identity=ownership.guard();sources=ownership.sources()
        if packet['sources']!=sources:raise ValueError('RAW native/profile Source differs')
        options=packet['options'];config=configuration(options['config']);validate_base(options['base_identity'])
        construction=policy.Construction('inference',packet['cohort_id'],hashlib.sha256(canonical(options).encode()).hexdigest())
        policy.validate_receipt(packet['initialization'],construction,options,sources)
        if (packet['model_identity']!=dict(options['base_identity'],network_runtime=identity)
                or packet['cache_identity']['base_identity']!=options['base_identity']
                or packet['completed_steps']>config['steps']):
            raise ValueError('original native/cache/model identity or saved update budget differs')
        self.device,self.devices=devices_for(options['device']);self.sources=sources;self.initialization=deepcopy(packet['initialization'])
        request=dict(operation='inference',profile=PROFILE,cohort_id=packet['cohort_id'],options=deepcopy(options),
            model_identity=deepcopy(packet['model_identity']),cache_metadata_only=deepcopy(packet['cache_identity']),
            new_teacher_or_trajectory_calls=0,fresh_PG_initialization_calls=0,particle_bank=None,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('resource/data owner did not admit native RAW inference')
        try:
            with torch.random.fork_rng(devices=self.devices):
                self.pipe=component_factory(deepcopy(request));pipeline_check(self.pipe,self.device,packet['cache_identity'])
                if (hasattr(self.pipe,'_moss_training_context_identity')
                        and options['base_identity'].get('conditioning')!=self.pipe._moss_training_context_identity):
                    raise ValueError('original owner conditioning identity differs')
                seed_initialization_rng(config['seed'],self.devices)
                self.attachment=ownership.Attachment(self.pipe.transformer);self.slider=self.attachment.attach(config['rank'])
                self.native=MossResponseAdapter(self.pipe.transformer,self.slider,model_identity=packet['model_identity'])
                self.native.network.load_state_dict(packet['network'],strict=True)
                if not equal(self.native.network.state_dict(),packet['network']):raise RuntimeError('strict native RAW tensor restore differs')
            self.completed_steps=packet['completed_steps']
            # No critic, game, optimizer, teacher cache load or PG initializer.
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('MOSS RAW entry and cleanup failed',[primary,cleanup])
            raise
