"""Public PG095 same-Parameter QR at original MOSS a=1 RMS; no bank."""
from dataclasses import dataclass
from importlib import metadata
import hashlib,math
import torch
from torch import nn
from particlegan import init as pg_init
from . import PROFILE
from .util import canonical,source_record

INIT_SHA='3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b'
SCHEMA='MOSS-explicit-host-a1-downQR-zeroUP-criticKEEP-initialization-v1'
MARKER='_moss_explicit_fresh95_receipt'

@dataclass(frozen=True)
class Construction:
    mode:str
    cohort:str
    options_sha256:str

class DeclaredOwned(nn.Module):
    def __init__(self,rows):
        super().__init__();self.specs={}
        for i,row in enumerate(rows):
            key='p'+str(i);self.register_parameter(key,row['parameter']);self.specs[key]=row['distribution']

pg_init.register(DeclaredOwned,lambda module:module.specs)

def tensor_sha(value):
    flat=torch.empty(value.numel(),dtype=value.dtype,device='cpu')
    flat.copy_(value.detach().reshape(-1).cpu())
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()

def seed_for(seed,name):
    return int.from_bytes(hashlib.sha256(f'{PROFILE}:{seed}:{name}'.encode()).digest()[:8],'big')%2**63

def declarations(slider,critic):
    from moss_sliders.lora import Slider
    from moss_sliders.native_hndl import HNDLLowRankResidual,MossHNDLFieldCritic
    from hndl.operators.linear import Linear
    if type(slider)is not Slider or type(critic)is not MossHNDLFieldCritic or not slider.residuals:
        raise ValueError('exact original MOSS HNDL adapter and spatial critic required')
    rows=[];seen=set();storages=set()
    for index,layer in enumerate(slider.residuals):
        if type(layer)is not HNDLLowRankResidual:raise ValueError('original HNDL residual class required')
        for role,leaf in(('down',layer.down),('up',layer.up)):
            if (type(leaf)is not Linear or leaf.equalized is not False or leaf.bias is not None
                    or any(leaf.children())or list(leaf.buffers())or nn.utils.parametrize.is_parametrized(leaf)
                    or set(leaf._parameters)!={'weight','bias'}
                    or tuple(leaf.weight.shape)!=(leaf.out_features,leaf.in_features)):
                raise ValueError('original non-equalized bias-free HNDL down/up required')
            p=leaf.weight
            if role=='up'and torch.count_nonzero(p).item():raise ValueError('genuine fresh zero-UP required')
            # kaiming_uniform_(a=1): gain1, U(+-sqrt(3/fan_in)), variance1/fan_in.
            bound=math.sqrt(3/leaf.in_features)
            rows.append(dict(name=f'adapter.residuals.{index}.{role}.weight',parameter=p,
                distribution=pg_init.Uniform(-bound,bound)if role=='down'else pg_init.KEEP,
                role='host-a1-down-QR'if role=='down'else'protected-zero-UP-KEEP'))
    rows.extend(dict(name='critic.'+name,parameter=p,distribution=pg_init.KEEP,
        role='original-HNDL-critic-constructor-KEEP')for name,p in critic.named_parameters(remove_duplicate=False))
    for row in rows:
        p=row['parameter'];storage=(str(p.device),p.untyped_storage().data_ptr())
        if (type(p)is not nn.Parameter or not p.requires_grad or p.grad is not None
                or p.dtype!=torch.float32 or not p.is_contiguous()or not torch.isfinite(p).all()
                or id(p)in seen or p.numel()and storage in storages or hasattr(p,MARKER)):
            raise ValueError('unique materialized fresh FP32 owned Parameters required')
        seen.add(id(p));storages.add(storage)
    adapter={id(p)for p in slider.parameters()};d={id(p)for p in critic.parameters()}
    if seen!=adapter|d or adapter&d:raise ValueError('declarations must cover only original adapter/critic; no bank')
    return rows

def initialize(slider,critic,construction,options,sources):
    if (not isinstance(construction,Construction)or construction.mode!='fresh'or not construction.cohort
            or construction.options_sha256!=hashlib.sha256(canonical(options).encode()).hexdigest()):
        raise ValueError('declared fresh construction before optimizers required')
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise ValueError('exact public PG095 initializer Source required')
    rows=declarations(slider,critic);before=[tensor_sha(x['parameter'])for x in rows]
    ids=[id(x['parameter'])for x in rows];holder=DeclaredOwned(rows)
    seeds={'p'+str(i):seed_for(options['config']['seed'],row['name'])for i,row in enumerate(rows)}
    pg_init.deterministic_orthogonal_(holder,strict=True,parameter_seeds=seeds)
    if ids!=[id(x['parameter'])for x in rows]:raise RuntimeError('owned Parameter replaced')
    parameters=[]
    for i,row in enumerate(rows):
        after=tensor_sha(row['parameter'])
        if row['distribution']is pg_init.KEEP and before[i]!=after:raise RuntimeError('zero-UP/constructor KEEP changed')
        parameters.append(dict(name=row['name'],role=row['role'],seed=seeds['p'+str(i)],
            shape=list(row['parameter'].shape),dtype=str(row['parameter'].dtype),distribution=repr(row['distribution']),
            before_sha256=before[i],after_sha256=after))
    receipt=dict(schema=SCHEMA,profile=PROFILE,cohort=construction.cohort,options_sha256=construction.options_sha256,
        completed_steps=0,restored_from=None,source_identity=sources,seed=options['config']['seed'],
        initializer=source_record(pg_init.__file__),parameters=parameters,parameter_identity_preserved=True,
        particle_bank=None,critic_policy='original-constructor-KEEP',host_down_kaiming_a=1,
        host_down_target_mean_square='1/fan_in',selection='raw',ema_decay=None)
    marker=hashlib.sha256(canonical(receipt).encode()).hexdigest()
    for row in rows:setattr(row['parameter'],MARKER,marker)
    return receipt

def validate_receipt(receipt,construction,options,sources):
    if (not isinstance(receipt,dict)or receipt.get('schema')!=SCHEMA or receipt.get('profile')!=PROFILE
            or receipt.get('cohort')!=construction.cohort or receipt.get('options_sha256')!=construction.options_sha256
            or receipt.get('source_identity')!=sources or receipt.get('seed')!=options['config']['seed']
            or receipt.get('completed_steps')!=0 or receipt.get('restored_from')is not None
            or receipt.get('particle_bank')is not None or receipt.get('ema_decay')is not None
            or receipt.get('selection')!='raw'or receipt.get('initializer',{}).get('sha256')!=INIT_SHA):
        raise ValueError('exact original new-profile initialization receipt required')
