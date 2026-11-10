"""Persist only already-completed original MOSS signed native responses."""
from dataclasses import fields
import math
from pathlib import Path
import torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow
from moss_sliders import MODEL_ID,MODEL_REVISION
from . import PROFILE,ownership
from .util import canonical,equal,read_packet,save_packet,source_record,tree

SCHEMA='MOSS-completed-native-signed-prepared-cache-fresh95-v1'
TEACHER_KEYS=('seed','resolution','rollout_steps','strength')

def teacher_configuration(config):return{k:config[k]for k in TEACHER_KEYS}

def file_record(row):
    if (not isinstance(row,dict)or set(row)!={'path','bytes','sha256'}
            or not isinstance(row['path'],str)or not row['path']or type(row['bytes'])is not int or row['bytes']<1
            or not Path(row['path']).is_absolute()or '..'in Path(row['path']).parts
            or not isinstance(row['sha256'],str)or len(row['sha256'])!=64
            or any(c not in'0123456789abcdef'for c in row['sha256'])):
        raise ValueError('explicit whole-file byte/SHA provenance required')

def validate_base(identity):
    if (not isinstance(identity,dict)or identity.get('model')!=MODEL_ID or identity.get('revision')!=MODEL_REVISION
            or identity.get('scope')not in('pretrained-native','synthetic-software-only')
            or not isinstance(identity.get('whole_files'),list)or not identity['whole_files']):
        raise ValueError('original model/revision and owner-authenticated whole files required')
    for record in identity['whole_files']:file_record(record)
    canonical(identity)

def finite(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad or value.layout!=torch.strided or value.is_complex()or not torch.isfinite(value).all():
            raise ValueError('frozen dense finite completed tensors required')
    elif isinstance(value,dict):
        if any(not isinstance(k,str)for k in value):raise ValueError('string cache keys required')
        for item in value.values():finite(item)
    elif isinstance(value,(tuple,list)):
        for item in value:finite(item)
    elif value is not None and type(value)not in(str,int,float,bool):raise ValueError('unsupported cached object')
    elif type(value)is float and not math.isfinite(value):raise ValueError('finite cache scalars required')

def validate_prepared(prepared,config):
    if not isinstance(prepared,(list,tuple))or len(prepared)!=2*config['rollout_steps']:
        raise ValueError('complete original paired-sign frozen row bank required')
    shape=None
    for index,item in enumerate(prepared):
        if type(item)is not PreparedResponse or type(item.row)is not ResponseRow:
            raise ValueError('original PreparedResponse/ResponseRow objects required')
        row=item.row
        if (row.schedule is not None or not isinstance(row.timestep,torch.Tensor)or row.timestep.numel()!=1
                or not 0<=float(row.timestep)<=1000 or row.scale!=(1.,-1.)[index%2]
                or row.strength!=config['strength']or row.guidance!=1.):
            raise ValueError('original MOSS paired signed raw-velocity contract differs')
        current=tuple(item.target.shape)
        if (len(current)!=4 or current[0]!=1 or current[1]<1
                or current[-2:]!=(config['resolution']//16,)*2 or shape is not None and shape!=current):
            raise ValueError('complete original MOSS [1,C,H/16,W/16] response geometry required')
        shape=current
        for value in(row.latent,item.target,item.neutral):
            if tuple(value.shape)!=shape or value.dtype!=torch.float32:raise ValueError('complete FP32 native velocity fields required')
        for context in(row.neutral,row.positive,row.negative):
            if not isinstance(context,tuple)or len(context)!=2:raise ValueError('original (embedding,mask) contexts required')
            embed,mask=context
            if (not isinstance(embed,torch.Tensor)or not isinstance(mask,torch.Tensor)or embed.ndim!=3
                    or embed.shape[0]!=1 or mask.shape!=embed.shape[:2]or embed.shape[-1]<1
                    or not embed.is_floating_point()or not torch.all((mask==0)|(mask==1))or not mask.any()):
                raise ValueError('full native token embeddings/binary mask required')
        for value in(row.neutral,row.positive,row.negative,row.latent,row.timestep,item.target,item.neutral):finite(value)
        if index:
            for name in('neutral','positive','negative'):
                if not equal(getattr(prepared[0].row,name),getattr(row,name)):
                    raise ValueError('all original rows use the same three declared contexts')
        if index%2:
            other=prepared[index-1].row
            for name in('neutral','positive','negative','latent','timestep','strength','guidance'):
                if not equal(getattr(other,name),getattr(row,name)):raise ValueError('signed rows must share exact original native inputs')
    return shape

def condition_features(prepared):
    """The exact original gan._build_run feature arithmetic; zero model calls."""
    result=[]
    for item in prepared:
        row=item.row;neutral=row.neutral
        pooled=(neutral[0].float()*neutral[1][...,None]).sum(1)
        pooled=pooled/neutral[1].sum(1,keepdim=True)
        pooled=pooled/pooled.square().mean().sqrt().clamp_min(1e-12)
        suffix=pooled.new_tensor([[float(row.timestep)/1000.,row.scale,row.strength,1./3.,0.]])
        result.append(torch.cat((pooled,suffix),dim=1))
    return torch.cat(result).detach()

def dump(item):
    return dict(row={f.name:tree(getattr(item.row,f.name),'cpu')for f in fields(ResponseRow)},target=tree(item.target,'cpu'),neutral=tree(item.neutral,'cpu'))

def restore(data,device):
    if not isinstance(data,dict)or set(data)!={'row','target','neutral'}or set(data['row'])!={f.name for f in fields(ResponseRow)}:
        raise ValueError('complete serialized original row required')
    return PreparedResponse(ResponseRow(**tree(data['row'],device)),tree(data['target'],device),tree(data['neutral'],device))

def origins_valid(origins,prepared):
    if not isinstance(origins,list)or len(origins)!=len(prepared):raise ValueError('every completed row needs original provenance')
    for origin,item in zip(origins,prepared):
        if (not isinstance(origin,dict)or origin.get('scope')not in('completed-native','synthetic-software-only')
                or origin.get('complete_prepared_output')is not True or origin.get('signed_scale')!=item.row.scale
                or not isinstance(origin.get('whole_output_files'),list)or not origin['whole_output_files']
                or not isinstance(origin.get('source_files'),list)or not origin['source_files']):
            raise ValueError('truthful complete row, whole-file and Source origins required')
        for record in origin['whole_output_files']+origin['source_files']:file_record(record)
    canonical(origins)

def save_prepared_cache(path,*,config,base_identity,prepared,teacher_origins,cache_id,owner_admission):
    """Caller-supplied completed outputs only; no collector/trajectory/teacher API.

    Whole-file provenance is authenticated by the actual resource/data owner.
    This serializer binds those original records without inventing a model proof.
    """
    from .api import configuration
    config=configuration(config)
    ownership.guard();validate_base(base_identity);validate_prepared(prepared,config);origins_valid(teacher_origins,prepared)
    expected_scope='completed-native'if base_identity['scope']=='pretrained-native'else'synthetic-software-only'
    if any(x['scope']!=expected_scope for x in teacher_origins):raise ValueError('synthetic/native origins must remain distinct')
    if not isinstance(cache_id,str)or not cache_id.strip()or not callable(owner_admission):raise ValueError('cache ID and external owner required')
    request=dict(operation='serialize-completed-cache',profile=PROFILE,cache_id=cache_id,
        base_identity=base_identity,teacher_origins=teacher_origins,model_or_teacher_calls=0)
    if owner_admission(request)is not True:raise ValueError('owner did not authenticate completed origins')
    return save_packet(path,dict(schema=SCHEMA,profile=PROFILE,cache_id=cache_id,
        teacher_configuration=teacher_configuration(config),base_identity=base_identity,
        original_sources=ownership.sources()['original'],serializer_source=source_record(__file__),
        teacher_origins=teacher_origins,rows=[dump(x)for x in prepared],conditions=condition_features(prepared).cpu(),
        complete=True,model_or_teacher_calls=0))

def load_cache(pin,*,config,base_identity):
    packet=read_packet(pin)
    keys={'schema','profile','cache_id','teacher_configuration','base_identity','original_sources','serializer_source','teacher_origins','rows','conditions','complete','model_or_teacher_calls'}
    if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=SCHEMA or packet['profile']!=PROFILE
            or packet['complete']is not True or packet['model_or_teacher_calls']!=0
            or packet['teacher_configuration']!=teacher_configuration(config)or packet['base_identity']!=base_identity
            or packet['original_sources']!=ownership.sources()['original']or packet['serializer_source']!=source_record(__file__)):
        raise ValueError('completed original cache identity/Source/configuration differs')
    prepared=[restore(x,'cpu')for x in packet['rows']];shape=validate_prepared(prepared,config);origins_valid(packet['teacher_origins'],prepared)
    expected_scope='completed-native'if base_identity['scope']=='pretrained-native'else'synthetic-software-only'
    if any(x['scope']!=expected_scope for x in packet['teacher_origins']):raise ValueError('cache provenance scope differs')
    finite(packet['conditions'])
    if not equal(condition_features(prepared),packet['conditions']):raise ValueError('original cached condition features differ')
    scale=float(torch.cat([(x.target-x.neutral).flatten()for x in prepared]).double().square().mean().sqrt())
    if not 0<scale<float('inf'):raise ValueError('finite nonzero original response scale required')
    identity={k:packet[k]for k in('schema','profile','cache_id','teacher_configuration','base_identity','original_sources','serializer_source','teacher_origins')}
    identity.update(whole_file=dict(path=pin.path,bytes=pin.bytes,sha256=pin.sha256),row_count=len(prepared),response_shape=list(shape),response_scale=scale)
    return prepared,packet['conditions'],identity

def to_device(prepared,device):return[restore(dump(x),device)for x in prepared]
