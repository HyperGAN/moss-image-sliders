"""Original guards and retryable ownership of only this entry's attachments."""
from pathlib import Path
from .util import source_record

def guard():
    from moss_sliders import native_hndl
    from particlegan import init as pg_init
    from importlib import metadata
    from .baseline import ORIGINAL_SOURCES
    from .policy import INIT_SHA
    root=Path(native_hndl.__file__).absolute().parent
    for name,pin in ORIGINAL_SOURCES.items():
        if source_record(root/name)!=pin:
            raise RuntimeError('captured original MOSS Source differs: '+name)
    identity=native_hndl.verify_runtime()  # Exact original recursive admission.
    if metadata.version('particlegan')!='0.9.5' or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise RuntimeError('exact public PG095 initializer required')
    return identity

def sources():
    from moss_sliders import native_hndl,lora,gan,inference,training,learning_rates,backend
    from moss_sliders.vendor import pipeline_moss_image
    from particle_sliders import pg09,pg095,pg09_backends,lifecycle
    modules=(native_hndl,lora,gan,inference,training,learning_rates,backend,pipeline_moss_image,
             pg09,pg095,pg09_backends,lifecycle)
    return dict(own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))},
        original={m.__name__:source_record(m.__file__)for m in modules})

class Attachment:
    """Retain original Modules/Parameters and retry only unfinished cleanup.

    Never close, move or unload the caller's pipeline. Different-owner forward
    or marker replacements are refused and left intact, not overwritten.
    """
    def __init__(self,transformer):
        self.transformer=transformer;self.closed=self.closing=False
        self.slider=None;self.pending_hooks=[];self.pending_checkpoints=[]
        self.pending_markers=[];self.pending_flags=[];self.pending_modes=[]
        self.flags=[(p,p.requires_grad)for p in transformer.parameters()]
        self.modes=[(m,m.training)for m in transformer.modules()]
        self.pending_flags=list(self.flags);self.pending_modes=list(self.modes)
        self.markers={id(m):('_moss_slider_attached'in vars(m),vars(m).get('_moss_slider_attached'))for m in transformer.modules()}
        if any(getattr(m,'_moss_slider_attached',False)for m in transformer.modules()):
            raise ValueError('a foreign MOSS slider is already attached')
        if any(getattr(m,'_pg09_scale_checkpoint',False)for m in transformer.layers):
            raise ValueError('a foreign scale checkpoint is already attached')
    def attach(self,rank):
        from moss_sliders.lora import Slider
        from moss_sliders.native_hndl import HNDLLowRankResidual
        # Slider's exact constructor/host laws; cleanup is owned separately.
        # Keep the partial original instance reachable even if its constructor
        # fails after freezing base flags or attaching one of its own hooks.
        self.slider=Slider.__new__(Slider)
        try:Slider.__init__(self.slider,self.transformer,rank=rank,residual_factory=HNDLLowRankResidual)
        finally:
            self.pending_hooks=list(getattr(self.slider,'_handles',()))
            self.pending_markers=[(m,*self.markers[id(m)])for m in getattr(self.slider,'_modules_attached',())]
        self.named_ids=[(n,id(p))for n,p in self.slider.named_parameters()]
        return self.slider
    def checkpoint(self,network):
        from particle_sliders.pg09_backends import checkpoint_with_adapter_scale
        for block in self.transformer.layers:
            self.pending_checkpoints.append(checkpoint_with_adapter_scale(block,network))
    def require_active(self):
        if self.closed or self.closing:raise RuntimeError('MOSS attachment closed/cleanup pending')
        if self.slider is None or [(n,id(p))for n,p in self.slider.named_parameters()]!=self.named_ids:
            raise RuntimeError('original adapter Parameter identity changed')
        if any(getattr(m,'_moss_slider_attached',None)is not True for m in self.slider._modules_attached):
            raise RuntimeError('MOSS attachment ownership changed')
        if any(p.requires_grad for p in self.transformer.parameters()):
            raise RuntimeError('frozen original MOSS base ownership changed')
    @staticmethod
    def _run(pending,operation,errors):
        failed=[]
        for item in pending:
            try:operation(item)
            except BaseException as error:failed.append(item);errors.append(error)
        return failed
    def close(self):
        if self.closed:return
        self.closing=True;errors=[]
        self.pending_checkpoints=self._run(self.pending_checkpoints,lambda h:h.close(),errors)
        self.pending_hooks=self._run(self.pending_hooks,lambda h:h.remove(),errors)
        # Never advertise a detached/available target while a hook is pending.
        if not self.pending_hooks:
            def marker(item):
                module,had,old=item
                if vars(module).get('_moss_slider_attached')is not True:
                    raise RuntimeError('foreign MOSS marker retained; restore owned marker before cleanup')
                if had:vars(module)['_moss_slider_attached']=old
                else:vars(module).pop('_moss_slider_attached',None)
            self.pending_markers=self._run(self.pending_markers,marker,errors)
        if not self.pending_checkpoints and not self.pending_hooks and not self.pending_markers:
            self.pending_flags=self._run(self.pending_flags,lambda item:item[0].requires_grad_(item[1]),errors)
            self.pending_modes=self._run(self.pending_modes,lambda item:setattr(item[0],'training',item[1]),errors)
        if errors:
            if len(errors)==1:raise errors[0]
            raise BaseExceptionGroup('MOSS pending attachment restoration failed',errors)
        if any((self.pending_checkpoints,self.pending_hooks,self.pending_markers,self.pending_flags,self.pending_modes)):
            raise RuntimeError('MOSS cleanup still pending')
        if self.slider is not None:
            if hasattr(self.slider,'_handles'):self.slider._handles.clear()
            if hasattr(self.slider,'_modules_attached'):self.slider._modules_attached.clear()
        self.closed,self.closing=True,False

class Owner:
    def __init__(self):self.closed=self.closing=False;self.attachment=None
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('MOSS owner closed/cleanup pending')
        if self.attachment is not None:self.attachment.require_active()
    def close(self):
        if self.closed:return
        self.closing=True
        if self.attachment is not None:self.attachment.close()
        self.closed,self.closing=True,False
    def __enter__(self):self.require_open();return self
    def __exit__(self,kind,primary,traceback):
        try:self.close()
        except BaseException as cleanup:
            if primary is not None:raise BaseExceptionGroup('MOSS work and cleanup failed',[primary,cleanup])
            raise
