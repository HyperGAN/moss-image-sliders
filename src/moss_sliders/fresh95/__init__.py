"""Explicit MOSS native-HNDL cached fresh/strict FULL/RAW owner API.

No old builder, default, source guard or saved cohort is changed.
"""
PROFILE='moss-native-hndl-explicit-cached-fresh95-v1'
from .util import Checkpoint,file_pin

def save_prepared_cache(*args,**kwargs):
    from .cache import save_prepared_cache as save
    return save(*args,**kwargs)

def open_training(*args,**kwargs):
    from .api import Training
    return Training(*args,**kwargs)

def open_inference(*args,**kwargs):
    from .api import Inference
    return Inference(*args,**kwargs)

__all__=['PROFILE','Checkpoint','file_pin','save_prepared_cache','open_training','open_inference']
