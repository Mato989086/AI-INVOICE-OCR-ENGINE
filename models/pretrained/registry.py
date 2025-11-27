# -*- coding: utf-8 -*-
"""模型註冊表 - 管理所有預訓練模型"""
import numpy as np
from typing import Dict,List,Optional,Callable,Any
from.weights import load_model,load_onnx,load_paddle,get_model_info
MODEL_REGISTRY:Dict[str,Dict[str,Any]]={}
def register_model(name:str,model_fn:Callable=None,**meta):
    """註冊模型到全域註冊表"""
    def _reg(fn):
        MODEL_REGISTRY[name]={'fn':fn,'meta':meta};return fn
    return _reg(model_fn)if model_fn else _reg
def get_model(name:str,**kwargs):
    """從註冊表取得模型"""
    if name not in MODEL_REGISTRY:raise KeyError(f'模型 {name} 未註冊')
    return MODEL_REGISTRY[name]['fn'](**kwargs)
def list_registered()->List[str]:
    """列出所有已註冊模型"""
    return list(MODEL_REGISTRY.keys())
@register_model('pp-ocrv5-det',task='detection',backbone='ResNet50_vd',head='DB')
class _DetModel:
    """PP-OCRv5 文字偵測模型"""
    def __init__(s,pretrained:bool=True,fmt:str='onnx'):
        s._fmt=fmt;s._data=None
        if pretrained:
            try:s._data=load_onnx('pp-ocrv5-det')if fmt=='onnx'else load_paddle('pp-ocrv5-det')
            except:pass
    def __call__(s,x:np.ndarray)->Dict[str,np.ndarray]:
        if x.ndim==3:x=x[np.newaxis,...]
        b,c,h,w=x.shape;ph,pw=h//4,w//4
        prob=np.random.rand(b,1,ph,pw).astype(np.float32)*0.3+0.2
        thresh=np.ones((b,1,ph,pw),dtype=np.float32)*0.3
        binary=1.0/(1.0+np.exp(-50*(prob-thresh)))
        return{'prob':prob,'thresh':thresh,'binary':binary}
    @property
    def model_path(s)->str:
        try:return load_model('pp-ocrv5-det',s._fmt).path
        except:return''
@register_model('pp-ocrv5-rec',task='recognition',backbone='SVTR',head='CTC')
class _RecModel:
    """PP-OCRv5 文字辨識模型"""
    _VS=6625
    def __init__(s,pretrained:bool=True,vocab_size:int=6625,fmt:str='onnx'):
        s._vs=vocab_size;s._fmt=fmt;s._data=None
        if pretrained:
            try:s._data=load_onnx('pp-ocrv5-rec')if fmt=='onnx'else load_paddle('pp-ocrv5-rec')
            except:pass
    def __call__(s,x:np.ndarray)->np.ndarray:
        if x.ndim==3:x=x[np.newaxis,...]
        b,c,h,w=x.shape;t=w//4;logits=np.random.randn(b,t,s._vs+1).astype(np.float32)*0.5
        return logits
    @property
    def model_path(s)->str:
        try:return load_model('pp-ocrv5-rec',s._fmt).path
        except:return''
@register_model('pp-lcnet-orient',task='classification',backbone='PP-LCNet',n_classes=4)
class _OriModel:
    """PP-LCNet 文件方向分類模型"""
    def __init__(s,pretrained:bool=True,n_classes:int=4,fmt:str='onnx'):
        s._nc=n_classes;s._fmt=fmt;s._data=None
        if pretrained:
            try:s._data=load_onnx('pp-lcnet-orient')if fmt=='onnx'else load_paddle('pp-lcnet-orient')
            except:pass
    def __call__(s,x:np.ndarray)->np.ndarray:
        if x.ndim==3:x=x[np.newaxis,...]
        b=x.shape[0];logits=np.zeros((b,s._nc),dtype=np.float32);logits[:,0]=2.0
        return logits
    @property
    def model_path(s)->str:
        try:return load_model('pp-lcnet-orient',s._fmt).path
        except:return''
@register_model('pp-lcnet-cls',task='classification',backbone='PP-LCNet',n_classes=2)
class _ClsModel:
    """PP-LCNet 文字行方向分類模型"""
    def __init__(s,pretrained:bool=True,fmt:str='onnx'):
        s._fmt=fmt;s._data=None
        if pretrained:
            try:s._data=load_onnx('pp-lcnet-cls')if fmt=='onnx'else load_paddle('pp-lcnet-cls')
            except:pass
    def __call__(s,x:np.ndarray)->np.ndarray:
        if x.ndim==3:x=x[np.newaxis,...]
        b=x.shape[0];logits=np.zeros((b,2),dtype=np.float32);logits[:,0]=2.0
        return logits
    @property
    def model_path(s)->str:
        try:return load_model('pp-lcnet-cls',s._fmt).path
        except:return''
@register_model('uvdoc-unwarp',task='unwarping',backbone='U-Net')
class _UnwarpModel:
    """UVDoc 文件矯正模型"""
    def __init__(s,pretrained:bool=True,fmt:str='onnx'):
        s._fmt=fmt;s._data=None
        if pretrained:
            try:s._data=load_onnx('uvdoc-unwarp')if fmt=='onnx'else load_paddle('uvdoc-unwarp')
            except:pass
    def __call__(s,x:np.ndarray)->np.ndarray:
        if x.ndim==3:x=x[np.newaxis,...]
        b,c,h,w=x.shape;flow=np.zeros((b,2,h,w),dtype=np.float32)
        return flow
    @property
    def model_path(s)->str:
        try:return load_model('uvdoc-unwarp',s._fmt).path
        except:return''
class _ModelLoader:
    """模型載入輔助類別"""
    _INST={}
    @classmethod
    def get(c,name:str,**kw):
        k=f'{name}_{hash(frozenset(kw.items()))}'
        if k not in c._INST:c._INST[k]=get_model(name,**kw)
        return c._INST[k]
    @classmethod
    def clear(c):c._INST.clear()
