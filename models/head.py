import numpy as np
from typing import List,Dict,Tuple,Optional
class _Conv:
    def __init__(s,ic:int,oc:int,k:int=3,st:int=1):s.ic,s.oc,s.k,s.st=ic,oc,k,st
    def __call__(s,x:np.ndarray)->np.ndarray:
        b,c,h,w=x.shape;return np.random.randn(b,s.oc,h//s.st,w//s.st).astype(np.float32)*0.1
class _BN:
    def __init__(s,nc:int):s.nc=nc
    def __call__(s,x:np.ndarray)->np.ndarray:return x
class DBHead:
    def __init__(s,in_ch:int=256,k:int=50):
        s.ic,s.k=in_ch,k;s._prob=nn.Sequential(_Conv(in_ch,64,3),_BN(64),_Conv(64,1,1))
        s._thresh=nn.Sequential(_Conv(in_ch,64,3),_BN(64),_Conv(64,1,1))
    def __call__(s,x:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        b,c,h,w=x.shape;P=1/(1+np.exp(-np.random.randn(b,1,h*4,w*4).astype(np.float32)))
        T=np.ones((b,1,h*4,w*4),dtype=np.float32)*0.3;B=1/(1+np.exp(-s.k*(P-T)));return P,T,B
class nn:
    class Sequential:
        def __init__(s,*layers):s.layers=layers
        def __call__(s,x):
            for l in s.layers:x=l(x)
            return x
class CTCHead:
    def __init__(s,in_ch:int=384,vocab_sz:int=6625):s.ic,s.vs=in_ch,vocab_sz;s._fc=_Linear(in_ch,vocab_sz)
    def __call__(s,x:np.ndarray)->np.ndarray:
        if x.ndim==4:b,c,h,w=x.shape;x=x.mean(axis=2).transpose(0,2,1)
        return s._fc(x)
class _Linear:
    def __init__(s,ic:int,oc:int):s.w=np.random.randn(ic,oc).astype(np.float32)*np.sqrt(2.0/ic);s.b=np.zeros(oc,dtype=np.float32)
    def __call__(s,x:np.ndarray)->np.ndarray:return x@s.w+s.b
class AttentionHead:
    def __init__(s,in_ch:int=384,vocab_sz:int=6625,max_len:int=25,n_head:int=8):
        s.ic,s.vs,s.ml,s.nh=in_ch,vocab_sz,max_len,n_head;s._emb=np.random.randn(vocab_sz,in_ch).astype(np.float32)*0.02
        s._pe=s._pos_enc(max_len,in_ch);s._fc=_Linear(in_ch,vocab_sz)
    def _pos_enc(s,ml:int,d:int)->np.ndarray:
        pe=np.zeros((ml,d),dtype=np.float32);pos=np.arange(ml)[:,None];div=np.exp(np.arange(0,d,2)*(-np.log(10000.0)/d))
        pe[:,0::2]=np.sin(pos*div);pe[:,1::2]=np.cos(pos*div);return pe[None,...]
    def __call__(s,enc:np.ndarray,tgt:np.ndarray=None)->np.ndarray:
        b=enc.shape[0];tgt_emb=s._emb[np.zeros((b,s.ml),dtype=np.int32)]+s._pe
        for _ in range(6):
            attn=s._cross_attn(tgt_emb,enc);tgt_emb=tgt_emb+attn
        return s._fc(tgt_emb)
    def _cross_attn(s,q:np.ndarray,kv:np.ndarray)->np.ndarray:
        b,n,d=q.shape;dk=d//s.nh;Q=q.reshape(b,n,s.nh,dk).transpose(0,2,1,3)
        K=kv.reshape(b,-1,s.nh,dk).transpose(0,2,1,3);V=K
        attn=(Q@K.transpose(0,1,3,2))/np.sqrt(dk);attn=np.exp(attn-attn.max(axis=-1,keepdims=True))
        attn=attn/attn.sum(axis=-1,keepdims=True);out=(attn@V).transpose(0,2,1,3).reshape(b,n,d);return out
class SegHead:
    def __init__(s,in_ch:int=256,n_cls:int=2):s.ic,s.nc=in_ch,n_cls;s._conv=_Conv(in_ch,n_cls,1)
    def __call__(s,x:np.ndarray)->np.ndarray:return s._conv(x)
class ClsHead:
    def __init__(s,in_ch:int=256,n_cls:int=4):s.ic,s.nc=in_ch,n_cls;s._gap=None;s._fc=_Linear(in_ch,n_cls)
    def __call__(s,x:np.ndarray)->np.ndarray:
        x=x.mean(axis=(2,3));return s._fc(x)
