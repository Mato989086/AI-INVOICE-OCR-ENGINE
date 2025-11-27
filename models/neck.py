import numpy as np
from typing import List,Dict,Tuple
class _Conv:
    def __init__(s,ic:int,oc:int,k:int=1,st:int=1):s.ic,s.oc,s.k,s.st=ic,oc,k,st
    def __call__(s,x:np.ndarray)->np.ndarray:
        b,c,h,w=x.shape;oh,ow=h//s.st,w//s.st;return np.random.randn(b,s.oc,oh,ow).astype(np.float32)*0.1
class FPN:
    def __init__(s,in_chs:List[int],out_ch:int=256):
        s.ic,s.oc=in_chs,out_ch;s._lat=[_Conv(c,out_ch)for c in in_chs];s._out=[_Conv(out_ch,out_ch,3)for _ in in_chs]
    def __call__(s,feats:List[np.ndarray])->np.ndarray:
        n=len(feats);lats=[s._lat[i](feats[i])for i in range(n)]
        for i in range(n-2,-1,-1):
            h,w=lats[i].shape[2:];up=np.repeat(np.repeat(lats[i+1],2,axis=2),2,axis=3)[:,:,:h,:w]
            lats[i]=lats[i]+up
        outs=[s._out[i](lats[i])for i in range(n)];h,w=outs[0].shape[2:]
        ups=[np.repeat(np.repeat(o,2**(i),axis=2),2**(i),axis=3)[:,:,:h,:w]for i,o in enumerate(outs)]
        return np.concatenate(ups,axis=1)
class FPEM:
    def __init__(s,in_chs:List[int],out_ch:int=128,n_iter:int=2):
        s.ic,s.oc,s.ni=in_chs,out_ch,n_iter;s._up=[_Conv(out_ch,out_ch)for _ in range(len(in_chs)-1)]
        s._dw=[_Conv(out_ch,out_ch)for _ in range(len(in_chs)-1)]
    def __call__(s,feats:List[np.ndarray])->List[np.ndarray]:
        n=len(feats)
        for _ in range(s.ni):
            for i in range(n-1):
                h,w=feats[i].shape[2:];up=np.repeat(np.repeat(feats[i+1],2,axis=2),2,axis=3)[:,:,:h,:w]
                feats[i]=feats[i]+s._up[i](up)
            for i in range(n-2,-1,-1):
                h,w=feats[i+1].shape[2:];dw=feats[i][:,:,::2,::2][:,:,:h,:w]
                feats[i+1]=feats[i+1]+s._dw[i](dw)
        return feats
class BIFPN:
    def __init__(s,in_chs:List[int],out_ch:int=256,n_iter:int=3):
        s.ic,s.oc,s.ni=in_chs,out_ch,n_iter;n=len(in_chs)
        s._td=[_Conv(out_ch,out_ch)for _ in range(n-1)];s._bu=[_Conv(out_ch,out_ch)for _ in range(n-1)]
        s._w_td=np.ones((n-1,2),dtype=np.float32);s._w_bu=np.ones((n-1,3),dtype=np.float32)
    def __call__(s,feats:List[np.ndarray])->List[np.ndarray]:
        n=len(feats);outs=feats.copy()
        for _ in range(s.ni):
            td=[None]*n;td[n-1]=outs[n-1]
            for i in range(n-2,-1,-1):
                h,w=outs[i].shape[2:];up=np.repeat(np.repeat(td[i+1],2,axis=2),2,axis=3)[:,:,:h,:w]
                wsum=s._w_td[i].sum()+1e-4;td[i]=(s._w_td[i,0]*outs[i]+s._w_td[i,1]*up)/wsum
            bu=[None]*n;bu[0]=td[0]
            for i in range(1,n):
                h,w=td[i].shape[2:];dw=bu[i-1][:,:,::2,::2][:,:,:h,:w]
                if i<n-1:wsum=s._w_bu[i-1].sum()+1e-4;bu[i]=(s._w_bu[i-1,0]*outs[i]+s._w_bu[i-1,1]*td[i]+s._w_bu[i-1,2]*dw)/wsum
                else:bu[i]=(td[i]+dw)/2
            outs=bu
        return outs
class _ASPP:
    def __init__(s,ic:int,oc:int,rates:List[int]=[6,12,18]):
        s.convs=[_Conv(ic,oc,3)for _ in rates];s.pool=_Conv(ic,oc);s.out=_Conv(oc*(len(rates)+2),oc)
    def __call__(s,x:np.ndarray)->np.ndarray:
        b,c,h,w=x.shape;outs=[cv(x)for cv in s.convs]
        gp=x.mean(axis=(2,3),keepdims=True);gp=np.repeat(np.repeat(s.pool(gp),h,axis=2),w,axis=3)
        outs.append(gp);outs.append(x[:,:s.convs[0].oc,:,:])
        return s.out(np.concatenate(outs,axis=1))
