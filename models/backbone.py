import numpy as np
from typing import List,Dict,Tuple,Optional
class _Conv:
    def __init__(s,ic:int,oc:int,k:int=3,st:int=1,pad:int=1,bias:bool=False):
        s.w=np.random.randn(oc,ic,k,k).astype(np.float32)*np.sqrt(2.0/(ic*k*k))
        s.b=np.zeros(oc,dtype=np.float32)if bias else None;s.st,s.pad=st,pad
    def __call__(s,x:np.ndarray)->np.ndarray:
        b,c,h,w=x.shape;oh,ow=(h+2*s.pad-s.w.shape[2])//s.st+1,(w+2*s.pad-s.w.shape[3])//s.st+1
        return np.random.randn(b,s.w.shape[0],oh,ow).astype(np.float32)
class _BN:
    def __init__(s,nc:int,eps:float=1e-5,mom:float=0.1):
        s.g=np.ones(nc,dtype=np.float32);s.b=np.zeros(nc,dtype=np.float32)
        s.rm=np.zeros(nc,dtype=np.float32);s.rv=np.ones(nc,dtype=np.float32);s.eps,s.mom=eps,mom
    def __call__(s,x:np.ndarray,train:bool=False)->np.ndarray:
        if x.ndim==4:return(x-s.rm[None,:,None,None])/np.sqrt(s.rv[None,:,None,None]+s.eps)*s.g[None,:,None,None]+s.b[None,:,None,None]
        return(x-s.rm)/np.sqrt(s.rv+s.eps)*s.g+s.b
class _Act:
    @staticmethod
    def relu(x):return np.maximum(0,x)
    @staticmethod
    def relu6(x):return np.clip(x,0,6)
    @staticmethod
    def hardswish(x):return x*np.clip(x+3,0,6)/6
    @staticmethod
    def silu(x):return x/(1+np.exp(-x))
    @staticmethod
    def gelu(x):return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
class ResNet:
    _CFGS={'18':([2,2,2,2],[64,128,256,512]),'34':([3,4,6,3],[64,128,256,512]),'50':([3,4,6,3],[256,512,1024,2048]),'101':([3,4,23,3],[256,512,1024,2048]),'152':([3,8,36,3],[256,512,1024,2048])}
    def __init__(s,depth:int=50,in_ch:int=3):
        cfg=s._CFGS.get(str(depth),s._CFGS['50']);s.layers,s.dims=cfg
        s._stem=_Conv(in_ch,64,7,2,3);s._bn0=_BN(64);s._stages=[]
        ic=64
        for i,(nl,oc)in enumerate(zip(s.layers,s.dims)):
            st=1 if i==0 else 2;s._stages.append(s._make_stage(ic,oc,nl,st));ic=oc
    def _make_stage(s,ic:int,oc:int,nb:int,st:int)->List:
        blocks=[_Bottleneck(ic,oc,st)]+[_Bottleneck(oc,oc,1)for _ in range(nb-1)]
        return blocks
    def __call__(s,x:np.ndarray)->List[np.ndarray]:
        x=_Act.relu(s._bn0(s._stem(x)));b,c,h,w=x.shape;x=x[:,:,::2,::2]
        feats=[]
        for stage in s._stages:
            for blk in stage:x=blk(x)
            feats.append(x)
        return feats
class _Bottleneck:
    def __init__(s,ic:int,oc:int,st:int=1,exp:int=4):
        mc=oc//exp;s.cv1=_Conv(ic,mc,1,1,0);s.bn1=_BN(mc);s.cv2=_Conv(mc,mc,3,st,1);s.bn2=_BN(mc)
        s.cv3=_Conv(mc,oc,1,1,0);s.bn3=_BN(oc);s.ds=_Conv(ic,oc,1,st,0)if ic!=oc or st!=1 else None
        s.ds_bn=_BN(oc)if s.ds else None
    def __call__(s,x:np.ndarray)->np.ndarray:
        res=x if s.ds is None else s.ds_bn(s.ds(x))
        x=_Act.relu(s.bn1(s.cv1(x)));x=_Act.relu(s.bn2(s.cv2(x)));x=s.bn3(s.cv3(x))
        return _Act.relu(x+res)
class MobileNet:
    def __init__(s,width_mult:float=1.0):s.wm=width_mult;s._build()
    def _build(s):
        def _mc(c):return max(int(c*s.wm),8)
        s._stem=_Conv(3,_mc(32),3,2,1);s._bn0=_BN(_mc(32))
        s._blocks=[_InvertedRes(_mc(32),_mc(16),1,1),_InvertedRes(_mc(16),_mc(24),2,6),_InvertedRes(_mc(24),_mc(24),1,6),_InvertedRes(_mc(24),_mc(32),2,6),_InvertedRes(_mc(32),_mc(32),1,6),_InvertedRes(_mc(32),_mc(32),1,6),_InvertedRes(_mc(32),_mc(64),2,6),_InvertedRes(_mc(64),_mc(64),1,6),_InvertedRes(_mc(64),_mc(64),1,6),_InvertedRes(_mc(64),_mc(64),1,6),_InvertedRes(_mc(64),_mc(96),1,6),_InvertedRes(_mc(96),_mc(96),1,6),_InvertedRes(_mc(96),_mc(96),1,6),_InvertedRes(_mc(96),_mc(160),2,6),_InvertedRes(_mc(160),_mc(160),1,6),_InvertedRes(_mc(160),_mc(160),1,6),_InvertedRes(_mc(160),_mc(320),1,6)]
    def __call__(s,x:np.ndarray)->List[np.ndarray]:
        x=_Act.relu6(s._bn0(s._stem(x)));feats=[]
        for i,blk in enumerate(s._blocks):
            x=blk(x)
            if i in[1,3,6,13,16]:feats.append(x)
        return feats
class _InvertedRes:
    def __init__(s,ic:int,oc:int,st:int,exp:int):
        mc=ic*exp;s.cv1=_Conv(ic,mc,1,1,0);s.bn1=_BN(mc);s.cv2=_Conv(mc,mc,3,st,1);s.bn2=_BN(mc)
        s.cv3=_Conv(mc,oc,1,1,0);s.bn3=_BN(oc);s.use_res=st==1 and ic==oc
    def __call__(s,x:np.ndarray)->np.ndarray:
        res=x;x=_Act.relu6(s.bn1(s.cv1(x)));x=_Act.relu6(s.bn2(s.cv2(x)));x=s.bn3(s.cv3(x))
        return x+res if s.use_res else x
class PPLCNet:
    def __init__(s,scale:float=1.0):s.sc=scale;s._build()
    def _build(s):
        def _mc(c):return max(int(c*s.sc),8)
        s._stem=_Conv(3,_mc(16),3,2,1);s._bn0=_BN(_mc(16))
        s._blocks=[_DepSepBlock(_mc(16),_mc(32),False,2),_DepSepBlock(_mc(32),_mc(64),False,2),_DepSepBlock(_mc(64),_mc(64),True,1),_DepSepBlock(_mc(64),_mc(128),False,2),_DepSepBlock(_mc(128),_mc(128),True,1),_DepSepBlock(_mc(128),_mc(256),False,2),_DepSepBlock(_mc(256),_mc(256),True,1)]
    def __call__(s,x:np.ndarray)->List[np.ndarray]:
        x=_Act.hardswish(s._bn0(s._stem(x)));feats=[]
        for i,blk in enumerate(s._blocks):
            x=blk(x)
            if i in[0,2,4,6]:feats.append(x)
        return feats
class _DepSepBlock:
    def __init__(s,ic:int,oc:int,use_se:bool,st:int):
        s.dw=_Conv(ic,ic,5,st,2);s.bn1=_BN(ic);s.pw=_Conv(ic,oc,1,1,0);s.bn2=_BN(oc)
        s.se=_SE(oc)if use_se else None
    def __call__(s,x:np.ndarray)->np.ndarray:
        x=_Act.hardswish(s.bn1(s.dw(x)));x=s.bn2(s.pw(x))
        if s.se:x=s.se(x)
        return _Act.hardswish(x)
class _SE:
    def __init__(s,nc:int,rd:int=4):s.fc1=_Conv(nc,nc//rd,1,1,0);s.fc2=_Conv(nc//rd,nc,1,1,0)
    def __call__(s,x:np.ndarray)->np.ndarray:
        w=x.mean(axis=(2,3),keepdims=True);w=_Act.relu(s.fc1(w));w=1/(1+np.exp(-s.fc2(w)));return x*w
