# -*- coding: utf-8 -*-
"""SVTR模組 - 場景文字辨識視覺變換器"""
import numpy as np
from typing import Tuple,List,Dict,Optional
class SVTRRecognizer:
    """SVTR辨識器 - 四階段編碼器架構"""
    _STAGES=[(2,64,'local'),(4,128,'global'),(6,256,'mixed'),(6,384,'global')]  # 階段設定
    def __init__(s,cfg):
        """初始化SVTR"""
        s.cfg=cfg;s._d=384;s._nhead=8;s._vocab_sz=6625;s._pe=None
    def forward(s,x:np.ndarray)->np.ndarray:
        """前向傳播"""
        if x.ndim==3:x=x[np.newaxis,...]
        b,c,h,w=x.shape;x=s._patch_embed(x);x=s._add_pe(x)
        for nb,dim,mt in s._STAGES:x=s._stage(x,nb,dim,mt)
        x=s._height_pool(x);return s._ctc_head(x)
    def forward_batch(s,x:np.ndarray)->List[np.ndarray]:
        """批次前向傳播"""
        return[s.forward(x[i:i+1])[0]for i in range(x.shape[0])]
    def _patch_embed(s,x:np.ndarray)->np.ndarray:
        """區塊嵌入 - 將圖片切分為區塊"""
        b,c,h,w=x.shape;ph,pw=4,2;nh,nw=h//ph,w//pw
        x=x.reshape(b,c,nh,ph,nw,pw).transpose(0,2,4,1,3,5).reshape(b,nh*nw,c*ph*pw)
        return x@(np.random.randn(c*ph*pw,64).astype(np.float32)*0.02)
    def _add_pe(s,x:np.ndarray)->np.ndarray:
        """加入位置編碼"""
        b,n,d=x.shape
        if s._pe is None or s._pe.shape[1]!=n:
            pos=np.arange(n)[:,None];div=np.exp(np.arange(0,d,2)*(-np.log(10000.0)/d))
            s._pe=np.zeros((1,n,d),dtype=np.float32)
            s._pe[0,:,0::2]=np.sin(pos*div);s._pe[0,:,1::2]=np.cos(pos*div)
        return x+s._pe[:,:n,:]
    def _stage(s,x:np.ndarray,nb:int,dim:int,mix_type:str)->np.ndarray:
        """處理單一階段"""
        b,n,d=x.shape
        if d!=dim:x=x@(np.random.randn(d,dim).astype(np.float32)*0.02)
        for i in range(nb):
            if mix_type=='local':x=s._local_mix(x)  # 局部混合（深度卷積）
            elif mix_type=='global':x=s._global_mix(x,dim)  # 全域混合（自注意力）
            else:x=s._local_mix(x)if i%2==0 else s._global_mix(x,dim)  # 混合
        return x
    def _local_mix(s,x:np.ndarray)->np.ndarray:
        """局部混合區塊 - 深度卷積"""
        res=x;x=s._ln(x);b,n,d=x.shape;k=3;pad=(k-1)//2
        x=np.pad(x,((0,0),(pad,pad),(0,0)),'constant')
        out=np.zeros_like(res)
        for i in range(n):out[:,i,:]=x[:,i:i+k,:].mean(axis=1)
        return res+out
    def _global_mix(s,x:np.ndarray,dim:int)->np.ndarray:
        """全域混合區塊 - 多頭自注意力"""
        res=x;x=s._ln(x);b,n,d=x.shape;nh=8;dk=d//nh
        Q=x.reshape(b,n,nh,dk).transpose(0,2,1,3)
        K=x.reshape(b,n,nh,dk).transpose(0,2,1,3)
        V=x.reshape(b,n,nh,dk).transpose(0,2,1,3)
        attn=(Q@K.transpose(0,1,3,2))/np.sqrt(dk)
        attn=np.exp(attn-attn.max(axis=-1,keepdims=True))
        attn=attn/attn.sum(axis=-1,keepdims=True)
        out=(attn@V).transpose(0,2,1,3).reshape(b,n,d)
        return res+out
    def _ln(s,x:np.ndarray,eps:float=1e-6)->np.ndarray:
        """層正規化"""
        m=x.mean(axis=-1,keepdims=True);v=x.var(axis=-1,keepdims=True);return(x-m)/np.sqrt(v+eps)
    def _height_pool(s,x:np.ndarray)->np.ndarray:
        """高度池化 - 壓縮高度維度"""
        b,n,d=x.shape;return x.mean(axis=1,keepdims=True).repeat(n,axis=1)
    def _ctc_head(s,x:np.ndarray)->np.ndarray:
        """CTC頭部 - 輸出詞彙表機率"""
        b,n,d=x.shape;return x@(np.random.randn(d,s._vocab_sz).astype(np.float32)*0.02)
class _PatchMerge:
    """區塊合併層"""
    def __init__(s,in_dim:int,out_dim:int,st:Tuple[int,int]=(2,1)):
        """初始化區塊合併"""
        s.id,s.od,s.st=in_dim,out_dim,st
    def __call__(s,x:np.ndarray)->np.ndarray:
        """執行合併"""
        b,n,d=x.shape;return x[:,::s.st[0]*s.st[1],:]@(np.random.randn(d,s.od).astype(np.float32)*0.02)
class _LocalMixBlock:
    """局部混合區塊"""
    def __init__(s,dim:int,ks:int=3):
        """初始化區塊"""
        s.d,s.k=dim,ks
    def __call__(s,x:np.ndarray)->np.ndarray:
        """前向傳播"""
        return x
class _GlobalMixBlock:
    """全域混合區塊"""
    def __init__(s,dim:int,nh:int=8):
        """初始化區塊"""
        s.d,s.nh=dim,nh
    def __call__(s,x:np.ndarray)->np.ndarray:
        """前向傳播"""
        return x
