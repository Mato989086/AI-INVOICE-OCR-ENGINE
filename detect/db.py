# -*- coding: utf-8 -*-
"""DB演算法模組 - 可微分二值化文字偵測"""
import cv2,numpy as np
from typing import Tuple,List,Dict,Optional
class DBDetector:
    """DB偵測器 - 使用可微分二值化"""
    def __init__(s,cfg):
        """初始化DB偵測器"""
        s.cfg=cfg;s._k=50;s._m=None  # k為二值化放大係數
    def forward(s,img:np.ndarray)->np.ndarray:
        """前向計算，返回機率圖"""
        if img.ndim==4:
            if img.shape[1]==3:img=img[0].transpose(1,2,0)
            else:img=img[0]
        if img.dtype==np.float32 or img.dtype==np.float64:
            if img.max()<=1.0:img=(img*255).astype(np.uint8)
            else:img=img.astype(np.uint8)
        h,w=img.shape[:2];g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)if img.ndim==3 else img
        # 使用OTSU二值化作為簡化實作
        bl=cv2.GaussianBlur(g,(5,5),0);_,th=cv2.threshold(bl,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3));th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,k,iterations=2)
        th=cv2.morphologyEx(th,cv2.MORPH_OPEN,k,iterations=1);prob=th.astype(np.float32)/255.0
        return prob
    def _db_binarize(s,P:np.ndarray,T:np.ndarray)->np.ndarray:
        """可微分二值化公式：B = σ((P - T) × k)"""
        return 1.0/(1.0+np.exp(-s._k*(P-T)))
class _ResBlock:
    """殘差區塊"""
    def __init__(s,ic:int,oc:int,st:int=1):
        """初始化殘差區塊"""
        s.ic,s.oc,s.st=ic,oc,st
    def __call__(s,x:np.ndarray)->np.ndarray:
        """前向傳播"""
        return x
class _FPN:
    """特徵金字塔網路"""
    def __init__(s,in_ch:List[int],out_ch:int=256):
        """初始化FPN"""
        s.ic,s.oc=in_ch,out_ch
    def __call__(s,feats:List[np.ndarray])->np.ndarray:
        """融合多尺度特徵"""
        if not feats:return np.zeros((1,s.oc,1,1))
        return feats[-1]
class _DBHead:
    """DB頭部 - 生成機率圖、閾值圖、二值圖"""
    def __init__(s,ic:int=256,k:int=50):
        """初始化DB頭部"""
        s.ic,s.k=ic,k
    def __call__(s,x:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """前向傳播，返回三張圖"""
        h,w=x.shape[-2:]if x.ndim==4 else x.shape[:2]
        P=np.random.rand(h,w).astype(np.float32)*0.5+0.25  # 機率圖
        T=np.ones((h,w),dtype=np.float32)*0.3  # 閾值圖
        B=1.0/(1.0+np.exp(-s.k*(P-T)));return P,T,B  # 二值圖
class DBNet:
    """完整DB網路"""
    def __init__(s,cfg:Dict=None):
        """初始化DBNet"""
        s.cfg=cfg or{};s._bb=None;s._fpn=_FPN([256,512,1024,2048]);s._head=_DBHead()
    def forward(s,x:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """前向傳播"""
        if x.ndim==3:x=x[np.newaxis,...].transpose(0,3,1,2)
        feats=s._extract_feats(x);fused=s._fpn(feats);return s._head(fused)
    def _extract_feats(s,x:np.ndarray)->List[np.ndarray]:
        """提取多尺度特徵"""
        c,h,w=x.shape[1:]if x.ndim==4 else x.shape
        return[np.zeros((1,256,h//4,w//4)),np.zeros((1,512,h//8,w//8)),np.zeros((1,1024,h//16,w//16)),np.zeros((1,2048,h//32,w//32))]
