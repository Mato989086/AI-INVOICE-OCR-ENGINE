# -*- coding: utf-8 -*-
"""資料增強模組 - 訓練時的圖片增強操作"""
import cv2,numpy as np
from typing import Tuple,List,Dict,Optional,Callable
import random as _R
class Augmenter:
    """資料增強器 - 支援多種增強操作"""
    def __init__(s,cfg:Dict=None):
        """初始化增強器"""
        s.cfg=cfg or{};s._ops=[]
        s._init_ops()
    def _init_ops(s):
        """初始化增強操作列表"""
        c=s.cfg;s._ops=[
            ('flip_h',lambda im:cv2.flip(im,1),c.get('flip_h',0.5)),  # 水平翻轉
            ('flip_v',lambda im:cv2.flip(im,0),c.get('flip_v',0.0)),  # 垂直翻轉
            ('rot',s._rotate,c.get('rotate',0.3)),  # 旋轉
            ('scale',s._scale,c.get('scale',0.3)),  # 縮放
            ('blur',s._blur,c.get('blur',0.2)),  # 模糊
            ('noise',s._noise,c.get('noise',0.2)),  # 雜訊
            ('bright',s._brightness,c.get('brightness',0.3)),  # 亮度
            ('contrast',s._contrast,c.get('contrast',0.3)),  # 對比度
            ('sharp',s._sharpen,c.get('sharpen',0.1)),  # 銳化
            ('erode',s._erode,c.get('erode',0.1)),  # 侵蝕
            ('dilate',s._dilate,c.get('dilate',0.1)),  # 膨脹
        ]
    def _rotate(s,im:np.ndarray)->np.ndarray:
        """隨機旋轉"""
        a=_R.uniform(-10,10);h,w=im.shape[:2];M=cv2.getRotationMatrix2D((w/2,h/2),a,1)
        return cv2.warpAffine(im,M,(w,h),borderMode=cv2.BORDER_REPLICATE)
    def _scale(s,im:np.ndarray)->np.ndarray:
        """隨機縮放"""
        sc=_R.uniform(0.8,1.2);h,w=im.shape[:2];nh,nw=int(h*sc),int(w*sc)
        im=cv2.resize(im,(nw,nh))
        if sc>1:y,x=(nh-h)//2,(nw-w)//2;im=im[y:y+h,x:x+w]
        else:
            pad=np.zeros((h,w,3)if im.ndim==3 else(h,w),dtype=im.dtype)
            y,x=(h-nh)//2,(w-nw)//2;pad[y:y+nh,x:x+nw]=im;im=pad
        return im
    def _blur(s,im:np.ndarray)->np.ndarray:
        """高斯模糊"""
        k=_R.choice([3,5,7]);return cv2.GaussianBlur(im,(k,k),0)
    def _noise(s,im:np.ndarray)->np.ndarray:
        """加入高斯雜訊"""
        n=np.random.normal(0,_R.uniform(5,25),im.shape).astype(np.float32)
        return np.clip(im.astype(np.float32)+n,0,255).astype(np.uint8)
    def _brightness(s,im:np.ndarray)->np.ndarray:
        """調整亮度"""
        b=_R.uniform(-50,50);return np.clip(im.astype(np.float32)+b,0,255).astype(np.uint8)
    def _contrast(s,im:np.ndarray)->np.ndarray:
        """調整對比度"""
        c=_R.uniform(0.7,1.3);m=im.mean();return np.clip((im.astype(np.float32)-m)*c+m,0,255).astype(np.uint8)
    def _sharpen(s,im:np.ndarray)->np.ndarray:
        """銳化"""
        k=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]);return cv2.filter2D(im,-1,k)
    def _erode(s,im:np.ndarray)->np.ndarray:
        """侵蝕操作"""
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(_R.choice([2,3]),_R.choice([2,3])))
        return cv2.erode(im,k,iterations=1)
    def _dilate(s,im:np.ndarray)->np.ndarray:
        """膨脹操作"""
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(_R.choice([2,3]),_R.choice([2,3])))
        return cv2.dilate(im,k,iterations=1)
    def __call__(s,im:np.ndarray,n:int=None)->np.ndarray:
        """執行隨機增強操作"""
        ops=[(nm,fn)for nm,fn,p in s._ops if _R.random()<p]
        if n:ops=_R.sample(ops,min(n,len(ops)))
        for _,fn in ops:im=fn(im)
        return im
    def apply(s,im:np.ndarray,ops:List[str])->np.ndarray:
        """執行指定增強操作"""
        op_map={nm:fn for nm,fn,_ in s._ops}
        for op in ops:
            if op in op_map:im=op_map[op](im)
        return im
class _TPS:
    """薄板樣條變換 - 用於文字扭曲增強"""
    def __init__(s,src:np.ndarray,dst:np.ndarray):
        """初始化TPS變換"""
        s.n=src.shape[0];s.src=src;s.dst=dst;s._calc_params()
    def _calc_params(s):
        """計算TPS參數"""
        n=s.n;K=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j:r=np.linalg.norm(s.src[i]-s.src[j]);K[i,j]=r*r*np.log(r+1e-6)if r>0 else 0
        P=np.hstack([np.ones((n,1)),s.src]);L=np.vstack([np.hstack([K,P]),np.hstack([P.T,np.zeros((3,3))])])
        Y=np.vstack([s.dst,np.zeros((3,2))]);s.W=np.linalg.lstsq(L,Y,rcond=None)[0]
    def transform(s,pts:np.ndarray)->np.ndarray:
        """變換點座標"""
        n,m=s.n,pts.shape[0];U=np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                r=np.linalg.norm(pts[i]-s.src[j]);U[i,j]=r*r*np.log(r+1e-6)if r>0 else 0
        P=np.hstack([np.ones((m,1)),pts]);return np.dot(np.hstack([U,P]),s.W)
