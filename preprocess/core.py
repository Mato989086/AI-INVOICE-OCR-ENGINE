# -*- coding: utf-8 -*-
"""前處理核心模組 - 整合所有前處理功能"""
import cv2,numpy as np
from typing import Tuple,Dict,List,Optional,Union
from.orientation import OrientationClassifier as _OC
from.unwarp import DocumentUnwarper as _DU
class Preprocessor:
    """前處理器 - 處理圖片正規化、方向校正等"""
    _ROT={0:None,1:cv2.ROTATE_90_CLOCKWISE,2:cv2.ROTATE_180,3:cv2.ROTATE_90_COUNTERCLOCKWISE}  # 旋轉對應表
    def __init__(s,cfg):
        """初始化前處理器"""
        s.cfg=cfg;s._oc=_OC();s._du=_DU();s._m=np.array(cfg.nm).reshape(1,1,3);s._s=np.array(cfg.ns).reshape(1,1,3)
    def correct_orientation(s,img:np.ndarray)->Tuple[np.ndarray,Dict]:
        """校正文件方向"""
        cls,sc=s._oc.classify(img);r=s._ROT.get(cls)
        if sc>=s.cfg.oth and r is not None:img=cv2.rotate(img,r)
        return img,{'cls':cls,'score':float(sc),'rotated':r is not None and sc>=s.cfg.oth}
    def unwarp(s,img:np.ndarray)->Tuple[np.ndarray,Dict]:
        """校正文件扭曲"""
        flow,sc=s._du.predict(img)
        if sc>=s.cfg.uwth:
            h,w=img.shape[:2];mx,my=flow[:,:,0]*w,flow[:,:,1]*h
            gx,gy=np.meshgrid(np.arange(w),np.arange(h))
            img=cv2.remap(img,(gx+mx).astype(np.float32),(gy+my).astype(np.float32),cv2.INTER_LINEAR)
        return img,{'score':float(sc),'unwarped':sc>=s.cfg.uwth}
    def resize(s,img:np.ndarray)->Tuple[np.ndarray,Dict]:
        """調整圖片尺寸，保持比例並填充至32的倍數"""
        h,w=img.shape[:2];mxh,mxw=s.cfg.rs;sc=min(mxh/h,mxw/w,1.0);nh,nw=int(h*sc),int(w*sc)
        nh,nw=(nh//32)*32 or 32,(nw//32)*32 or 32;img=cv2.resize(img,(nw,nh))
        return img,{'orig':(w,h),'new':(nw,nh),'scale':sc}
    def normalize(s,img:np.ndarray)->np.ndarray:
        """正規化圖片到模型輸入格式"""
        img=img.astype(np.float32)/255.0;img=(img-s._m)/s._s;return img.transpose(2,0,1)[np.newaxis,...]
    def correct_textline(s,img:np.ndarray)->np.ndarray:
        """校正文字行方向（0°或180°）"""
        cls,sc=s._oc.classify_textline(img)
        return cv2.rotate(img,cv2.ROTATE_180)if cls==1 and sc>=s.cfg.oth else img
    def pad_batch(s,imgs:List[np.ndarray],h:int=48)->np.ndarray:
        """將多張圖片填充成統一大小的批次"""
        def _rs(im):
            oh,ow=im.shape[:2];nw=int(ow*h/oh);im=cv2.resize(im,(nw,h))
            return im,nw
        rs=[_rs(im)for im in imgs];mw=max(r[1]for r in rs)
        batch=np.zeros((len(imgs),h,mw,3),dtype=np.uint8)
        for i,(im,w)in enumerate(rs):batch[i,:,:w,:]=im
        return batch
    def denormalize(s,img:np.ndarray)->np.ndarray:
        """反正規化圖片"""
        if img.ndim==4:img=img[0]
        if img.shape[0]==3:img=img.transpose(1,2,0)
        img=img*s._s+s._m;return(img*255).clip(0,255).astype(np.uint8)
class _ColorNorm:
    """色彩空間轉換工具類別"""
    @staticmethod
    def gray(img):
        """轉換為灰階"""
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)if img.ndim==3 else img
    @staticmethod
    def rgb(img):
        """轉換為RGB"""
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)if img.ndim==3 else cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    @staticmethod
    def bgr(img):
        """轉換為BGR"""
        return img if img.ndim==3 and img.shape[2]==3 else cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    @staticmethod
    def hsv(img):
        """轉換為HSV"""
        return cv2.cvtColor(_ColorNorm.bgr(img),cv2.COLOR_BGR2HSV)
    @staticmethod
    def lab(img):
        """轉換為LAB"""
        return cv2.cvtColor(_ColorNorm.bgr(img),cv2.COLOR_BGR2LAB)
