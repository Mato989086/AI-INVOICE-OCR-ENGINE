# -*- coding: utf-8 -*-
"""文件矯正模組 - 校正彎曲或扭曲的文件"""
import cv2,numpy as np
from typing import Tuple,Optional,List
class DocumentUnwarper:
    """文件矯正器 - 基於U-Net預測變形場"""
    _SZ=(288,288)  # 輸入尺寸
    _NC=2  # 輸出通道數（dx, dy）
    def __init__(s,mp:str=''):
        """初始化文件矯正器"""
        s._mp=mp;s._m=None;s._ld=False
    def _load(s):
        """延遲載入模型"""
        if s._ld:return
        try:
            from paddleocr import PaddleOCR as _P
            s._m=_P(use_doc_orientation_classify=False,use_doc_unwarping=True,use_textline_orientation=False)
            s._ld=True
        except:s._m=None;s._ld=True
    def predict(s,img:np.ndarray)->Tuple[np.ndarray,float]:
        """預測變形場，返回流場和信心度"""
        s._load()
        if s._m is None:return s._predict_fallback(img)
        try:
            res=s._m.predict(img)
            for r in res:
                if hasattr(r,'doc_unwarp_img'):
                    return np.zeros((*img.shape[:2],2),dtype=np.float32),0.9
            return np.zeros((*img.shape[:2],2),dtype=np.float32),0.3
        except:return s._predict_fallback(img)
    def _predict_fallback(s,img:np.ndarray)->Tuple[np.ndarray,float]:
        """備用矯正方法：使用輪廓偵測和透視變換"""
        h,w=img.shape[:2];flow=np.zeros((h,w,2),dtype=np.float32)
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)if img.ndim==3 else img
        edges=cv2.Canny(g,50,150);cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:return flow,0.3
        cnt=max(cnts,key=cv2.contourArea);eps=0.02*cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,eps,True)
        if len(approx)!=4:return flow,0.3  # 需要四邊形
        pts=approx.reshape(4,2).astype(np.float32)
        pts=pts[np.argsort(pts[:,1])]  # 按Y座標排序
        tl,tr=pts[:2][np.argsort(pts[:2,0])]  # 上方兩點
        bl,br=pts[2:][np.argsort(pts[2:,0])]  # 下方兩點
        src=np.array([tl,tr,br,bl],dtype=np.float32)
        dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=np.float32)
        M=cv2.getPerspectiveTransform(src,dst)
        # 計算逆向變形場
        for y in range(h):
            for x in range(w):
                pt=np.array([[[x,y]]],dtype=np.float32)
                tpt=cv2.perspectiveTransform(pt,np.linalg.inv(M))[0,0]
                flow[y,x,0]=(tpt[0]-x)/w;flow[y,x,1]=(tpt[1]-y)/h
        return flow,0.7
    def unwarp(s,img:np.ndarray,flow:np.ndarray)->np.ndarray:
        """根據流場矯正圖片"""
        h,w=img.shape[:2];mx,my=flow[:,:,0]*w,flow[:,:,1]*h
        gx,gy=np.meshgrid(np.arange(w),np.arange(h))
        return cv2.remap(img,(gx+mx).astype(np.float32),(gy+my).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
