# -*- coding: utf-8 -*-
"""方向分類模組 - 文件和文字行方向偵測"""
import cv2,numpy as np
from typing import Tuple,List,Optional
class OrientationClassifier:
    """方向分類器 - 偵測文件旋轉角度"""
    _SZ=(192,48)  # 輸入尺寸
    _NC=4  # 文件方向類別數（0°,90°,180°,270°）
    _NCL=2  # 文字行方向類別數（0°,180°）
    def __init__(s,mp:str=''):
        """初始化方向分類器"""
        s._mp=mp;s._m=None;s._ml=None;s._ld=False
    def _load(s):
        """延遲載入模型"""
        if s._ld:return
        try:
            from paddleocr import PaddleOCR as _P
            s._m=_P(use_doc_orientation_classify=True,use_doc_unwarping=False,use_textline_orientation=False)
            s._ld=True
        except:s._m=None;s._ld=True
    def _extract_feat(s,img:np.ndarray)->np.ndarray:
        """提取特徵向量"""
        img=cv2.resize(img,s._SZ);img=img.astype(np.float32)/255.0
        m,st=np.array([.485,.456,.406]),np.array([.229,.224,.225])
        img=(img-m)/st;return img.transpose(2,0,1)[np.newaxis,...].astype(np.float32)
    def _softmax(s,x:np.ndarray)->np.ndarray:
        """計算softmax機率"""
        ex=np.exp(x-np.max(x,axis=-1,keepdims=True))
        return ex/np.sum(ex,axis=-1,keepdims=True)
    def classify(s,img:np.ndarray)->Tuple[int,float]:
        """分類文件方向，返回類別和信心度"""
        s._load()
        if s._m is None:return s._classify_fallback(img)
        try:
            res=s._m.predict(img,use_doc_orientation_classify=True)
            for r in res:
                if hasattr(r,'doc_orientation_cls'):
                    return int(r.doc_orientation_cls),float(getattr(r,'doc_orientation_score',0.9))
            return 0,0.99
        except:return s._classify_fallback(img)
    def _classify_fallback(s,img:np.ndarray)->Tuple[int,float]:
        """備用分類方法：使用邊緣偵測和霍夫線轉換"""
        h,w=img.shape[:2];g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)if img.ndim==3 else img
        edges=cv2.Canny(g,50,150);lines=cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=w//4,maxLineGap=10)
        if lines is None:return 0,0.5
        angles=[np.arctan2(l[0][3]-l[0][1],l[0][2]-l[0][0])*180/np.pi for l in lines]
        angles=[a%90 for a in angles];ma=np.median(angles)
        if ma<15:return 0,0.7
        elif ma<35:return 1,0.6
        elif ma<55:return 0,0.5
        elif ma<75:return 3,0.6
        return 0,0.7
    def classify_textline(s,img:np.ndarray)->Tuple[int,float]:
        """分類文字行方向（正向或翻轉180°）"""
        h,w=img.shape[:2]
        if h>w*2:return 0,0.5  # 太窄的圖片不處理
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)if img.ndim==3 else img
        lh,rh=g[:,:w//3],g[:,2*w//3:]  # 取左右兩側區域
        lv,rv=np.var(lh),np.var(rh)  # 計算變異數
        if abs(lv-rv)<50:return 0,0.6
        return(1,0.7)if lv>rv else(0,0.7)
    def classify_batch(s,imgs:List[np.ndarray])->List[Tuple[int,float]]:
        """批次分類多張圖片"""
        return[s.classify(im)for im in imgs]
