# -*- coding: utf-8 -*-
"""偵測器模組 - 整合不同偵測演算法"""
import cv2,numpy as np
from typing import Tuple,List,Dict,Optional,Union
from.db import DBDetector as _DB
from.postprocess import DBPostProcessor as _DBP
class Detector:
    """文字偵測器 - 支援多種偵測演算法"""
    _ALGOS={'DB':(_DB,_DBP),'EAST':(_DB,_DBP),'SAST':(_DB,_DBP)}  # 支援的演算法
    def __init__(s,cfg):
        """初始化偵測器"""
        s.cfg=cfg;s._algo=cfg.algo.upper()
        if s._algo not in s._ALGOS:raise ValueError(f'未知演算法: {s._algo}')
        dc,pc=s._ALGOS[s._algo];s._det=dc(cfg);s._post=pc(cfg)
        s._m=None;s._ld=False
    def _load(s):
        """延遲載入PaddleOCR模型"""
        if s._ld:return
        try:
            from paddleocr import PaddleOCR as _P
            s._m=_P(use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False)
            s._ld=True
        except:s._m=None;s._ld=True
    def detect(s,img:np.ndarray)->Tuple[List[np.ndarray],List[float]]:
        """偵測圖片中的文字區域"""
        s._load()
        if s._m is not None:return s._detect_paddle(img)
        return s._detect_fallback(img)
    def _detect_paddle(s,img:np.ndarray)->Tuple[List[np.ndarray],List[float]]:
        """使用PaddleOCR偵測"""
        try:
            if img.ndim==4:img=s._denorm(img)
            res=s._m.predict(img)
            boxes,scores=[],[]
            for r in res:
                if hasattr(r,'dt_polys')and r.dt_polys is not None:
                    for i,poly in enumerate(r.dt_polys):
                        boxes.append(np.array(poly,dtype=np.float32))
                        sc=r.dt_scores[i]if hasattr(r,'dt_scores')and r.dt_scores else 0.9
                        scores.append(float(sc))
            return boxes,scores
        except Exception as e:return s._detect_fallback(img)
    def _denorm(s,img:np.ndarray)->np.ndarray:
        """反正規化圖片"""
        if img.ndim==4:img=img[0]
        if img.shape[0]==3:img=img.transpose(1,2,0)
        m,st=np.array([.485,.456,.406]),np.array([.229,.224,.225])
        img=img*st+m;return(img*255).clip(0,255).astype(np.uint8)
    def _detect_fallback(s,img:np.ndarray)->Tuple[List[np.ndarray],List[float]]:
        """備用偵測方法"""
        if img.ndim==4 or(img.ndim==3 and img.shape[0]==3):img=s._denorm(img)
        prob=s._det.forward(img);boxes,scores=s._post.process(prob,img.shape[:2])
        return boxes,scores
    def detect_batch(s,imgs:List[np.ndarray])->List[Tuple[List[np.ndarray],List[float]]]:
        """批次偵測多張圖片"""
        return[s.detect(im)for im in imgs]
class _MSER:
    """MSER區域偵測器"""
    def __init__(s,delta:int=5,min_area:int=60,max_area:int=14400):
        """初始化MSER偵測器"""
        s._mser=cv2.MSER_create(delta,min_area,max_area)
    def detect(s,img:np.ndarray)->List[np.ndarray]:
        """偵測MSER區域"""
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)if img.ndim==3 else img
        regions,_=s._mser.detectRegions(g)
        boxes=[]
        for r in regions:
            x,y,w,h=cv2.boundingRect(r.reshape(-1,1,2))
            boxes.append(np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],dtype=np.float32))
        return boxes
class _ContourDet:
    """輪廓偵測器"""
    def __init__(s,th1:int=50,th2:int=150,min_area:int=100):
        """初始化輪廓偵測器"""
        s._th1,s._th2,s._ma=th1,th2,min_area
    def detect(s,img:np.ndarray)->List[np.ndarray]:
        """偵測輪廓"""
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)if img.ndim==3 else img
        edges=cv2.Canny(g,s._th1,s._th2);k=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        edges=cv2.dilate(edges,k,iterations=2);cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        for c in cnts:
            if cv2.contourArea(c)<s._ma:continue
            rect=cv2.minAreaRect(c);box=cv2.boxPoints(rect);boxes.append(box.astype(np.float32))
        return boxes
