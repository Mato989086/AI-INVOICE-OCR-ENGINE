# -*- coding: utf-8 -*-
"""OCR引擎核心模組 - 整合偵測、辨識、後處理流程"""
import os,sys,time,logging
from typing import List,Dict,Any,Optional,Union,Tuple,Callable
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from.config import Config
from.preprocess import Preprocessor as _Prep
from.detect import Detector as _Det
from.recognize import Recognizer as _Rec
from.postprocess import PostProcessor as _Post
from.utils.image import imread as _im,imwrite as _iw
from.utils.logger import get_logger as _gl
_L=_gl(__name__)
class OCREngine:
    """OCR引擎主類別 - 單例模式實作"""
    _inst=None;_cfg=None
    def __new__(c,cfg:Config=None):
        # 單例模式：確保只有一個實例
        if c._inst is None or(cfg and hash(cfg)!=hash(c._cfg)):c._inst=super().__new__(c);c._cfg=cfg
        return c._inst
    def __init__(s,cfg:Config=None):
        # 防止重複初始化
        if hasattr(s,'_init')and s._init:return
        s.cfg=cfg or Config();s._prep=_Prep(s.cfg.prep);s._det=_Det(s.cfg.det)
        s._rec=_Rec(s.cfg.rec);s._post=_Post();s._init=True;s._cache={};s._stats={'n':0,'t':0}
        _L.info(f'OCR引擎初始化: 模式={s.cfg.mode.name}, 語言={s.cfg.lang}')
    def _preprocess(s,img:np.ndarray)->Tuple[np.ndarray,Dict]:
        """前處理：方向校正、扭曲矯正、正規化"""
        m={};t0=time.perf_counter()
        if s.cfg.prep.ori:img,m['ori']=s._prep.correct_orientation(img)
        if s.cfg.prep.uwp:img,m['uwp']=s._prep.unwarp(img)
        img,m['rsz']=s._prep.resize(img);img=s._prep.normalize(img);m['t']=time.perf_counter()-t0;return img,m
    def _detect(s,img:np.ndarray)->Tuple[List[np.ndarray],Dict]:
        """文字偵測：找出文字區域"""
        t0=time.perf_counter();boxes,scores=s._det.detect(img)
        boxes=[b for b,sc in zip(boxes,scores)if sc>=s.cfg.det.bth]
        return boxes,{'n':len(boxes),'t':time.perf_counter()-t0}
    def _recognize(s,img:np.ndarray,boxes:List[np.ndarray])->Tuple[List[str],List[float],Dict]:
        """文字辨識：辨識裁切區域內的文字"""
        t0=time.perf_counter();crops=[s._post.crop_poly(img,b)for b in boxes]
        if s.cfg.cls.en:crops=[s._prep.correct_textline(c)for c in crops]
        txts,scs=s._rec.recognize_batch(crops);return txts,scs,{'n':len(txts),'t':time.perf_counter()-t0}
    def predict(s,inp:Union[str,np.ndarray,List],**kw)->List[Dict[str,Any]]:
        """執行OCR預測 - 支援單張圖片、目錄或陣列輸入"""
        if isinstance(inp,str):inp=[inp]if os.path.isfile(inp)else[os.path.join(inp,f)for f in os.listdir(inp)if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
        if isinstance(inp,np.ndarray):inp=[inp]
        res=[];nth=kw.get('num_threads',s.cfg.nth)
        def _proc(x):
            # 處理單張圖片
            img=_im(x)if isinstance(x,str)else x;h,w=img.shape[:2];t0=time.perf_counter()
            img_p,pm=s._preprocess(img.copy());boxes,dm=s._detect(img_p)
            boxes=s._post.scale_boxes(boxes,pm.get('rsz',{}).get('scale',1.0))
            txts,scs,rm=s._recognize(img,boxes);s._stats['n']+=1;s._stats['t']+=time.perf_counter()-t0
            return{'path':x if isinstance(x,str)else None,'size':(w,h),'boxes':boxes,'texts':txts,'scores':scs,'meta':{'prep':pm,'det':dm,'rec':rm}}
        # 多執行緒處理
        if nth>1 and len(inp)>1:
            with ThreadPoolExecutor(max_workers=nth)as ex:res=list(ex.map(_proc,inp))
        else:res=[_proc(x)for x in inp]
        return res
    def __call__(s,*a,**kw):
        """允許直接呼叫實例"""
        return s.predict(*a,**kw)
    def detect_only(s,img:Union[str,np.ndarray])->List[np.ndarray]:
        """僅執行文字偵測"""
        img=_im(img)if isinstance(img,str)else img;img_p,pm=s._preprocess(img.copy())
        boxes,_=s._detect(img_p);return s._post.scale_boxes(boxes,pm.get('rsz',{}).get('scale',1.0))
    def recognize_only(s,crops:List[np.ndarray])->Tuple[List[str],List[float]]:
        """僅執行文字辨識"""
        if s.cfg.cls.en:crops=[s._prep.correct_textline(c)for c in crops]
        return s._rec.recognize_batch(crops)
    def get_stats(s)->Dict:
        """取得統計資訊"""
        return{**s._stats,'avg':s._stats['t']/max(s._stats['n'],1)}
    def reset_stats(s):
        """重置統計"""
        s._stats={'n':0,'t':0}
    def warmup(s,n:int=3):
        """預熱模型"""
        dummy=np.random.randint(0,255,(480,640,3),dtype=np.uint8)
        for _ in range(n):s.predict(dummy)
        s.reset_stats();_L.info('預熱完成')
    @property
    def config(s)->Config:
        """取得設定"""
        return s.cfg
    def update_config(s,**kw):
        """更新設定參數"""
        for k,v in kw.items():
            if'.'in k:p,a=k.rsplit('.',1);o=getattr(s.cfg,p,None);o and setattr(o,a,v)
            else:hasattr(s.cfg,k)and setattr(s.cfg,k,v)
