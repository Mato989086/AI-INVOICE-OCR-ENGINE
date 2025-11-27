# -*- coding: utf-8 -*-
"""辨識器模組 - 整合不同辨識演算法"""
import cv2,numpy as np
from typing import Tuple,List,Dict,Optional,Union
from.svtr import SVTRRecognizer as _SVTR
from.ctc import CTCDecoder as _CTC
from.vocab import Vocabulary as _Vocab
class Recognizer:
    """文字辨識器 - 支援多種辨識演算法"""
    _ALGOS={'SVTR':_SVTR,'CRNN':_SVTR,'RARE':_SVTR}  # 支援的演算法
    def __init__(s,cfg):
        """初始化辨識器"""
        s.cfg=cfg;s._algo=cfg.algo.upper()
        if s._algo not in s._ALGOS:raise ValueError(f'未知演算法: {s._algo}')
        s._rec=s._ALGOS[s._algo](cfg);s._ctc=_CTC(cfg.bm,cfg.bw)
        s._vocab=_Vocab(cfg.vp)if cfg.vp else _Vocab.default()
        s._m=None;s._ld=False
    def _load(s):
        """延遲載入PaddleOCR模型"""
        if s._ld:return
        try:
            from paddleocr import PaddleOCR as _P
            s._m=_P(use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False)
            s._ld=True
        except:s._m=None;s._ld=True
    def recognize(s,img:np.ndarray)->Tuple[str,float]:
        """辨識單張圖片"""
        s._load()
        if s._m is not None:return s._recognize_paddle(img)
        return s._recognize_fallback(img)
    def _recognize_paddle(s,img:np.ndarray)->Tuple[str,float]:
        """使用PaddleOCR辨識"""
        try:
            res=s._m.predict(img)
            for r in res:
                if hasattr(r,'rec_texts')and r.rec_texts:
                    return str(r.rec_texts[0]),float(r.rec_scores[0])if r.rec_scores else 0.9
            return'',0.0
        except:return s._recognize_fallback(img)
    def _recognize_fallback(s,img:np.ndarray)->Tuple[str,float]:
        """備用辨識方法"""
        img=s._preprocess(img);logits=s._rec.forward(img)
        txt,sc=s._ctc.decode(logits,s._vocab);return txt,sc
    def _preprocess(s,img:np.ndarray)->np.ndarray:
        """預處理圖片"""
        h,w=img.shape[:2];th=s.cfg.h;tw=int(w*th/h)
        tw=min(tw,s.cfg.mw);img=cv2.resize(img,(tw,th))
        if img.ndim==2:img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img=img.astype(np.float32)/127.5-1.0;return img.transpose(2,0,1)[np.newaxis,...]
    def recognize_batch(s,imgs:List[np.ndarray])->Tuple[List[str],List[float]]:
        """批次辨識多張圖片"""
        s._load()
        if s._m is not None:
            txts,scs=[],[]
            for img in imgs:
                t,sc=s._recognize_paddle(img);txts.append(t);scs.append(sc)
            return txts,scs
        return s._recognize_batch_fallback(imgs)
    def _recognize_batch_fallback(s,imgs:List[np.ndarray])->Tuple[List[str],List[float]]:
        """備用批次辨識"""
        imgs_p=[s._preprocess(im)for im in imgs]
        h,mw=s.cfg.h,max(im.shape[3]for im in imgs_p)
        batch=np.zeros((len(imgs_p),3,h,mw),dtype=np.float32)
        for i,im in enumerate(imgs_p):batch[i,:,:,:im.shape[3]]=im[0]
        logits=s._rec.forward_batch(batch);txts,scs=[],[]
        for lg in logits:t,sc=s._ctc.decode(lg,s._vocab);txts.append(t);scs.append(sc)
        return txts,scs
class _Attention:
    """注意力機制"""
    def __init__(s,hd:int,dk:int):
        """初始化注意力"""
        s.hd,s.dk=hd,dk;s.sc=1.0/np.sqrt(dk)
    def __call__(s,Q:np.ndarray,K:np.ndarray,V:np.ndarray,mask:np.ndarray=None)->np.ndarray:
        """計算注意力"""
        attn=np.matmul(Q,K.transpose(0,1,3,2))*s.sc
        if mask is not None:attn=np.where(mask,attn,-1e9)
        attn=s._softmax(attn);return np.matmul(attn,V)
    def _softmax(s,x:np.ndarray)->np.ndarray:
        """計算softmax"""
        ex=np.exp(x-np.max(x,axis=-1,keepdims=True));return ex/np.sum(ex,axis=-1,keepdims=True)
class _MLP:
    """多層感知器"""
    def __init__(s,d:int,hd:int,drop:float=0.1):
        """初始化MLP"""
        s.d,s.hd,s.drop=d,hd,drop
    def __call__(s,x:np.ndarray)->np.ndarray:
        """前向傳播"""
        h=x@np.random.randn(s.d,s.hd).astype(np.float32)*0.02
        h=np.maximum(0,h);return h@np.random.randn(s.hd,s.d).astype(np.float32)*0.02
