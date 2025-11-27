# -*- coding: utf-8 -*-
"""偵測後處理模組 - 從機率圖提取文字框"""
import cv2,numpy as np
from typing import Tuple,List,Optional
import pyclipper
from shapely.geometry import Polygon as _Poly
class DBPostProcessor:
    """DB後處理器 - 提取並擴展文字區域"""
    def __init__(s,cfg):
        """初始化後處理器"""
        s.cfg=cfg;s._th=cfg.th;s._bth=cfg.bth;s._ur=cfg.ur;s._ms=cfg.ms;s._dil=cfg.dil
    def process(s,prob:np.ndarray,orig_sz:Tuple[int,int])->Tuple[List[np.ndarray],List[float]]:
        """處理機率圖，返回框和分數"""
        if prob.ndim==3:prob=prob[0]
        h,w=prob.shape[:2];oh,ow=orig_sz;mask=(prob>s._th).astype(np.uint8)*255
        # 膨脹操作增強連通性
        if s._dil>0:k=cv2.getStructuringElement(cv2.MORPH_RECT,(s._dil*2+1,s._dil*2+1));mask=cv2.dilate(mask,k)
        cnts,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        boxes,scores=[],[]
        for cnt in cnts:
            if len(cnt)<4:continue
            sc=s._box_score(prob,cnt)  # 計算框內平均機率
            if sc<s._bth:continue
            box=s._unclip(cnt,s._ur)  # 擴展框
            if box is None:continue
            box=s._get_mini_box(box)  # 取得最小外接矩形
            if box is None or min(box[:,0].max()-box[:,0].min(),box[:,1].max()-box[:,1].min())<s._ms:continue
            # 縮放回原始尺寸
            box[:,0]=np.clip(box[:,0]*ow/w,0,ow);box[:,1]=np.clip(box[:,1]*oh/h,0,oh)
            boxes.append(box);scores.append(sc)
        return boxes,scores
    def _box_score(s,prob:np.ndarray,cnt:np.ndarray)->float:
        """計算框內平均機率分數"""
        h,w=prob.shape[:2];mask=np.zeros((h,w),dtype=np.uint8)
        cv2.fillPoly(mask,[cnt.reshape(-1,1,2)],1);return cv2.mean(prob,mask)[0]
    def _unclip(s,cnt:np.ndarray,ratio:float)->Optional[np.ndarray]:
        """使用Vatti裁剪演算法擴展多邊形"""
        try:
            poly=_Poly(cnt.reshape(-1,2))
            if not poly.is_valid or poly.area<1:return None
            d=poly.area*ratio/poly.length;offset=pyclipper.PyclipperOffset()
            offset.AddPath(cnt.reshape(-1,2).tolist(),pyclipper.JT_ROUND,pyclipper.ET_CLOSEDPOLYGON)
            expanded=offset.Execute(d)
            if not expanded:return None
            return np.array(expanded[0])
        except:return None
    def _get_mini_box(s,cnt:np.ndarray)->Optional[np.ndarray]:
        """取得最小面積外接矩形"""
        try:
            rect=cv2.minAreaRect(cnt.reshape(-1,1,2));box=cv2.boxPoints(rect)
            box=s._order_points(box);return box.astype(np.float32)
        except:return None
    def _order_points(s,pts:np.ndarray)->np.ndarray:
        """按左上、右上、右下、左下順序排列點"""
        rect=np.zeros((4,2),dtype=np.float32);sm=pts.sum(axis=1);df=np.diff(pts,axis=1)
        rect[0]=pts[np.argmin(sm)];rect[2]=pts[np.argmax(sm)]  # 左上、右下
        rect[1]=pts[np.argmin(df)];rect[3]=pts[np.argmax(df)];return rect  # 右上、左下
class _NMS:
    """非極大值抑制"""
    @staticmethod
    def nms(boxes:List[np.ndarray],scores:List[float],th:float=0.5)->List[int]:
        """執行NMS，返回保留的索引"""
        if not boxes:return[]
        idxs=np.argsort(scores)[::-1];keep=[]
        while len(idxs)>0:
            i=idxs[0];keep.append(i)
            if len(idxs)==1:break
            ious=np.array([_NMS._iou(boxes[i],boxes[j])for j in idxs[1:]])
            idxs=idxs[1:][ious<th]
        return keep
    @staticmethod
    def _iou(b1:np.ndarray,b2:np.ndarray)->float:
        """計算兩個多邊形的IoU"""
        try:
            p1,p2=_Poly(b1),_Poly(b2)
            if not p1.is_valid or not p2.is_valid:return 0
            inter=p1.intersection(p2).area;union=p1.area+p2.area-inter
            return inter/union if union>0 else 0
        except:return 0
class _BoxMerger:
    """文字框合併器"""
    @staticmethod
    def merge_horizontal(boxes:List[np.ndarray],th_y:float=10,th_x:float=50)->List[np.ndarray]:
        """合併水平方向相近的框"""
        if len(boxes)<2:return boxes
        boxes=sorted(boxes,key=lambda b:(b[:,1].mean(),b[:,0].min()))
        merged=[];used=set()
        for i,b1 in enumerate(boxes):
            if i in used:continue
            group=[b1];used.add(i)
            for j,b2 in enumerate(boxes[i+1:],i+1):
                if j in used:continue
                # 檢查Y座標和X距離
                if abs(b1[:,1].mean()-b2[:,1].mean())<th_y and b2[:,0].min()-b1[:,0].max()<th_x:
                    group.append(b2);used.add(j);b1=_BoxMerger._merge_boxes(group)
            merged.append(_BoxMerger._merge_boxes(group))
        return merged
    @staticmethod
    def _merge_boxes(boxes:List[np.ndarray])->np.ndarray:
        """合併多個框為一個"""
        pts=np.vstack(boxes);x1,y1=pts.min(axis=0);x2,y2=pts.max(axis=0)
        return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
