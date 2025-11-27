import cv2,numpy as np
from typing import List,Dict,Tuple,Optional
class PostProcessor:
    def __init__(s):s._cache={}
    def crop_poly(s,img:np.ndarray,poly:np.ndarray)->np.ndarray:
        poly=poly.astype(np.float32);pts=s._order_points(poly)
        tw=int(max(np.linalg.norm(pts[0]-pts[1]),np.linalg.norm(pts[2]-pts[3])))
        th=int(max(np.linalg.norm(pts[0]-pts[3]),np.linalg.norm(pts[1]-pts[2])))
        tw,th=max(tw,1),max(th,1);dst=np.array([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]],dtype=np.float32)
        M=cv2.getPerspectiveTransform(pts,dst);return cv2.warpPerspective(img,M,(tw,th))
    def _order_points(s,pts:np.ndarray)->np.ndarray:
        rect=np.zeros((4,2),dtype=np.float32);sm=pts.sum(axis=1);df=np.diff(pts,axis=1)
        rect[0]=pts[np.argmin(sm)];rect[2]=pts[np.argmax(sm)];rect[1]=pts[np.argmin(df)];rect[3]=pts[np.argmax(df)]
        return rect
    def scale_boxes(s,boxes:List[np.ndarray],scale:float)->List[np.ndarray]:
        if abs(scale-1.0)<1e-6:return boxes
        return[b/scale for b in boxes]
    def sort_boxes(s,boxes:List[np.ndarray],mode:str='tb_lr')->List[int]:
        if not boxes:return[]
        if mode=='tb_lr':key=lambda i:(boxes[i][:,1].mean(),boxes[i][:,0].mean())
        elif mode=='lr_tb':key=lambda i:(boxes[i][:,0].mean(),boxes[i][:,1].mean())
        else:key=lambda i:i
        return sorted(range(len(boxes)),key=key)
    def filter_boxes(s,boxes:List[np.ndarray],scores:List[float],min_score:float=0.5,min_size:int=3)->Tuple[List[np.ndarray],List[float]]:
        fb,fs=[],[]
        for b,sc in zip(boxes,scores):
            if sc<min_score:continue
            w,h=b[:,0].max()-b[:,0].min(),b[:,1].max()-b[:,1].min()
            if min(w,h)<min_size:continue
            fb.append(b);fs.append(sc)
        return fb,fs
    def merge_boxes(s,boxes:List[np.ndarray],iou_th:float=0.5)->List[np.ndarray]:
        if len(boxes)<2:return boxes
        merged=[];used=set()
        for i,b1 in enumerate(boxes):
            if i in used:continue
            group=[b1];used.add(i)
            for j,b2 in enumerate(boxes[i+1:],i+1):
                if j in used:continue
                if s._iou(b1,b2)>iou_th:group.append(b2);used.add(j)
            merged.append(s._merge_group(group))
        return merged
    def _iou(s,b1:np.ndarray,b2:np.ndarray)->float:
        try:
            from shapely.geometry import Polygon
            p1,p2=Polygon(b1),Polygon(b2)
            if not p1.is_valid or not p2.is_valid:return 0
            inter=p1.intersection(p2).area;union=p1.area+p2.area-inter
            return inter/union if union>0 else 0
        except:return 0
    def _merge_group(s,boxes:List[np.ndarray])->np.ndarray:
        pts=np.vstack(boxes);x1,y1=pts.min(axis=0);x2,y2=pts.max(axis=0)
        return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
class _Reading:
    @staticmethod
    def sort_reading_order(boxes:List[np.ndarray],texts:List[str],scores:List[float],line_th:float=10)->Tuple[List[np.ndarray],List[str],List[float]]:
        if not boxes:return[],[],[]
        items=list(zip(boxes,texts,scores));items.sort(key=lambda x:(x[0][:,1].mean(),x[0][:,0].mean()))
        lines=[];curr_line=[];curr_y=items[0][0][:,1].mean()
        for b,t,s in items:
            y=b[:,1].mean()
            if abs(y-curr_y)>line_th:
                if curr_line:lines.append(sorted(curr_line,key=lambda x:x[0][:,0].mean()));curr_line=[]
                curr_y=y
            curr_line.append((b,t,s))
        if curr_line:lines.append(sorted(curr_line,key=lambda x:x[0][:,0].mean()))
        result=[];[result.extend(line)for line in lines]
        return[r[0]for r in result],[r[1]for r in result],[r[2]for r in result]
    @staticmethod
    def group_paragraphs(boxes:List[np.ndarray],texts:List[str],para_th:float=30)->List[List[str]]:
        if not boxes:return[]
        items=list(zip(boxes,texts));items.sort(key=lambda x:x[0][:,1].mean())
        paras=[];curr_para=[];prev_y=items[0][0][:,1].max()
        for b,t in items:
            y=b[:,1].min()
            if y-prev_y>para_th and curr_para:paras.append(curr_para);curr_para=[]
            curr_para.append(t);prev_y=b[:,1].max()
        if curr_para:paras.append(curr_para)
        return paras
