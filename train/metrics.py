import numpy as np
from typing import Dict,List,Tuple,Optional
from shapely.geometry import Polygon as _Poly
class DetMetrics:
    def __init__(s,iou_th:float=0.5):s.th=iou_th;s.reset()
    def reset(s):s._tp,s._fp,s._fn,s._gt_n,s._pred_n=0,0,0,0,0
    def update(s,pred_boxes:List[np.ndarray],gt_boxes:List[np.ndarray],gt_ignore:List[bool]=None):
        if gt_ignore is None:gt_ignore=[False]*len(gt_boxes)
        matched_gt=set();s._gt_n+=sum(1 for i in gt_ignore if not i);s._pred_n+=len(pred_boxes)
        for pb in pred_boxes:
            best_iou,best_idx=0,-1
            for j,(gb,ig)in enumerate(zip(gt_boxes,gt_ignore)):
                if ig or j in matched_gt:continue
                iou=s._iou(pb,gb)
                if iou>best_iou:best_iou,best_idx=iou,j
            if best_iou>=s.th:s._tp+=1;matched_gt.add(best_idx)
            else:s._fp+=1
        s._fn+=len([i for i,ig in enumerate(gt_ignore)if not ig and i not in matched_gt])
    def _iou(s,b1:np.ndarray,b2:np.ndarray)->float:
        try:
            p1,p2=_Poly(b1),_Poly(b2)
            if not p1.is_valid or not p2.is_valid:return 0
            inter=p1.intersection(p2).area;union=p1.union(p2).area
            return inter/union if union>0 else 0
        except:return 0
    def compute(s)->Dict[str,float]:
        p=s._tp/(s._tp+s._fp+1e-6);r=s._tp/(s._tp+s._fn+1e-6);f1=2*p*r/(p+r+1e-6)
        return{'precision':p,'recall':r,'f1':f1,'tp':s._tp,'fp':s._fp,'fn':s._fn}
class RecMetrics:
    def __init__(s):s.reset()
    def reset(s):s._correct,s._total,s._ed_sum,s._len_sum,s._cer_sum=0,0,0,0,0
    def update(s,preds:List[str],gts:List[str]):
        for p,g in zip(preds,gts):
            s._total+=1
            if p==g:s._correct+=1
            ed=s._edit_distance(p,g);s._ed_sum+=ed;s._len_sum+=max(len(p),len(g))
            s._cer_sum+=ed/max(len(g),1)
    def _edit_distance(s,a:str,b:str)->int:
        m,n=len(a),len(b);dp=[[0]*(n+1)for _ in range(m+1)]
        for i in range(m+1):dp[i][0]=i
        for j in range(n+1):dp[0][j]=j
        for i in range(1,m+1):
            for j in range(1,n+1):
                if a[i-1]==b[j-1]:dp[i][j]=dp[i-1][j-1]
                else:dp[i][j]=1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
        return dp[m][n]
    def compute(s)->Dict[str,float]:
        acc=s._correct/max(s._total,1);ned=1-s._ed_sum/max(s._len_sum,1);cer=s._cer_sum/max(s._total,1)
        return{'accuracy':acc,'ned':ned,'cer':cer,'correct':s._correct,'total':s._total}
class ClsMetrics:
    def __init__(s,n_cls:int=4):s.nc=n_cls;s.reset()
    def reset(s):s._cm=np.zeros((s.nc,s.nc),dtype=np.int32)
    def update(s,preds:np.ndarray,gts:np.ndarray):
        for p,g in zip(preds.flatten(),gts.flatten()):
            if 0<=p<s.nc and 0<=g<s.nc:s._cm[g,p]+=1
    def compute(s)->Dict[str,float]:
        acc=np.diag(s._cm).sum()/max(s._cm.sum(),1);per_cls={}
        for i in range(s.nc):
            tp=s._cm[i,i];fp=s._cm[:,i].sum()-tp;fn=s._cm[i,:].sum()-tp
            p=tp/(tp+fp+1e-6);r=tp/(tp+fn+1e-6);per_cls[f'cls_{i}']={'p':p,'r':r,'f1':2*p*r/(p+r+1e-6)}
        return{'accuracy':acc,'confusion_matrix':s._cm.tolist(),'per_class':per_cls}
class _mAP:
    def __init__(s,iou_ths:List[float]=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]):s.ths=iou_ths
    def compute(s,pred_boxes:List,pred_scores:List,gt_boxes:List)->float:
        aps=[]
        for th in s.ths:
            m=DetMetrics(th);m.update(pred_boxes,gt_boxes);r=m.compute();aps.append(r['precision'])
        return float(np.mean(aps))
