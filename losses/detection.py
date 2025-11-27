import numpy as np
from typing import Dict,Tuple,Optional
class DBLoss:
    def __init__(s,alpha:float=5.0,beta:float=10.0,ohem_ratio:float=3.0):s.a,s.b,s.or_=alpha,beta,ohem_ratio
    def __call__(s,pred:Dict,gt:Dict)->Tuple[float,Dict]:
        P,T,B=pred['prob'],pred['thresh'],pred['binary']
        gt_prob,gt_thresh,gt_mask=gt['prob'],gt['thresh'],gt['mask']
        l_bce=s._bce_ohem(P,gt_prob,gt_mask);l_dice=s._dice(B,gt_prob,gt_mask)
        l_thresh=s._l1(T,gt_thresh,gt_mask);total=l_bce+s.a*l_dice+s.b*l_thresh
        return float(total),{'bce':float(l_bce),'dice':float(l_dice),'thresh':float(l_thresh)}
    def _bce_ohem(s,pred:np.ndarray,gt:np.ndarray,mask:np.ndarray)->float:
        eps=1e-6;pred=np.clip(pred,eps,1-eps);loss=-gt*np.log(pred)-(1-gt)*np.log(1-pred)
        loss=loss*mask;pos_loss=loss[gt>0.5];neg_loss=loss[gt<=0.5]
        n_pos=pos_loss.size;n_neg=min(int(n_pos*s.or_),neg_loss.size)
        if n_neg>0:neg_loss=np.sort(neg_loss.flatten())[-n_neg:]
        return(pos_loss.sum()+neg_loss.sum())/(n_pos+n_neg+eps)
    def _dice(s,pred:np.ndarray,gt:np.ndarray,mask:np.ndarray)->float:
        pred,gt=pred*mask,gt*mask;inter=2*(pred*gt).sum();union=pred.sum()+gt.sum()+1e-6
        return 1-inter/union
    def _l1(s,pred:np.ndarray,gt:np.ndarray,mask:np.ndarray)->float:
        diff=np.abs(pred-gt)*mask;return diff.sum()/(mask.sum()+1e-6)
class DiceLoss:
    def __init__(s,smooth:float=1.0):s.sm=smooth
    def __call__(s,pred:np.ndarray,gt:np.ndarray,mask:np.ndarray=None)->float:
        if mask is not None:pred,gt=pred*mask,gt*mask
        inter=2*(pred*gt).sum()+s.sm;union=pred.sum()+gt.sum()+s.sm;return 1-inter/union
class BCELoss:
    def __init__(s,reduction:str='mean'):s.rd=reduction
    def __call__(s,pred:np.ndarray,gt:np.ndarray)->float:
        eps=1e-6;pred=np.clip(pred,eps,1-eps);loss=-gt*np.log(pred)-(1-gt)*np.log(1-pred)
        if s.rd=='mean':return loss.mean()
        elif s.rd=='sum':return loss.sum()
        return loss
class _IoULoss:
    def __call__(s,pred:np.ndarray,gt:np.ndarray)->float:
        inter=(np.minimum(pred,gt)).sum();union=(np.maximum(pred,gt)).sum()+1e-6;return 1-inter/union
class _MaskL1Loss:
    def __call__(s,pred:np.ndarray,gt:np.ndarray,mask:np.ndarray)->float:
        diff=np.abs(pred-gt)*mask;return diff.sum()/(mask.sum()+1e-6)
