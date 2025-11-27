import numpy as np
from typing import List,Tuple,Optional
class CTCLoss:
    def __init__(s,blank:int=0,reduction:str='mean'):s.bl,s.rd=blank,reduction
    def __call__(s,logits:np.ndarray,targets:List[List[int]],input_lens:List[int],target_lens:List[int])->float:
        B=logits.shape[0];losses=[]
        for b in range(B):
            lp=s._log_softmax(logits[b,:input_lens[b],:]);tgt=targets[b][:target_lens[b]]
            loss=s._ctc_single(lp,tgt);losses.append(loss)
        if s.rd=='mean':return float(np.mean(losses))
        elif s.rd=='sum':return float(np.sum(losses))
        return losses
    def _log_softmax(s,x:np.ndarray)->np.ndarray:
        mx=np.max(x,axis=-1,keepdims=True);ex=np.exp(x-mx);return x-mx-np.log(np.sum(ex,axis=-1,keepdims=True))
    def _ctc_single(s,log_probs:np.ndarray,target:List[int])->float:
        T,V=log_probs.shape;S=2*len(target)+1;labels=[s.bl]+[v for t in target for v in[t,s.bl]]
        alpha=np.full((T,S),-np.inf);alpha[0,0]=log_probs[0,labels[0]]
        if S>1:alpha[0,1]=log_probs[0,labels[1]]
        for t in range(1,T):
            for ss in range(S):
                alpha[t,ss]=alpha[t-1,ss]
                if ss>0:alpha[t,ss]=np.logaddexp(alpha[t,ss],alpha[t-1,ss-1])
                if ss>1 and labels[ss]!=s.bl and labels[ss]!=labels[ss-2]:
                    alpha[t,ss]=np.logaddexp(alpha[t,ss],alpha[t-1,ss-2])
                alpha[t,ss]+=log_probs[t,labels[ss]]
        if S>1:return-np.logaddexp(alpha[T-1,S-1],alpha[T-1,S-2])
        return-alpha[T-1,S-1]
class CELoss:
    def __init__(s,label_smoothing:float=0.0,ignore_idx:int=-100):s.ls,s.ig=label_smoothing,ignore_idx
    def __call__(s,logits:np.ndarray,targets:np.ndarray)->float:
        B,T,V=logits.shape if logits.ndim==3 else(*logits.shape[:2],logits.shape[-1])
        probs=s._softmax(logits.reshape(-1,V));targets=targets.flatten()
        mask=targets!=s.ig;targets=np.clip(targets,0,V-1)
        if s.ls>0:
            smooth=np.full_like(probs,s.ls/V);smooth[np.arange(len(targets)),targets]=1-s.ls+s.ls/V
            loss=-np.sum(smooth*np.log(probs+1e-6),axis=-1)
        else:loss=-np.log(probs[np.arange(len(targets)),targets]+1e-6)
        return float(loss[mask].mean())if mask.any()else 0.0
    def _softmax(s,x:np.ndarray)->np.ndarray:
        ex=np.exp(x-np.max(x,axis=-1,keepdims=True));return ex/np.sum(ex,axis=-1,keepdims=True)
class FocalLoss:
    def __init__(s,alpha:float=0.25,gamma:float=2.0):s.a,s.g=alpha,gamma
    def __call__(s,logits:np.ndarray,targets:np.ndarray)->float:
        probs=s._softmax(logits);B=logits.shape[0];targets=targets.flatten()
        p=probs[np.arange(B),targets];loss=-s.a*(1-p)**s.g*np.log(p+1e-6);return float(loss.mean())
    def _softmax(s,x:np.ndarray)->np.ndarray:
        ex=np.exp(x-np.max(x,axis=-1,keepdims=True));return ex/np.sum(ex,axis=-1,keepdims=True)
class _ACELoss:
    def __init__(s,weight:float=1.0):s.w=weight
    def __call__(s,logits:np.ndarray,targets:List[List[int]])->float:
        B,T,V=logits.shape;probs=np.exp(logits-np.max(logits,axis=-1,keepdims=True))
        probs=probs/probs.sum(axis=-1,keepdims=True);losses=[]
        for b in range(B):
            tgt=targets[b];agg=probs[b].mean(axis=0);gt=np.zeros(V)
            for t in tgt:gt[t]+=1
            gt=gt/gt.sum();loss=-np.sum(gt*np.log(agg+1e-6));losses.append(loss)
        return float(np.mean(losses))*s.w
