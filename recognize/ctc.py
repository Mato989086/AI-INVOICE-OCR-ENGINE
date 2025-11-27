# -*- coding: utf-8 -*-
"""CTC解碼模組 - 連結時序分類"""
import numpy as np
from typing import Tuple,List,Optional
from.vocab import Vocabulary as _Vocab
class CTCDecoder:
    """CTC解碼器 - 支援貪婪和束搜尋"""
    def __init__(s,mode:str='greedy',beam_width:int=5):
        """初始化解碼器"""
        s.mode,s.bw=mode.lower(),beam_width
    def decode(s,logits:np.ndarray,vocab:_Vocab)->Tuple[str,float]:
        """解碼logits為文字"""
        if logits.ndim==3:logits=logits[0]
        if s.mode=='beam':return s._beam_search(logits,vocab)
        return s._greedy(logits,vocab)
    def _greedy(s,logits:np.ndarray,vocab:_Vocab)->Tuple[str,float]:
        """貪婪解碼 - 每個時間步取最大機率"""
        probs=s._softmax(logits);preds=np.argmax(probs,axis=-1);scores=probs.max(axis=-1)
        chars,scs=[],[]
        prev=-1
        for i,(p,sc)in enumerate(zip(preds,scores)):
            if p!=0 and p!=prev:chars.append(vocab.idx2char(p));scs.append(sc)  # 移除blank和重複
            prev=p
        return''.join(chars),float(np.mean(scs))if scs else 0.0
    def _beam_search(s,logits:np.ndarray,vocab:_Vocab)->Tuple[str,float]:
        """束搜尋解碼 - 保留前k個最佳路徑"""
        probs=s._softmax(logits);T,V=probs.shape;beams=[(tuple(),1.0)]
        for t in range(T):
            new_beams={}
            for seq,sc in beams:
                for v in range(V):
                    p=probs[t,v];nsc=sc*p
                    if v==0:ns=seq  # blank不加入序列
                    elif len(seq)==0 or seq[-1]!=v:ns=seq+(v,)  # 不重複
                    else:ns=seq
                    if ns in new_beams:new_beams[ns]=max(new_beams[ns],nsc)
                    else:new_beams[ns]=nsc
            beams=sorted(new_beams.items(),key=lambda x:-x[1])[:s.bw]
        if not beams:return'',0.0
        best_seq,best_sc=beams[0];chars=[vocab.idx2char(i)for i in best_seq]
        return''.join(chars),float(best_sc**(1.0/max(len(best_seq),1)))
    def _softmax(s,x:np.ndarray)->np.ndarray:
        """計算softmax機率"""
        ex=np.exp(x-np.max(x,axis=-1,keepdims=True));return ex/np.sum(ex,axis=-1,keepdims=True)
    def decode_batch(s,logits:np.ndarray,vocab:_Vocab)->List[Tuple[str,float]]:
        """批次解碼"""
        return[s.decode(lg,vocab)for lg in logits]
class _CTCLoss:
    """CTC損失函數"""
    @staticmethod
    def forward(logits:np.ndarray,targets:List[List[int]],input_lens:List[int],target_lens:List[int])->float:
        """計算CTC損失"""
        B=logits.shape[0];losses=[]
        for b in range(B):
            log_probs=_CTCLoss._log_softmax(logits[b,:input_lens[b],:])
            tgt=targets[b][:target_lens[b]];loss=_CTCLoss._ctc_loss_single(log_probs,tgt);losses.append(loss)
        return float(np.mean(losses))
    @staticmethod
    def _log_softmax(x:np.ndarray)->np.ndarray:
        """計算log-softmax"""
        return x-np.log(np.sum(np.exp(x-np.max(x,axis=-1,keepdims=True)),axis=-1,keepdims=True))-np.max(x,axis=-1,keepdims=True)
    @staticmethod
    def _ctc_loss_single(log_probs:np.ndarray,target:List[int])->float:
        """計算單樣本CTC損失（前後向演算法）"""
        T,V=log_probs.shape;S=2*len(target)+1
        labels=[0]+[v for t in target for v in[t,0]]  # 插入blank
        alpha=np.full((T,S),-np.inf);alpha[0,0]=log_probs[0,labels[0]]
        if S>1:alpha[0,1]=log_probs[0,labels[1]]
        # 前向遞推
        for t in range(1,T):
            for s in range(S):
                alpha[t,s]=alpha[t-1,s]
                if s>0:alpha[t,s]=np.logaddexp(alpha[t,s],alpha[t-1,s-1])
                if s>1 and labels[s]!=0 and labels[s]!=labels[s-2]:
                    alpha[t,s]=np.logaddexp(alpha[t,s],alpha[t-1,s-2])
                alpha[t,s]+=log_probs[t,labels[s]]
        if S>1:return-np.logaddexp(alpha[T-1,S-1],alpha[T-1,S-2])
        return-alpha[T-1,S-1]
