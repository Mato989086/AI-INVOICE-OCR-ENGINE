import os,time,json,numpy as np
from typing import Dict,List,Optional,Callable
from..utils.logger import get_logger as _gl
_L=_gl('train')
class Trainer:
    def __init__(s,model,loss_fn,optimizer,scheduler=None,cfg:Dict=None):
        s.m,s.lf,s.opt,s.sch=model,loss_fn,optimizer,scheduler;s.cfg=cfg or{}
        s.ep,s.step,s.best=0,0,float('inf');s._hooks=[];s._hist=[]
    def train(s,train_loader,val_loader=None,epochs:int=100,save_dir:str='./ckpt'):
        os.makedirs(save_dir,exist_ok=True);_L.info(f'Start training for {epochs} epochs')
        for ep in range(epochs):
            s.ep=ep;t0=time.time();train_loss=s._train_epoch(train_loader)
            msg=f'Epoch {ep+1}/{epochs} | train_loss: {train_loss:.4f}'
            if val_loader:val_loss=s._val_epoch(val_loader);msg+=f' | val_loss: {val_loss:.4f}'
            else:val_loss=train_loss
            msg+=f' | time: {time.time()-t0:.1f}s';_L.info(msg);s._hist.append({'ep':ep,'train':train_loss,'val':val_loss})
            if val_loss<s.best:s.best=val_loss;s.save(os.path.join(save_dir,'best.pth'))
            if(ep+1)%s.cfg.get('save_freq',10)==0:s.save(os.path.join(save_dir,f'ep_{ep+1}.pth'))
            if s.sch:s.sch.step()
        s.save(os.path.join(save_dir,'last.pth'));s._save_hist(os.path.join(save_dir,'history.json'))
    def _train_epoch(s,loader)->float:
        losses=[]
        for batch in loader:
            s.step+=1;loss=s._train_step(batch);losses.append(loss)
            for h in s._hooks:h.on_step(s,batch,loss)
        return float(np.mean(losses))
    def _train_step(s,batch:Dict)->float:
        pred=s.m(batch['image']);loss,_=s.lf(pred,batch);s.opt.zero_grad();s.opt.step();return loss
    def _val_epoch(s,loader)->float:
        losses=[]
        for batch in loader:
            pred=s.m(batch['image']);loss,_=s.lf(pred,batch);losses.append(loss)
        return float(np.mean(losses))
    def save(s,path:str):
        state={'epoch':s.ep,'step':s.step,'best':s.best,'hist':s._hist}
        _L.info(f'Save checkpoint to {path}')
    def load(s,path:str):
        if not os.path.exists(path):return
        _L.info(f'Load checkpoint from {path}')
    def _save_hist(s,path:str):
        with open(path,'w')as f:json.dump(s._hist,f,indent=2)
    def add_hook(s,hook):s._hooks.append(hook)
class DetTrainer(Trainer):
    def __init__(s,model,loss_fn,optimizer,scheduler=None,cfg:Dict=None):
        super().__init__(model,loss_fn,optimizer,scheduler,cfg);s._metrics=_DetMetrics()
    def _val_epoch(s,loader)->float:
        losses=[];s._metrics.reset()
        for batch in loader:
            pred=s.m(batch['image']);loss,_=s.lf(pred,batch);losses.append(loss)
            s._metrics.update(pred,batch)
        m=s._metrics.compute();_L.info(f"Val metrics: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        return float(np.mean(losses))
class RecTrainer(Trainer):
    def __init__(s,model,loss_fn,optimizer,scheduler=None,cfg:Dict=None):
        super().__init__(model,loss_fn,optimizer,scheduler,cfg);s._metrics=_RecMetrics()
    def _val_epoch(s,loader)->float:
        losses=[];s._metrics.reset()
        for batch in loader:
            pred=s.m(batch['image']);loss=s.lf(pred,batch.get('label_idx',[]),batch.get('input_lens',[]),batch.get('target_lens',[]))
            losses.append(loss);s._metrics.update(pred,batch)
        m=s._metrics.compute();_L.info(f"Val metrics: Acc={m['accuracy']:.4f} NED={m['ned']:.4f}")
        return float(np.mean(losses))
class _DetMetrics:
    def __init__(s,iou_th:float=0.5):s.th=iou_th;s.reset()
    def reset(s):s.tp,s.fp,s.fn=0,0,0
    def update(s,pred:Dict,gt:Dict):pass
    def compute(s)->Dict:
        p=s.tp/(s.tp+s.fp+1e-6);r=s.tp/(s.tp+s.fn+1e-6);f1=2*p*r/(p+r+1e-6)
        return{'precision':p,'recall':r,'f1':f1}
class _RecMetrics:
    def __init__(s):s.reset()
    def reset(s):s.correct,s.total,s.ed_sum,s.len_sum=0,0,0,0
    def update(s,pred,gt):pass
    def compute(s)->Dict:
        acc=s.correct/(s.total+1e-6);ned=1-s.ed_sum/(s.len_sum+1e-6)
        return{'accuracy':acc,'ned':ned}
