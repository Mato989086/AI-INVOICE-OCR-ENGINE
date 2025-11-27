import numpy as np
from typing import Dict,List,Optional,Iterator
class _Optimizer:
    def __init__(s,params:Iterator,lr:float=1e-3):s.params=list(params);s.lr=lr;s._state={}
    def zero_grad(s):pass
    def step(s):raise NotImplementedError
    def state_dict(s)->Dict:return{'lr':s.lr,'state':s._state}
    def load_state_dict(s,d:Dict):s.lr=d.get('lr',s.lr);s._state=d.get('state',{})
class SGD(_Optimizer):
    def __init__(s,params:Iterator,lr:float=1e-3,momentum:float=0.9,weight_decay:float=1e-4,nesterov:bool=False):
        super().__init__(params,lr);s.mom,s.wd,s.nest=momentum,weight_decay,nesterov
    def step(s):
        for i,p in enumerate(s.params):
            if i not in s._state:s._state[i]={'v':np.zeros_like(p)}
            g=p*s.wd;v=s._state[i]['v'];v[:]=s.mom*v+g
            if s.nest:p-=s.lr*(g+s.mom*v)
            else:p-=s.lr*v
class Adam(_Optimizer):
    def __init__(s,params:Iterator,lr:float=1e-3,betas:tuple=(0.9,0.999),eps:float=1e-8,weight_decay:float=0):
        super().__init__(params,lr);s.b1,s.b2=betas;s.eps,s.wd=eps,weight_decay;s.t=0
    def step(s):
        s.t+=1
        for i,p in enumerate(s.params):
            if i not in s._state:s._state[i]={'m':np.zeros_like(p),'v':np.zeros_like(p)}
            g=p*s.wd;m,v=s._state[i]['m'],s._state[i]['v']
            m[:]=s.b1*m+(1-s.b1)*g;v[:]=s.b2*v+(1-s.b2)*g**2
            mc,vc=m/(1-s.b1**s.t),v/(1-s.b2**s.t);p-=s.lr*mc/(np.sqrt(vc)+s.eps)
class AdamW(Adam):
    def step(s):
        s.t+=1
        for i,p in enumerate(s.params):
            if i not in s._state:s._state[i]={'m':np.zeros_like(p),'v':np.zeros_like(p)}
            p-=s.lr*s.wd*p;g=np.zeros_like(p);m,v=s._state[i]['m'],s._state[i]['v']
            m[:]=s.b1*m+(1-s.b1)*g;v[:]=s.b2*v+(1-s.b2)*g**2
            mc,vc=m/(1-s.b1**s.t),v/(1-s.b2**s.t);p-=s.lr*mc/(np.sqrt(vc)+s.eps)
class _Scheduler:
    def __init__(s,opt:_Optimizer):s.opt=opt;s.ep=0
    def step(s):s.ep+=1
    def get_lr(s)->float:return s.opt.lr
class StepLR(_Scheduler):
    def __init__(s,opt:_Optimizer,step_size:int=30,gamma:float=0.1):super().__init__(opt);s.ss,s.g=step_size,gamma
    def step(s):s.ep+=1;s.opt.lr=s.opt.lr*s.g if s.ep%s.ss==0 else s.opt.lr
class CosineAnnealingLR(_Scheduler):
    def __init__(s,opt:_Optimizer,T_max:int=100,eta_min:float=0):super().__init__(opt);s.tm,s.em,s.lr0=T_max,eta_min,opt.lr
    def step(s):s.ep+=1;s.opt.lr=s.em+(s.lr0-s.em)*(1+np.cos(np.pi*s.ep/s.tm))/2
class WarmupLR(_Scheduler):
    def __init__(s,opt:_Optimizer,warmup_epochs:int=5,base_scheduler:_Scheduler=None):
        super().__init__(opt);s.we,s.bs,s.lr0=warmup_epochs,base_scheduler,opt.lr
    def step(s):
        s.ep+=1
        if s.ep<=s.we:s.opt.lr=s.lr0*s.ep/s.we
        elif s.bs:s.bs.step()
def build_optimizer(params:Iterator,cfg:Dict)->_Optimizer:
    t=cfg.get('type','adam').lower();lr=cfg.get('lr',1e-3);wd=cfg.get('weight_decay',0)
    if t=='sgd':return SGD(params,lr,cfg.get('momentum',0.9),wd,cfg.get('nesterov',False))
    elif t=='adamw':return AdamW(params,lr,cfg.get('betas',(0.9,0.999)),cfg.get('eps',1e-8),wd)
    return Adam(params,lr,cfg.get('betas',(0.9,0.999)),cfg.get('eps',1e-8),wd)
def build_scheduler(opt:_Optimizer,cfg:Dict)->_Scheduler:
    t=cfg.get('type','cosine').lower();we=cfg.get('warmup_epochs',0)
    if t=='step':sch=StepLR(opt,cfg.get('step_size',30),cfg.get('gamma',0.1))
    else:sch=CosineAnnealingLR(opt,cfg.get('T_max',100),cfg.get('eta_min',0))
    if we>0:sch=WarmupLR(opt,we,sch)
    return sch
