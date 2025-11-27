import logging,sys,os
from typing import Optional
from datetime import datetime
_FMT='%(asctime)s|%(levelname)s|%(name)s|%(message)s'
_DFMT='%Y-%m-%d %H:%M:%S'
_LOGGERS={}
def get_logger(name:str=None,level:int=logging.INFO)->logging.Logger:
    name=name or'ocr';
    if name in _LOGGERS:return _LOGGERS[name]
    lg=logging.getLogger(name);lg.setLevel(level);lg.propagate=False
    if not lg.handlers:
        ch=logging.StreamHandler(sys.stdout);ch.setLevel(level)
        ch.setFormatter(logging.Formatter(_FMT,_DFMT));lg.addHandler(ch)
    _LOGGERS[name]=lg;return lg
def setup_logger(name:str,log_dir:str=None,level:int=logging.INFO,console:bool=True,file:bool=True)->logging.Logger:
    lg=logging.getLogger(name);lg.setLevel(level);lg.handlers.clear()
    fmt=logging.Formatter(_FMT,_DFMT)
    if console:ch=logging.StreamHandler(sys.stdout);ch.setLevel(level);ch.setFormatter(fmt);lg.addHandler(ch)
    if file and log_dir:
        os.makedirs(log_dir,exist_ok=True);fn=f'{name}_{datetime.now():%Y%m%d_%H%M%S}.log'
        fh=logging.FileHandler(os.path.join(log_dir,fn),encoding='utf-8');fh.setLevel(level);fh.setFormatter(fmt);lg.addHandler(fh)
    _LOGGERS[name]=lg;return lg
class _Timer:
    def __init__(s,name:str='',logger:logging.Logger=None):s.name,s.lg=name,logger or get_logger()
    def __enter__(s):import time;s._t0=time.perf_counter();return s
    def __exit__(s,*a):import time;s.elapsed=time.perf_counter()-s._t0;s.lg.debug(f'{s.name}: {s.elapsed*1000:.2f}ms')
class _Progress:
    def __init__(s,total:int,name:str='',logger:logging.Logger=None):s.total,s.name,s.lg,s.n=total,name,logger or get_logger(),0
    def update(s,n:int=1):s.n+=n;pct=s.n*100/s.total;s.lg.info(f'{s.name}: {s.n}/{s.total} ({pct:.1f}%)')
    def __enter__(s):return s
    def __exit__(s,*a):pass
