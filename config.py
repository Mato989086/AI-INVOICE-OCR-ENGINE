# -*- coding: utf-8 -*-
"""設定管理模組 - 處理所有OCR相關設定參數"""
import os,json
from dataclasses import dataclass,field
from typing import List,Dict,Any,Optional,Tuple,Union
from enum import IntEnum,auto
class _M(IntEnum):
    """運算模式列舉"""
    CPU=0;GPU=auto();TPU=auto();NPU=auto()
class _P(IntEnum):
    """精度等級列舉"""
    LOW=0;MED=auto();HIGH=auto();ULTRA=auto()
@dataclass
class _DetCfg:
    """文字偵測設定"""
    algo:str='DB'  # 偵測演算法
    bb:str='ResNet50_vd'  # 主幹網路
    th:float=0.3  # 機率閾值
    bth:float=0.6  # 框閾值
    ur:float=1.5  # 擴展比例
    mxs:int=960  # 最大邊長
    ms:int=3  # 最小尺寸
    ub:bool=True  # 使用擴展
    dil:int=1  # 膨脹核心
    sc:bool=False  # 縮放模式
@dataclass
class _RecCfg:
    """文字辨識設定"""
    algo:str='SVTR'  # 辨識演算法
    bb:str='PPLCNetV3'  # 主幹網路
    mw:int=320  # 最大寬度
    h:int=48  # 固定高度
    bs:int=6  # 批次大小
    vp:str=''  # 詞彙表路徑
    cs:bool=True  # 區分大小寫
    bm:str='greedy'  # 解碼模式
    bw:int=5  # 束搜尋寬度
@dataclass
class _ClsCfg:
    """方向分類設定"""
    en:bool=True  # 啟用分類
    th:float=0.9  # 信心閾值
    bs:int=6  # 批次大小
    lb:List[str]=field(default_factory=lambda:['0','180'])  # 類別標籤
@dataclass
class _PrepCfg:
    """前處理設定"""
    ori:bool=True  # 方向校正
    uwp:bool=False  # 扭曲矯正
    oth:float=0.9  # 方向閾值
    uwth:float=0.5  # 矯正閾值
    rs:Tuple[int,int]=(960,960)  # 調整尺寸
    nm:List[float]=field(default_factory=lambda:[.485,.456,.406])  # 正規化均值
    ns:List[float]=field(default_factory=lambda:[.229,.224,.225])  # 正規化標準差
@dataclass
class Config:
    """主設定類別 - 整合所有子設定"""
    det:_DetCfg=field(default_factory=_DetCfg)
    rec:_RecCfg=field(default_factory=_RecCfg)
    cls:_ClsCfg=field(default_factory=_ClsCfg)
    prep:_PrepCfg=field(default_factory=_PrepCfg)
    mode:_M=_M.CPU  # 運算模式
    prec:_P=_P.HIGH  # 精度等級
    lang:str='ch'  # 語言
    dbg:bool=False  # 除錯模式
    nth:int=4  # 執行緒數
    mp:str=''  # 模型路徑
    cp:str=''  # 快取路徑
    lp:str='./logs'  # 日誌路徑
    def __post_init__(s):
        """初始化後處理"""
        s._v={};s._h=hash(json.dumps(s.to_dict(),sort_keys=1))
    def to_dict(s)->Dict[str,Any]:
        """轉換為字典格式"""
        return{'det':{'algo':s.det.algo,'backbone':s.det.bb,'thresh':s.det.th,'box_thresh':s.det.bth,
        'unclip':s.det.ur,'max_side':s.det.mxs,'min_size':s.det.ms},'rec':{'algo':s.rec.algo,
        'backbone':s.rec.bb,'max_width':s.rec.mw,'height':s.rec.h},'cls':{'enabled':s.cls.en,
        'thresh':s.cls.th},'prep':{'orientation':s.prep.ori,'unwarp':s.prep.uwp},'mode':s.mode.name}
    @classmethod
    def from_dict(c,d:Dict)->Config:
        """從字典建立設定"""
        cfg=c();[setattr(cfg.det,k,v)for k,v in d.get('det',{}).items()if hasattr(cfg.det,k)]
        [setattr(cfg.rec,k,v)for k,v in d.get('rec',{}).items()if hasattr(cfg.rec,k)];return cfg
    @classmethod
    def load(c,p:str)->'Config':
        """從檔案載入設定"""
        with open(p,'r',encoding='utf-8')as f:return c.from_dict(json.load(f))
    def save(s,p:str):
        """儲存設定到檔案"""
        os.makedirs(os.path.dirname(p)or'.',exist_ok=1)
        with open(p,'w',encoding='utf-8')as f:json.dump(s.to_dict(),f,indent=2,ensure_ascii=0)
    def validate(s)->Tuple[bool,List[str]]:
        """驗證設定有效性"""
        e=[];s.det.th<0 or s.det.th>1 or e.append(f'det.th:{s.det.th}')
        s.rec.h not in[32,48,64]and e.append(f'rec.h:{s.rec.h}');return len(e)==0,e
    def __hash__(s):return s._h
    def clone(s)->'Config':
        """複製設定物件"""
        import copy;return copy.deepcopy(s)
