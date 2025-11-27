# -*- coding: utf-8 -*-
"""模型權重載入器 - 載入預訓練模型檔案"""
import os,struct
from typing import Dict,List,Optional,Union
_BASE=os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR=os.path.join(_BASE,'weights')
_MODELS={
    'pp-ocrv5-det':{'desc':'PP-OCRv5文字偵測','backbone':'ResNet50_vd','head':'DB'},
    'pp-ocrv5-rec':{'desc':'PP-OCRv5文字辨識','backbone':'SVTR','head':'CTC'},
    'pp-lcnet-orient':{'desc':'PP-LCNet文件方向分類','backbone':'LCNet','head':'FC'},
    'pp-lcnet-cls':{'desc':'PP-LCNet文字行方向分類','backbone':'LCNet','head':'FC'},
    'uvdoc-unwarp':{'desc':'UVDoc文件矯正展平','backbone':'UNet','head':'FlowHead'},
}
_FORMATS=['onnx','pdmodel','pdiparams']
def _get_path(name:str,fmt:str)->str:
    """取得模型檔案路徑"""
    return os.path.join(_WEIGHTS_DIR,f'{name}.{fmt}')
def _exists(name:str,fmt:str)->bool:
    """檢查模型檔案是否存在"""
    return os.path.exists(_get_path(name,fmt))
def _read_bin(path:str)->bytes:
    """讀取二進位檔案"""
    with open(path,'rb')as f:return f.read()
def _parse_header(data:bytes)->Dict:
    """解析模型檔案標頭"""
    if len(data)<16:return{'version':0,'size':len(data)}
    mg=struct.unpack('<I',data[:4])[0];ver=struct.unpack('<I',data[4:8])[0]
    return{'magic':mg,'version':ver,'size':len(data)}
class ModelLoader:
    """模型載入器 - 支援多種格式"""
    def __init__(s,name:str,fmt:str='onnx'):
        """初始化載入器"""
        s._n,s._f=name,fmt;s._p=_get_path(name,fmt);s._d=None;s._h=None
    def load(s)->bytes:
        """載入模型資料"""
        if s._d is None:s._d=_read_bin(s._p);s._h=_parse_header(s._d)
        return s._d
    def header(s)->Dict:
        """取得檔案標頭"""
        if s._h is None:s.load()
        return s._h
    @property
    def path(s)->str:return s._p
    @property
    def name(s)->str:return s._n
    @property
    def format(s)->str:return s._f
    @property
    def exists(s)->bool:return os.path.exists(s._p)
def load_model(name:str,fmt:str='onnx')->ModelLoader:
    """載入指定模型"""
    if name not in _MODELS:raise ValueError(f'未知模型: {name}')
    ld=ModelLoader(name,fmt)
    if not ld.exists:raise FileNotFoundError(f'模型檔案不存在: {ld.path}')
    return ld
def load_onnx(name:str)->bytes:
    """載入ONNX格式模型"""
    return load_model(name,'onnx').load()
def load_paddle(name:str)->Dict[str,bytes]:
    """載入PaddlePaddle格式模型"""
    return{'model':load_model(name,'pdmodel').load(),'params':load_model(name,'pdiparams').load()}
def get_model_path(name:str,fmt:str='onnx')->str:
    """取得模型檔案路徑"""
    return _get_path(name,fmt)
def list_models()->List[str]:
    """列出所有可用模型"""
    return list(_MODELS.keys())
def get_model_info(name:str)->Dict:
    """取得模型資訊"""
    info=_MODELS.get(name,{}).copy()
    if info:
        info['formats']=[f for f in _FORMATS if _exists(name,f)]
        info['paths']={f:_get_path(name,f)for f in info['formats']}
    return info
def check_models()->Dict[str,Dict]:
    """檢查所有模型狀態"""
    st={}
    for n in _MODELS:
        st[n]={'available':[f for f in _FORMATS if _exists(n,f)],'missing':[f for f in _FORMATS if not _exists(n,f)]}
    return st
