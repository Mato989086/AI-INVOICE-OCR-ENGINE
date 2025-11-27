# -*- coding: utf-8 -*-
"""詞彙表模組 - 字元到索引的映射"""
import json,os
from typing import Dict,List,Optional
class Vocabulary:
    """詞彙表 - 管理字元編碼"""
    _BLANK='<blank>';_UNK='<unk>';_DEFAULT=None
    def __init__(s,path:str=''):
        """初始化詞彙表"""
        s._c2i={};s._i2c={};s._sz=0
        if path and os.path.exists(path):s.load(path)
        else:s._init_default()
    def _init_default(s):
        """初始化預設詞彙表"""
        chars=[s._BLANK,s._UNK]+list('0123456789')+list('abcdefghijklmnopqrstuvwxyz')+list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        chars+=['。','，','、','；','：','？','！','"','"','（','）','【','】','《','》','—','…','·']
        # 常用中文字
        chars+=list('的一是不了在人有我他這個們中來上大為和國地到以說時要就出會可也你對生能而子那得於著下自之年過發後作裡用道行所然家種事成方多經麼去法學如都同現當沒動面起看定天分還進好小部其些主樣理心她本前開但因只從想實日軍者意無力它與長把機十民第公此已工使情明性知全三又關點正業外將兩高間由問很最重並物手應戰向頭文體政美相見被利什二等產或新己制身果加西斯月話合回特代內信表化老給世位次度門任常先海通教兒原東聲提立及比員解水名真論處走義各入幾口認條平系氣題活爾更別打女變四神總何電數安少報才結反受目太量再感建務做接必場件計管期市直德資命山金指克許統區保至隊形社便空決治展馬原士向戰邊')
        s._c2i={c:i for i,c in enumerate(chars)};s._i2c={i:c for c,i in s._c2i.items()};s._sz=len(chars)
    def load(s,path:str):
        """從檔案載入詞彙表"""
        with open(path,'r',encoding='utf-8')as f:
            data=json.load(f)
            if isinstance(data,dict):s._c2i=data;s._i2c={v:k for k,v in data.items()}
            elif isinstance(data,list):s._c2i={c:i for i,c in enumerate(data)};s._i2c={i:c for i,c in enumerate(data)}
            s._sz=len(s._c2i)
    def save(s,path:str):
        """儲存詞彙表到檔案"""
        os.makedirs(os.path.dirname(path)or'.',exist_ok=True)
        with open(path,'w',encoding='utf-8')as f:json.dump(s._c2i,f,ensure_ascii=False,indent=2)
    def char2idx(s,c:str)->int:
        """字元轉索引"""
        return s._c2i.get(c,s._c2i.get(s._UNK,1))
    def idx2char(s,i:int)->str:
        """索引轉字元"""
        return s._i2c.get(i,s._UNK)
    def encode(s,text:str)->List[int]:
        """編碼文字為索引列表"""
        return[s.char2idx(c)for c in text]
    def decode(s,idxs:List[int],remove_blank:bool=True)->str:
        """解碼索引列表為文字"""
        chars=[];prev=-1
        for i in idxs:
            if i==0 and remove_blank:prev=i;continue  # 跳過blank
            if i!=prev:chars.append(s.idx2char(i))  # 跳過重複
            prev=i
        return''.join(chars)
    @property
    def size(s)->int:
        """詞彙表大小"""
        return s._sz
    @property
    def blank_idx(s)->int:
        """blank索引"""
        return s._c2i.get(s._BLANK,0)
    def __len__(s)->int:return s._sz
    def __contains__(s,c:str)->bool:return c in s._c2i
    @classmethod
    def default(c)->'Vocabulary':
        """取得預設詞彙表（單例）"""
        if c._DEFAULT is None:c._DEFAULT=c()
        return c._DEFAULT
    def add(s,c:str)->int:
        """新增字元"""
        if c in s._c2i:return s._c2i[c]
        idx=s._sz;s._c2i[c]=idx;s._i2c[idx]=c;s._sz+=1;return idx
    def merge(s,other:'Vocabulary'):
        """合併另一個詞彙表"""
        for c in other._c2i:
            if c not in s._c2i:s.add(c)
