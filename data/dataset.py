import os,json,cv2,numpy as np
from typing import List,Dict,Tuple,Optional,Callable
class Dataset:
    def __init__(s,root:str,ann_file:str=None,transform:Callable=None):
        s.root,s.tf=root,transform;s._data=[];s._load(ann_file)
    def _load(s,ann_file:str):
        if ann_file and os.path.exists(ann_file):
            with open(ann_file,'r',encoding='utf-8')as f:
                for line in f:
                    parts=line.strip().split('\t')
                    if len(parts)>=2:s._data.append({'path':parts[0],'label':parts[1]})
        elif os.path.isdir(s.root):
            for fn in os.listdir(s.root):
                if fn.lower().endswith(('.jpg','.png','.jpeg','.bmp')):s._data.append({'path':os.path.join(s.root,fn),'label':''})
    def __len__(s)->int:return len(s._data)
    def __getitem__(s,idx:int)->Dict:
        item=s._data[idx].copy();fp=item['path']
        if not os.path.isabs(fp):fp=os.path.join(s.root,fp)
        img=cv2.imread(fp);item['image']=img
        if s.tf:item=s.tf(item)
        return item
class DetDataset(Dataset):
    def _load(s,ann_file:str):
        if not ann_file or not os.path.exists(ann_file):return
        with open(ann_file,'r',encoding='utf-8')as f:
            for line in f:
                parts=line.strip().split('\t')
                if len(parts)<2:continue
                try:
                    ann=json.loads(parts[1]);polys=[np.array(a['points'],dtype=np.float32)for a in ann]
                    txts=[a.get('transcription','')for a in ann];ign=[a.get('illegibility',False)for a in ann]
                    s._data.append({'path':parts[0],'polys':polys,'texts':txts,'ignore':ign})
                except:continue
    def __getitem__(s,idx:int)->Dict:
        item=s._data[idx].copy();fp=item['path']
        if not os.path.isabs(fp):fp=os.path.join(s.root,fp)
        img=cv2.imread(fp);item['image']=img
        if s.tf:item=s.tf(item)
        return item
class RecDataset(Dataset):
    def __init__(s,root:str,ann_file:str=None,vocab:str=None,transform:Callable=None,max_len:int=25):
        s.root,s.tf,s.ml=root,transform,max_len;s._data=[];s._vocab={};s._load(ann_file)
        if vocab:s._load_vocab(vocab)
    def _load_vocab(s,path:str):
        with open(path,'r',encoding='utf-8')as f:
            data=json.load(f)
            if isinstance(data,dict):s._vocab=data
            elif isinstance(data,list):s._vocab={c:i for i,c in enumerate(data)}
    def _encode(s,txt:str)->List[int]:
        return[s._vocab.get(c,s._vocab.get('<unk>',1))for c in txt[:s.ml]]
    def __getitem__(s,idx:int)->Dict:
        item=s._data[idx].copy();fp=item['path']
        if not os.path.isabs(fp):fp=os.path.join(s.root,fp)
        img=cv2.imread(fp);item['image']=img;item['label_idx']=s._encode(item.get('label',''))
        if s.tf:item=s.tf(item)
        return item
class _Sampler:
    def __init__(s,ds:Dataset,shuffle:bool=True,seed:int=42):s.ds,s.sh,s.sd=ds,shuffle,seed;s._idx=list(range(len(ds)))
    def __iter__(s):
        if s.sh:np.random.seed(s.sd);np.random.shuffle(s._idx)
        return iter(s._idx)
    def __len__(s)->int:return len(s._idx)
class _DistSampler(_Sampler):
    def __init__(s,ds:Dataset,n_rep:int=1,rank:int=0,shuffle:bool=True):
        super().__init__(ds,shuffle);s.nr,s.rk=n_rep,rank
        s._total=len(ds)//n_rep*n_rep;s._local=s._total//n_rep
    def __iter__(s):
        idx=list(super().__iter__())[:s._total]
        return iter(idx[s.rk::s.nr])
    def __len__(s)->int:return s._local
