import cv2,numpy as np
from typing import Dict,List,Tuple,Optional,Union
import random as _R
class Transform:
    def __init__(s,ops:List=None):s.ops=ops or[]
    def __call__(s,item:Dict)->Dict:
        for op in s.ops:item=op(item)
        return item
    def add(s,op)->'Transform':s.ops.append(op);return s
class _Resize:
    def __init__(s,size:Union[int,Tuple[int,int]],keep_ratio:bool=True):
        s.sz=size if isinstance(size,tuple)else(size,size);s.kr=keep_ratio
    def __call__(s,item:Dict)->Dict:
        img=item['image'];h,w=img.shape[:2];tw,th=s.sz
        if s.kr:sc=min(tw/w,th/h);nw,nh=int(w*sc),int(h*sc)
        else:nw,nh=tw,th
        item['image']=cv2.resize(img,(nw,nh));item['scale']=nw/w
        if'polys'in item:item['polys']=[p*(nw/w)for p in item['polys']]
        return item
class _Normalize:
    def __init__(s,mean:List[float]=[0.485,0.456,0.406],std:List[float]=[0.229,0.224,0.225]):
        s.m,s.s=np.array(mean).reshape(1,1,3),np.array(std).reshape(1,1,3)
    def __call__(s,item:Dict)->Dict:
        img=item['image'].astype(np.float32)/255.0;item['image']=(img-s.m)/s.s;return item
class _RandomCrop:
    def __init__(s,size:Tuple[int,int],min_iou:float=0.5):s.sz,s.mi=size,min_iou
    def __call__(s,item:Dict)->Dict:
        img=item['image'];h,w=img.shape[:2];th,tw=s.sz
        if h<=th and w<=tw:return item
        for _ in range(50):
            y,x=_R.randint(0,h-th),_R.randint(0,w-tw)
            if'polys'not in item:break
            valid=True
            for p in item['polys']:
                if not s._check_iou(p,x,y,tw,th,s.mi):valid=False;break
            if valid:break
        item['image']=img[y:y+th,x:x+tw]
        if'polys'in item:item['polys']=[p-np.array([x,y])for p in item['polys']]
        return item
    def _check_iou(s,poly:np.ndarray,x:int,y:int,w:int,h:int,th:float)->bool:
        px1,py1=poly.min(axis=0);px2,py2=poly.max(axis=0)
        ix1,iy1=max(px1,x),max(py1,y);ix2,iy2=min(px2,x+w),min(py2,y+h)
        if ix1>=ix2 or iy1>=iy2:return False
        inter=(ix2-ix1)*(iy2-iy1);area=(px2-px1)*(py2-py1)
        return inter/area>=th if area>0 else False
class _RandomRotate:
    def __init__(s,max_angle:float=10):s.ma=max_angle
    def __call__(s,item:Dict)->Dict:
        img=item['image'];h,w=img.shape[:2];a=_R.uniform(-s.ma,s.ma)
        M=cv2.getRotationMatrix2D((w/2,h/2),a,1);item['image']=cv2.warpAffine(img,M,(w,h))
        if'polys'in item:
            polys=[]
            for p in item['polys']:
                p=np.hstack([p,np.ones((len(p),1))]);p=(M@p.T).T;polys.append(p)
            item['polys']=polys
        return item
class _ColorJitter:
    def __init__(s,brightness:float=0.3,contrast:float=0.3,saturation:float=0.3):s.b,s.c,s.sat=brightness,contrast,saturation
    def __call__(s,item:Dict)->Dict:
        img=item['image'].astype(np.float32)
        img=img+_R.uniform(-s.b,s.b)*255;img=np.clip(img,0,255)
        m=img.mean();img=(img-m)*_R.uniform(1-s.c,1+s.c)+m;img=np.clip(img,0,255)
        item['image']=img.astype(np.uint8);return item
class DetTransform(Transform):
    def __init__(s,size:Tuple[int,int]=(640,640),augment:bool=True):
        ops=[_Resize(size)];
        if augment:ops+=[_RandomRotate(10),_ColorJitter()]
        ops.append(_Normalize());super().__init__(ops)
class RecTransform(Transform):
    def __init__(s,height:int=48,max_width:int=320,augment:bool=True):
        s.h,s.mw=height,max_width;ops=[]
        if augment:ops.append(_ColorJitter(0.2,0.2,0.2))
        ops.append(_RecResize(height,max_width));ops.append(_Normalize([0.5]*3,[0.5]*3));super().__init__(ops)
class _RecResize:
    def __init__(s,h:int,mw:int):s.h,s.mw=h,mw
    def __call__(s,item:Dict)->Dict:
        img=item['image'];oh,ow=img.shape[:2];nw=min(int(ow*s.h/oh),s.mw)
        item['image']=cv2.resize(img,(nw,s.h));return item
