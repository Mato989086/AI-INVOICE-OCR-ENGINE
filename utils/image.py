import cv2,numpy as np,os,base64,io
from typing import Tuple,List,Optional,Union
from urllib.request import urlopen
def imread(src:Union[str,bytes,np.ndarray],flags:int=cv2.IMREAD_COLOR)->np.ndarray:
    if isinstance(src,np.ndarray):return src
    if isinstance(src,bytes):return cv2.imdecode(np.frombuffer(src,np.uint8),flags)
    if src.startswith(('http://','https://')):
        with urlopen(src,timeout=30)as resp:return cv2.imdecode(np.frombuffer(resp.read(),np.uint8),flags)
    if src.startswith('data:image'):
        _,data=src.split(',',1);return cv2.imdecode(np.frombuffer(base64.b64decode(data),np.uint8),flags)
    return cv2.imread(src,flags)
def imwrite(path:str,img:np.ndarray,params:List[int]=None)->bool:
    os.makedirs(os.path.dirname(path)or'.',exist_ok=True);return cv2.imwrite(path,img,params or[])
def resize(img:np.ndarray,size:Tuple[int,int]=None,scale:float=None,keep_ratio:bool=True,pad:bool=False)->Tuple[np.ndarray,Dict]:
    h,w=img.shape[:2];meta={'orig':(w,h)}
    if scale:nw,nh=int(w*scale),int(h*scale)
    elif size:
        tw,th=size
        if keep_ratio:
            sc=min(tw/w,th/h);nw,nh=int(w*sc),int(h*sc)
        else:nw,nh=tw,th
    else:return img,meta
    img=cv2.resize(img,(nw,nh));meta['new']=(nw,nh);meta['scale']=nw/w
    if pad and size:
        tw,th=size;top,left=(th-nh)//2,(tw-nw)//2
        img=cv2.copyMakeBorder(img,top,th-nh-top,left,tw-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
        meta['pad']=(left,top)
    return img,meta
def crop_poly(img:np.ndarray,poly:np.ndarray)->np.ndarray:
    poly=poly.astype(np.float32);rect=cv2.boundingRect(poly)
    x,y,w,h=rect;w,h=max(w,1),max(h,1)
    pts=_order_points(poly);tw=int(max(np.linalg.norm(pts[0]-pts[1]),np.linalg.norm(pts[2]-pts[3])))
    th=int(max(np.linalg.norm(pts[0]-pts[3]),np.linalg.norm(pts[1]-pts[2])))
    tw,th=max(tw,1),max(th,1);dst=np.array([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]],dtype=np.float32)
    M=cv2.getPerspectiveTransform(pts,dst);return cv2.warpPerspective(img,M,(tw,th))
def _order_points(pts:np.ndarray)->np.ndarray:
    rect=np.zeros((4,2),dtype=np.float32);s=pts.sum(axis=1);d=np.diff(pts,axis=1)
    rect[0]=pts[np.argmin(s)];rect[2]=pts[np.argmax(s)];rect[1]=pts[np.argmin(d)];rect[3]=pts[np.argmax(d)]
    return rect
def pad_to_multiple(img:np.ndarray,m:int=32)->Tuple[np.ndarray,Tuple[int,int]]:
    h,w=img.shape[:2];nh,nw=(h+m-1)//m*m,(w+m-1)//m*m
    if nh==h and nw==w:return img,(0,0)
    padded=np.zeros((nh,nw,3)if img.ndim==3 else(nh,nw),dtype=img.dtype)
    padded[:h,:w]=img;return padded,(nw-w,nh-h)
def rotate(img:np.ndarray,angle:float,expand:bool=True)->np.ndarray:
    h,w=img.shape[:2];cx,cy=w/2,h/2;M=cv2.getRotationMatrix2D((cx,cy),angle,1.0)
    if expand:
        cos,sin=abs(M[0,0]),abs(M[0,1]);nw,nh=int(h*sin+w*cos),int(h*cos+w*sin)
        M[0,2]+=nw/2-cx;M[1,2]+=nh/2-cy
    else:nw,nh=w,h
    return cv2.warpAffine(img,M,(nw,nh))
def to_base64(img:np.ndarray,fmt:str='png')->str:
    _,buf=cv2.imencode(f'.{fmt}',img);return base64.b64encode(buf).decode('utf-8')
def from_base64(s:str)->np.ndarray:
    if','in s:s=s.split(',',1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(s),np.uint8),cv2.IMREAD_COLOR)
from typing import Dict
