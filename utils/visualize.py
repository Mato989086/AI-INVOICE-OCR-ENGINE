import cv2,numpy as np
from typing import List,Dict,Tuple,Optional,Union
from PIL import Image,ImageDraw,ImageFont
import os
_COLORS=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0)]
def draw_boxes(img:np.ndarray,boxes:List[np.ndarray],color:Tuple[int,int,int]=(0,255,0),thickness:int=2)->np.ndarray:
    img=img.copy()
    for i,box in enumerate(boxes):
        pts=box.astype(np.int32).reshape((-1,1,2));c=_COLORS[i%len(_COLORS)]if color is None else color
        cv2.polylines(img,[pts],True,c,thickness)
    return img
def draw_ocr_result(img:np.ndarray,boxes:List[np.ndarray],texts:List[str],scores:List[float]=None,font_path:str='',font_size:int=18)->np.ndarray:
    img_pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(img_pil)
    try:font=ImageFont.truetype(font_path,font_size)if font_path and os.path.exists(font_path)else ImageFont.load_default()
    except:font=ImageFont.load_default()
    for i,(box,txt)in enumerate(zip(boxes,texts)):
        pts=[(int(p[0]),int(p[1]))for p in box];c=_COLORS[i%len(_COLORS)]
        draw.polygon(pts,outline=c);x,y=pts[0]
        sc=f' ({scores[i]:.2f})'if scores and i<len(scores)else''
        draw.text((x,y-font_size-2),f'{txt}{sc}',fill=c,font=font)
    return cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
def draw_mask(img:np.ndarray,mask:np.ndarray,alpha:float=0.5,color:Tuple[int,int,int]=(0,255,0))->np.ndarray:
    overlay=img.copy();mask_bool=mask>0
    overlay[mask_bool]=np.array(color,dtype=np.uint8);return cv2.addWeighted(img,1-alpha,overlay,alpha,0)
def draw_heatmap(img:np.ndarray,heatmap:np.ndarray,alpha:float=0.6)->np.ndarray:
    hm=(heatmap*255).astype(np.uint8);hm=cv2.applyColorMap(hm,cv2.COLORMAP_JET)
    if img.shape[:2]!=hm.shape[:2]:hm=cv2.resize(hm,img.shape[1::-1])
    return cv2.addWeighted(img,1-alpha,hm,alpha,0)
def create_grid(imgs:List[np.ndarray],cols:int=4,size:Tuple[int,int]=(200,200),padding:int=5)->np.ndarray:
    n=len(imgs);rows=(n+cols-1)//cols;w,h=size
    grid=np.ones((rows*(h+padding)+padding,cols*(w+padding)+padding,3),dtype=np.uint8)*255
    for i,im in enumerate(imgs):
        r,c=i//cols,i%cols;im=cv2.resize(im,(w,h));y,x=r*(h+padding)+padding,c*(w+padding)+padding
        grid[y:y+h,x:x+w]=im
    return grid
class _Annotator:
    def __init__(s,img:np.ndarray):s.img=img.copy();s.h,s.w=img.shape[:2]
    def box(s,pts:np.ndarray,color:Tuple=(0,255,0),thickness:int=2)->'_Annotator':
        cv2.polylines(s.img,[pts.astype(np.int32).reshape(-1,1,2)],True,color,thickness);return s
    def text(s,pos:Tuple[int,int],txt:str,color:Tuple=(255,255,255),scale:float=0.6,thickness:int=1)->'_Annotator':
        cv2.putText(s.img,txt,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness);return s
    def line(s,p1:Tuple,p2:Tuple,color:Tuple=(0,255,0),thickness:int=2)->'_Annotator':
        cv2.line(s.img,p1,p2,color,thickness);return s
    def circle(s,center:Tuple,radius:int,color:Tuple=(0,0,255),thickness:int=-1)->'_Annotator':
        cv2.circle(s.img,center,radius,color,thickness);return s
    def result(s)->np.ndarray:return s.img
