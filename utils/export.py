import json,os
from typing import List,Dict,Any,Optional
from datetime import datetime
def to_json(results:List[Dict],path:str=None,indent:int=2)->str:
    def _serialize(obj):
        if hasattr(obj,'tolist'):return obj.tolist()
        if hasattr(obj,'__dict__'):return obj.__dict__
        return str(obj)
    out={'version':'2.1.0','timestamp':datetime.now().isoformat(),'results':[]}
    for r in results:
        item={'path':r.get('path'),'size':r.get('size'),'boxes':[b.tolist()if hasattr(b,'tolist')else b for b in r.get('boxes',[])],'texts':r.get('texts',[]),'scores':r.get('scores',[])}
        out['results'].append(item)
    s=json.dumps(out,indent=indent,ensure_ascii=False,default=_serialize)
    if path:os.makedirs(os.path.dirname(path)or'.',exist_ok=True);open(path,'w',encoding='utf-8').write(s)
    return s
def to_excel(results:List[Dict],path:str)->str:
    try:import pandas as pd
    except:raise ImportError('pandas required')
    rows=[]
    for r in results:
        fp=r.get('path','');txts,scs=r.get('texts',[]),r.get('scores',[])
        for i,txt in enumerate(txts):
            sc=scs[i]if i<len(scs)else 0
            rows.append({'file':fp,'index':i,'text':txt,'score':sc})
    df=pd.DataFrame(rows);os.makedirs(os.path.dirname(path)or'.',exist_ok=True);df.to_excel(path,index=False);return path
def to_html(results:List[Dict],path:str,img_dir:str=None)->str:
    html=['<!DOCTYPE html><html><head><meta charset="utf-8"><title>OCR Results</title>',
    '<style>body{font-family:Arial;margin:20px}table{border-collapse:collapse;width:100%}',
    'th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#4CAF50;color:white}',
    'tr:nth-child(even){background:#f2f2f2}.score{color:#666;font-size:0.9em}</style></head><body>',
    f'<h1>OCR Results</h1><p>Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>']
    for i,r in enumerate(results):
        fp=r.get('path','');html.append(f'<h2>Document {i+1}: {os.path.basename(fp)}</h2>')
        if img_dir and fp:
            rp=os.path.relpath(fp,os.path.dirname(path))if os.path.exists(fp)else fp
            html.append(f'<img src="{rp}" style="max-width:600px;margin:10px 0">')
        html.append('<table><tr><th>#</th><th>Text</th><th>Confidence</th></tr>')
        for j,(txt,sc)in enumerate(zip(r.get('texts',[]),r.get('scores',[]))):
            html.append(f'<tr><td>{j+1}</td><td>{txt}</td><td class="score">{sc:.4f}</td></tr>')
        html.append('</table>')
    html.append('</body></html>');s='\n'.join(html)
    os.makedirs(os.path.dirname(path)or'.',exist_ok=True);open(path,'w',encoding='utf-8').write(s);return path
def to_csv(results:List[Dict],path:str,delimiter:str=',')->'str':
    lines=[f'file{delimiter}index{delimiter}text{delimiter}score']
    for r in results:
        fp=r.get('path','').replace(delimiter,'_');txts,scs=r.get('texts',[]),r.get('scores',[])
        for i,txt in enumerate(txts):
            sc=scs[i]if i<len(scs)else 0;txt=txt.replace(delimiter,' ').replace('\n',' ')
            lines.append(f'{fp}{delimiter}{i}{delimiter}{txt}{delimiter}{sc:.4f}')
    s='\n'.join(lines);os.makedirs(os.path.dirname(path)or'.',exist_ok=True);open(path,'w',encoding='utf-8').write(s);return path
