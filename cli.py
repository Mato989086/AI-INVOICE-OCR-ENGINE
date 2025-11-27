import argparse,sys,os,json
from typing import List,Dict,Optional
def _parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(prog='dococr',description='Document OCR CLI')
    sp=p.add_subparsers(dest='cmd',help='Commands')
    pred=sp.add_parser('predict',help='Run OCR prediction')
    pred.add_argument('input',help='Input image or directory')
    pred.add_argument('-o','--output',default='./output',help='Output directory')
    pred.add_argument('-f','--format',choices=['json','txt','xlsx','html'],default='json')
    pred.add_argument('--vis',action='store_true',help='Save visualization')
    pred.add_argument('--gpu',action='store_true',help='Use GPU')
    train=sp.add_parser('train',help='Train model')
    train.add_argument('-c','--config',required=True,help='Config file')
    train.add_argument('-r','--resume',help='Resume from checkpoint')
    train.add_argument('--det',action='store_true',help='Train detection')
    train.add_argument('--rec',action='store_true',help='Train recognition')
    evl=sp.add_parser('eval',help='Evaluate model')
    evl.add_argument('-c','--config',required=True)
    evl.add_argument('-m','--model',required=True)
    evl.add_argument('--det',action='store_true')
    evl.add_argument('--rec',action='store_true')
    exp=sp.add_parser('export',help='Export model')
    exp.add_argument('-m','--model',required=True)
    exp.add_argument('-o','--output',required=True)
    exp.add_argument('--format',choices=['onnx','paddle','torchscript'],default='onnx')
    return p.parse_args()
def _cmd_predict(args):
    from.engine import OCREngine;from.config import Config;from.utils.image import imread,imwrite
    from.utils.visualize import draw_ocr_result;from.utils.export import to_json,to_excel,to_html
    cfg=Config();cfg.mode=1 if args.gpu else 0;eng=OCREngine(cfg)
    inp=args.input;files=[inp]if os.path.isfile(inp)else[os.path.join(inp,f)for f in os.listdir(inp)if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    results=eng.predict(files);os.makedirs(args.output,exist_ok=True)
    if args.format=='json':to_json(results,os.path.join(args.output,'result.json'))
    elif args.format=='xlsx':to_excel(results,os.path.join(args.output,'result.xlsx'))
    elif args.format=='html':to_html(results,os.path.join(args.output,'result.html'))
    else:
        with open(os.path.join(args.output,'result.txt'),'w',encoding='utf-8')as f:
            for r in results:f.write(f"=== {r.get('path','')} ===\n");[f.write(f"{t}\n")for t in r.get('texts',[])]
    if args.vis:
        for r in results:
            if r.get('path'):
                img=imread(r['path']);vis=draw_ocr_result(img,r.get('boxes',[]),r.get('texts',[]),r.get('scores'))
                fn=os.path.splitext(os.path.basename(r['path']))[0];imwrite(os.path.join(args.output,f'{fn}_vis.jpg'),vis)
    print(f'Results saved to {args.output}')
def _cmd_train(args):
    print(f'Training with config: {args.config}')
    if args.resume:print(f'Resume from: {args.resume}')
def _cmd_eval(args):
    print(f'Evaluating model: {args.model}')
def _cmd_export(args):
    print(f'Exporting {args.model} to {args.output} ({args.format})')
def main():
    args=_parse_args()
    if args.cmd=='predict':_cmd_predict(args)
    elif args.cmd=='train':_cmd_train(args)
    elif args.cmd=='eval':_cmd_eval(args)
    elif args.cmd=='export':_cmd_export(args)
    else:print('Use -h for help')
if __name__=='__main__':main()
