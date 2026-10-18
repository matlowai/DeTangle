"""Portable entry point. Viewing and importing measurements need only Python."""
from pathlib import Path
import argparse,csv,json,math
from .dataset import read,write_json,validate,digest,local_file

def import_csv(source,out,title,baseline,protocol_path=None):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if source.parent!=out.parent:raise ValueError('Place the output dataset beside the CSV so relative image paths retain their meaning')
    conditions={};cases={};metrics={};records=[]
    protocol=json.loads(Path(protocol_path).read_text()) if protocol_path else {'id':'unspecified','label':'Unspecified protocol','matchable':False,'settings':{}}
    with source.open(newline='') as f:
        for i,row in enumerate(csv.DictReader(f)):
            cid=row['condition'];case=row['case'];axis=row['axis'];prompt=row['prompt'];conditions.setdefault(cid,{'id':cid,'label':row.get('condition_label') or cid,'description':'User-supplied condition','settings':{}})
            c=cases.setdefault(case,{'id':case,'label':row.get('case_label') or case,'prompts':{}})
            if axis in c['prompts'] and c['prompts'][axis]!=prompt:raise ValueError('One case/axis has inconsistent prompts')
            c['prompts'][axis]=prompt;scores={}
            for key,value in row.items():
                if key.startswith('score.'):
                    mid=key.removeprefix('score.');metrics.setdefault(mid,{'id':mid,'label':mid.replace('_',' ').title(),'description':'Imported measurement; consult evaluator provenance.'})
                    scores[mid]={'value':float(value) if value else None,'status':'ok' if value else 'missing'}
            records.append({'id':row.get('id') or f'image-{i+1:05d}','condition':cid,'case':case,'axis':axis,'prompt':prompt,'seed':row['seed'],'strength':float(row.get('strength') or 1),'protocol':protocol['id'],'image':row['image'],'sha256':digest(local_file(out.parent,row['image'])),'scores':scores})
    data={'schema_version':1,'id':out.stem,'title':title,'description':'Imported comparison dataset','baseline_condition':baseline,'conditions':list(conditions.values()),'cases':list(cases.values()),'metrics':list(metrics.values()) or [{'id':'face_similarity','label':'Face similarity','description':'Largest detected face cosine to your reference centroid.'}],'protocols':[protocol],'records':records,'reference_images':[],'lessons':[]}
    validate(data,out.parent);write_json(out,data);return {'records':len(records),'out':str(out)}

def main():
    p=argparse.ArgumentParser(description='DeltaScope: paired comparisons, scoring and case studies for adapters and models')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('serve');s.add_argument('dataset',nargs='?');s.add_argument('--port',type=int,default=8890);s.add_argument('--face-model-dir')
    s=sub.add_parser('validate');s.add_argument('dataset')
    s=sub.add_parser('references');s.add_argument('dataset');s.add_argument('--folder',required=True)
    s=sub.add_parser('score');s.add_argument('dataset');s.add_argument('--evaluator',default='face_similarity');s.add_argument('--face-model-dir')
    s=sub.add_parser('export');s.add_argument('dataset');s.add_argument('--out',required=True)
    s=sub.add_parser('import-csv');s.add_argument('csv');s.add_argument('--out',required=True);s.add_argument('--title',default='My comparison');s.add_argument('--baseline',default='base');s.add_argument('--protocol')
    args=p.parse_args()
    if args.command=='serve':
        from .server import serve
        serve(args.dataset or Path(__file__).parent/'case_studies/conditional_identity.json',args.port,args.face_model_dir);return
    if args.command=='references':
        from .dataset import IMAGE_SUFFIXES
        path=Path(args.dataset).resolve();data,root=read(path);folder=(root/args.folder).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():raise ValueError('Reference folder must be inside the dataset directory')
        files=sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        if not files:raise ValueError('No reference images found')
        data['reference_images']=[str(p.relative_to(root)) for p in files];validate(data,root);write_json(path,data);result={'reference_images':len(files)}
    elif args.command=='validate':
        data,_=read(args.dataset);result={'valid':True,'records':len(data['records'])}
    elif args.command=='score':
        from .scoring import score_dataset
        result=score_dataset(args.dataset,args.evaluator,args.face_model_dir,lambda v:print(json.dumps(v),flush=True))
    elif args.command=='export':
        from .export import export_dataset
        result=export_dataset(args.dataset,args.out)
    else:result=import_csv(args.csv,args.out,args.title,args.baseline,args.protocol)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
