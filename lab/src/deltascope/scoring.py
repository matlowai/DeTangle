"""Resumable scoring with image, evaluator, and matched-baseline cache keys."""
from pathlib import Path
import json,math,sqlite3,time
from . import evaluators
from .dataset import digest,read,signature,write_json,local_file,summarize,check_score_integrity

def score_dataset(path,evaluator_name='face_similarity',model_dir=None,progress=None):
    path=Path(path).resolve();data,root=read(path)
    if data.get('display_only'):raise ValueError('This built-in study contains display copies and precomputed original-image scores. Score a dataset with your own images and references instead.')
    hashes=check_score_integrity(data,root)
    evaluator=evaluators.make(evaluator_name,data,root,model_dir)
    mid=evaluator.metric_id
    if mid not in {m['id'] for m in data['metrics']}:
        data['metrics'].append({'id':mid,'label':mid.replace('_',' ').title(),'description':evaluator.provenance.get('note',''),'higher_is_better':mid!='image_rmse'})
    cache=root/'.adapter-review';cache.mkdir(exist_ok=True)
    db=sqlite3.connect(cache/'scores.sqlite');db.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    rows,_=summarize(data);by_id={r['id']:r for r in rows}
    errors=0
    try:
        for index,r in enumerate(data['records']):
            baseline=by_id[r['id']]['baseline_id']
            hashes[r['id']]=digest(local_file(root,r['image']))
            if baseline:hashes[baseline]=digest(local_file(root,by_id[baseline]['image']))
            key=signature({'metric':mid,'evaluator':evaluator.signature,'image':hashes[r['id']],'baseline':hashes.get(baseline) if getattr(evaluator,'uses_baseline',True) else None,'condition':r['condition'],'case':r['case'],'axis':r['axis'],'prompt':r['prompt'],'seed':str(r['seed']),'strength':r['strength'],'protocol':next(p for p in data['protocols'] if p['id']==r['protocol']),'record_inputs':{k:v for k,v in r.items() if k not in ['scores','sha256']},'condition_settings':next(c for c in data['conditions'] if c['id']==r['condition'])})
            saved=db.execute('SELECT value FROM scores WHERE key=?',(key,)).fetchone()
            if saved:result=json.loads(saved[0])
            else:
                try:
                    result=evaluator.score(local_file(root,r['image']),r)
                    if digest(local_file(root,r['image']))!=hashes[r['id']]:raise ValueError('Image changed while it was being measured')
                    if baseline and getattr(evaluator,'uses_baseline',True) and digest(local_file(root,by_id[baseline]['image']))!=hashes[baseline]:raise ValueError('Baseline changed while it was being measured')
                    value=result.get('value')
                    if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('Evaluator returned a nonfinite score')
                    if result.get('status') not in ['ok','no_face','missing','error']:raise ValueError('Evaluator returned an invalid status')
                    if (result.get('status')=='ok')!=(value is not None):raise ValueError('Evaluator returned inconsistent status/value')
                except Exception as error:result={'value':None,'status':'error','details':{'reason':str(error)}}
                if result['status']!='error':db.execute('INSERT OR REPLACE INTO scores VALUES (?,?)',(key,json.dumps(result,allow_nan=False)));db.commit()
            result['asset_sha256']=hashes[r['id']];result['input_sha256']=hashes[r['id']];result['evaluator_signature']=evaluator.signature
            if getattr(evaluator,'uses_baseline',True):result['baseline_asset_sha256']=hashes.get(baseline)
            errors+=result['status']=='error';r.setdefault('scores',{})[mid]=result;r['sha256']=hashes[r['id']]
            if progress:progress({'status':'running','done':index+1,'total':len(rows),'errors':errors,'metric':mid})
        data.setdefault('evaluators',{})[mid]=evaluator.provenance
        check_score_integrity(data,root)
        write_json(path,data)
        if progress:progress({'status':'complete','done':len(rows),'total':len(rows),'errors':errors,'metric':mid})
    finally:db.close()
    return {'images':len(rows),'errors':errors,'metric':mid,'dataset':str(path)}
