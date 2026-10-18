"""Validate experiment records and pair scores without silently changing cohorts."""
from pathlib import Path
import copy,hashlib,json,math,re
from collections import defaultdict
from statistics import mean

IMAGE_SUFFIXES={'.png','.jpg','.jpeg','.webp'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def local_file(root,value):
    if not isinstance(value,str) or not value:raise ValueError('Expected a relative file path')
    if Path(value).is_absolute():raise ValueError('Dataset assets must use relative paths')
    path=(root/value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Asset is missing or outside the dataset directory: '+value)
    return path

def read(path):
    path=Path(path).resolve();data=json.loads(path.read_text());validate(data,path.parent);return data,path.parent

def validate(data,root):
    if data.get('schema_version')!=1:raise ValueError('Expected schema_version 1')
    for field in ['conditions','cases','metrics','protocols']:
        rows=data.get(field)
        if not isinstance(rows,list) or not rows:raise ValueError('Missing '+field)
        ids=[r.get('id') for r in rows]
        if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)):raise ValueError('Duplicate or invalid IDs in '+field)
    conditions={r['id'] for r in data['conditions']};cases={r['id']:r for r in data['cases']};metrics={r['id'] for r in data['metrics']};protocols={r['id'] for r in data['protocols']}
    if data.get('baseline_condition') not in conditions:raise ValueError('Missing baseline condition')
    if not isinstance(data.get('records'),list):raise ValueError('Expected records list')
    ids=set();cells=set()
    for r in data['records']:
        if not isinstance(r.get('id'),str) or not r['id'] or r['id'] in ids:raise ValueError('Duplicate or missing record ID')
        ids.add(r['id'])
        if r.get('condition') not in conditions or r.get('case') not in cases or r.get('protocol') not in protocols:raise ValueError('Unknown condition, case or protocol')
        if r.get('axis') not in cases[r['case']].get('prompts',{}):raise ValueError('Axis has no exact prompt')
        if r.get('prompt')!=cases[r['case']]['prompts'][r['axis']]:raise ValueError('Record prompt differs from case prompt')
        if isinstance(r.get('seed'),bool) or not isinstance(r.get('seed'),(int,str)):raise ValueError('Seeds must be integers or exact strings')
        t=r.get('strength',1)
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t):raise ValueError('Strength must be finite')
        r['strength']=t
        cell=(r['condition'],r['case'],r['axis'],str(r['seed']),t,r['protocol'])
        if cell in cells:raise ValueError('Duplicate render cell')
        cells.add(cell)
        image=local_file(root,r.get('image'))
        if image.suffix.lower() not in IMAGE_SUFFIXES:raise ValueError('Only PNG, JPEG and WebP image assets are supported')
        for key,s in r.get('scores',{}).items():
            if key not in metrics:raise ValueError('Unknown metric '+key)
            value=s.get('value');status=s.get('status')
            if status not in ['ok','no_face','missing','error','pending','stale']:raise ValueError('Invalid score status')
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('Scores must be finite numbers or null')
            if (status=='ok')!=(value is not None):raise ValueError('Only successful measurements may have numeric values')
            binding=s.get('asset_sha256',r.get('sha256'))
            if status=='ok' and (not isinstance(binding,str) or not re.fullmatch(r'[0-9a-f]{64}',binding)):raise ValueError('Precomputed scores require an asset_sha256 or record sha256 binding')
    for figure in data.get('figures',[]):
        if local_file(root,figure['image']).suffix.lower() not in IMAGE_SUFFIXES:raise ValueError('Invalid figure image')
    if 'default_columns' in data:
        if not isinstance(data['default_columns'],list) or not data['default_columns']:raise ValueError('default_columns must be a nonempty list')
        for c in data['default_columns']:
            if not isinstance(c,dict) or c.get('condition') not in conditions or not any(r['condition']==c['condition'] and r['strength']==c.get('strength',1) for r in data['records']):raise ValueError('Invalid default column condition or strength')
    for lesson in data.get('lessons',[]):
        if lesson.get('case') and lesson['case'] not in cases:raise ValueError('Lesson refers to an unknown case')
    for ref in data.get('reference_images',[]):
        if local_file(root,ref).suffix.lower() not in IMAGE_SUFFIXES:raise ValueError('Invalid reference image')
    return data

def summarize(data):
    baselines=defaultdict(list)
    matchable={p['id']:p.get('matchable',True) for p in data['protocols']}
    def match(r):return r['case'],r['axis'],r['prompt'],str(r['seed']),r['protocol']
    for r in data['records']:
        if r['condition']==data['baseline_condition']:baselines[match(r)].append(r)
    rows=[];groups=defaultdict(list)
    for source in data['records']:
        r=copy.deepcopy(source);candidates=baselines[match(r)]
        # Multiple baselines are ambiguous even when their values happen to match.
        base=candidates[0] if len(candidates)==1 and matchable[r['protocol']] else None
        r['baseline_status']='unverified_protocol' if not matchable[r['protocol']] else 'matched' if base else 'ambiguous' if len(candidates)>1 else 'absent'
        r['baseline_id']=base['id'] if base else None;r['excess']={}
        for metric in data['metrics']:
            mid=metric['id'];s=r.get('scores',{}).get(mid,{'value':None,'status':'pending'});b=base.get('scores',{}).get(mid,{}) if base else {}
            r.setdefault('scores',{})[mid]=s
            r['excess'][mid]=s['value']-b['value'] if s.get('status')=='ok' and b.get('status')=='ok' else None
            groups[(r['condition'],r['strength'],r['axis'],r['protocol'],mid)].append((s,r['excess'][mid]))
        r['seed']=str(r['seed']);rows.append(r)
    stats=[]
    for (condition,strength,axis,protocol,metric),values in groups.items():
        raw=[s['value'] for s,_ in values if s.get('status')=='ok'];delta=[d for _,d in values if d is not None]
        stats.append(dict(condition=condition,strength=strength,axis=axis,protocol=protocol,metric=metric,images=len(values),measured=len(raw),missing=len(values)-len(raw),mean=mean(raw) if raw else None,paired=len(delta),mean_excess=mean(delta) if delta else None))
    return rows,stats


def check_score_integrity(data,root):
    """Mark obsolete measurements, retaining their original input bindings."""
    hashes={r['id']:digest(local_file(root,r['image'])) for r in data['records']}
    views,_=summarize(data);base_ids={r['id']:r['baseline_id'] for r in views}
    stale_references=set()
    for mid,provenance in data.get('evaluators',{}).items():
        expected=provenance.get('reference_assets')
        if expected is not None:
            current={ref:digest(local_file(root,ref)) for ref in data.get('reference_images',[])}
            if expected!=current:stale_references.add(mid)
    for r in data['records']:
        for mid,s in r.get('scores',{}).items():
            binding=s.get('asset_sha256',r.get('sha256'))
            if binding:s.setdefault('asset_sha256',binding)
            stale=binding is not None and binding!=hashes[r['id']]
            if s.get('baseline_asset_sha256') is not None:stale=stale or s['baseline_asset_sha256']!=hashes.get(base_ids[r['id']])
            stale=stale or mid in stale_references
            if stale and s.get('status')!='stale':
                s['previous_value']=s.get('value');s['value']=None;s['status']='stale';s['details']={'reason':'A measured image or reference set changed; rescore this metric.'}
    return hashes

def write_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
