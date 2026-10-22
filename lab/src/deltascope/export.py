"""Export an explicit, read-only static dataset for Pages or offline use."""
from pathlib import Path
import json,re,shutil
from .dataset import read,summarize,local_file,digest,check_score_integrity
from .server import WEB

SAFE_SETTING_KEYS={'step','training_steps','rank','alpha','mu','ridge','strength','seed','model','precision','width','height','steps','cfg','sampler','scheduler','description','name','dtype','capture_seed','noise_seed','timesteps','capacity'}
def settings(value):
    if not isinstance(value,dict):return {}
    out={}
    for k,v in value.items():
        if k not in SAFE_SETTING_KEYS:continue
        if isinstance(v,str) and (v.startswith(('/','file:','~')) or re.match(r'^[A-Za-z]:[\\/]',v)):continue
        if isinstance(v,(str,int,float,bool,list)) or v is None:out[k]=v
    return out

def reject_local_paths(value):
    if isinstance(value,dict):
        for v in value.values():reject_local_paths(v)
    elif isinstance(value,list):
        for v in value:reject_local_paths(v)
    elif isinstance(value,str):
        if value.startswith(('/', '~/', 'file:')) or re.search(r'(?<![:/\w])/(?:[A-Za-z0-9_.-]+/)+[^\s]*|(?<![A-Za-z0-9])[A-Za-z]:[\\/]',value):raise ValueError('Export metadata contains an absolute local path; remove it before sharing')

def export_dataset(path,out):
    from PIL import Image,ImageOps
    data,root=read(path);out=Path(out).resolve()
    if out.exists():raise FileExistsError('Export destination already exists; choose a new directory')
    check_score_integrity(data,root)
    if any(s.get('status')=='stale' for r in data['records'] for s in r.get('scores',{}).values()):raise ValueError('Stale measurements: an image or reference set changed. Rescore before exporting.')
    out.mkdir(parents=True);(out/'images').mkdir();public={k:data[k] for k in ['schema_version','id','title','description','baseline_condition','cases','metrics','lessons','default_columns','default_protocol','default_case','research_guide','deep_dive','method_library'] if k in data}
    public['cases']=[{k:c[k] for k in ['id','label','split','prompts','axis_labels'] if k in c} for c in data['cases']]
    public['metrics']=[{k:m[k] for k in ['id','label','description','higher_is_better','units','flag_threshold'] if k in m} for m in data['metrics']]
    public['conditions']=[{k:c[k] for k in ['id','label','description','color'] if k in c}|{'settings':settings(c.get('settings',{}))} for c in data['conditions']]
    for c,source in zip(public['conditions'],data['conditions']):
        c['about']={k:v for k,v in source.get('about',{}).items() if k in ['summary','steps','difference','observe','technical']}
    public['protocols']=[{k:p[k] for k in ['id','label','description','matchable','default_columns'] if k in p}|{'settings':settings(p.get('settings',{}))} for p in data['protocols']]
    public['records']=[]
    for r in data['records']:
        source=local_file(root,r['image']);name=digest(source)+'.jpg';target=out/'images'/name
        if not target.exists():
            with Image.open(source) as image:ImageOps.exif_transpose(image).convert('RGB').save(target,quality=95,optimize=True)
        item={k:r[k] for k in ['id','condition','case','axis','seed','strength','protocol','prompt','sha256','native_image_sha256'] if k in r};item['image']='images/'+name;item['native_image_sha256']=r.get('native_image_sha256',digest(source));item['sha256']=digest(target)
        item['scores']={k:{'value':s['value'],'status':s['status'],'asset_sha256':digest(target),'input_sha256':s.get('input_sha256',r.get('native_image_sha256',r.get('sha256')))} for k,s in r.get('scores',{}).items()};public['records'].append(item)
    public['figures']=[]
    for f in data.get('figures',[]):
        source=local_file(root,f['image'])
        if f.get('sha256') and f['sha256']!=digest(source):raise ValueError('Figure changed since its manifest was recorded')
        name=digest(source)+'.png';target=out/'images'/name
        with Image.open(source) as image:ImageOps.exif_transpose(image).convert('RGB').save(target)
        public['figures'].append({'id':f['id'],'image':'images/'+name,'caption':f.get('caption',''),'sha256':digest(target)})
    reject_local_paths(public)
    public['display_only']=True;public['can_score']=False;public['records'],public['summary']=summarize(public)
    for name in ['app.js','styles.css']:shutil.copyfile(WEB/name,out/name)
    if (root/'explorer.html').is_file():shutil.copyfile(root/'explorer.html',out/'explorer.html')  # optional interactive explorer page shipped with the case study
    html=(WEB/'index.html').read_text().replace('<script src="app.js" defer>', '<script src="dataset.js"></script>\n<script src="app.js" defer>')
    (out/'index.html').write_text(html);(out/'.nojekyll').write_text('')
    payload=json.dumps(public,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    (out/'dataset.js').write_text('window.DELTASCOPE_DATA = '+payload+';\n');(out/'dataset.json').write_text(json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return {'images':len(public['records']),'out':str(out),'scoring':'disabled in static export','reference_images_exported':0}
