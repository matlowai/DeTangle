"""Native scoring and the pinned Fizgig profile recommendation; no test-set tuning."""
from common import *
import argparse,os,time
from PIL import Image
import cv2
from safetensors import safe_open
from safetensors.torch import load_file,save_file
import torch


def setup():
 sys.path.insert(0,str(OLD_STUDY))
 from score_study import prepare_evaluator
 app,centroid,provenance,bleed=prepare_evaluator()
 centroid2,refs,used=bleed.build_reference_centroid(app);assert np.array_equal(centroid,centroid2) and len(refs)==34
 provenance['profile_score']='Upstream mean of reference-to-image unit-embedding cosines, using the common34-reference set; evaluation uses normalized reference centroid.'
 return app,centroid,refs,provenance

def measure(app,centroid,refs,path):
 pixels=cv2.imread(str(path));assert pixels is not None
 faces=sorted(app.get(pixels),key=lambda f:float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])),reverse=True)
 if not faces:return {'similarity':None,'profile_similarity':None,'face_count':0,'status':'no_face'}
 e=faces[0].normed_embedding
 return {'similarity':float(e@centroid),'profile_similarity':float((refs@e).mean()),'face_count':len(faces),'status':'ok'}

def score(stage,partial=False):
 import fcntl
 with (RUN/(stage+'-scoring.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _score(stage,partial)

def _score(stage,partial=False):
 p=read_plan();out=RUN/'render';manifest=json.loads((out/(stage+'-manifest.json')).read_text())
 if stage in ['secondary','preset']:
  stage_plan=HERE/('secondary_protocol.json' if stage=='secondary' else 'preset_protocol.json')
  secondary=json.loads(stage_plan.read_text());assert secondary['parent_protocol_sha256']==sha(PLAN);cells=secondary['cells']
 else:cells=[c for c in p['cells'] if c['stage']==stage]
 expected={c['id']:c for c in cells}
 stage_hash=sha(stage_plan) if stage in ['secondary','preset'] else sha(PLAN)
 assert len({r['id'] for r in manifest})==len(manifest)
 assert set(r['id'] for r in manifest)<=set(expected)
 if not partial:assert len(manifest)==len(expected)
 path=out/(stage+'-scores.json');saved=json.loads(path.read_text()) if path.exists() else None
 rows=saved['records'] if saved else [];cache={r['id']:r for r in rows};assert len(cache)==len(rows)
 app,centroid,refs,provenance=setup()
 if saved:assert saved['protocol_sha256']==sha(PLAN) and saved.get('stage_protocol_sha256',saved['protocol_sha256'])==stage_hash and saved['provenance']['signature']==provenance['signature']
 start=time.perf_counter();initial_count=len(rows)
 for r in manifest:
  cell=expected[r['id']];assert all(r[k]==v for k,v in cell.items())
  image_path=out/r['file'];assert sha(image_path)==r['sha256'];g=graph(cell,p)
  assert json.loads((out/r['graph']).read_text())==g
  with Image.open(image_path) as im:assert im.size==(832,1216) and json.loads(im.info['prompt'])==g
  if r['id'] in cache:
   assert all(cache[r['id']][k]==v for k,v in r.items());continue
  row={**r,**measure(app,centroid,refs,image_path)};rows.append(row);cache[row['id']]=row
  write(path,{'protocol_sha256':sha(PLAN),'stage_protocol_sha256':stage_hash,'complete':False,'provenance':provenance,'records':rows})
  print('Scored',stage,len(rows),'/',len(manifest),r['id'],flush=True)
 complete=len(rows)==len(expected);write(path,{'protocol_sha256':sha(PLAN),'stage_protocol_sha256':stage_hash,'complete':complete,'provenance':provenance,'records':rows})
 timing_path=RUN/(stage+'-scoring-timing.json');history=[]
 if timing_path.exists():
  previous=json.loads(timing_path.read_text());history=previous.get('invocations',[{k:v for k,v in previous.items() if k!='invocations'}])
 current={'wall_seconds_after_evaluator_initialization':time.perf_counter()-start,'records':len(rows),'newly_scored':len(rows)-initial_count,'complete':complete}
 write(timing_path,{**current,'invocations':history+[current]})
 return rows

def native_bake(src,block_settings,mapping):
 """Use upstream FamilyLoRA.bake, with one adapter and a PEFT key mapping."""
 tree=ast.parse((VENDOR/'lora.py').read_text());fn=next(x for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='FamilyLoRA' for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='bake')
 dummy=type('OtherAdapter',(),{});ns={'torch':torch,'LoKR':dummy,'LoHa':dummy};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(VENDOR/'lora.py'),'exec'),ns)
 wrapped={}
 for ak,a in src.items():
  if not ak.endswith('.lora_A.weight'):continue
  m=ak.removeprefix('diffusion_model.').removesuffix('.lora_A.weight');bk=ak.removesuffix('lora_A.weight')+'lora_B.weight';on,scale=block_settings.get(mapping[m],(True,1.0))
  assert a.shape[0]==32 and src[bk].shape[1]==32
  wrapped[m]=SimpleNamespace(adapters={'primary':(SimpleNamespace(weight=a),SimpleNamespace(weight=src[bk]))},scales={'primary':scale if on else 0.})
 def keys(m):
  prefix='diffusion_model.'+m;return prefix+'.lora_A.weight',prefix+'.lora_B.weight',prefix+'.alpha'
 state=SimpleNamespace(wrapped=wrapped,_keys=keys)
 baked,ranks=ns['bake'](state,['primary'],dtype=torch.bfloat16)
 # Our source uses implicit alpha/rank=1. The explicit upstream alpha is the
 # rank; dropping it when restoring the original PEFT layout preserves scale.
 for m,r in ranks.items():assert float(baked.pop(keys(m)[2]))==r==32
 assert baked and set(baked)<=set(src)
 for ak,a in src.items():
  if not ak.endswith('.lora_A.weight'):continue
  m=ak.removeprefix('diffusion_model.').removesuffix('.lora_A.weight');bk=ak.removesuffix('lora_A.weight')+'lora_B.weight';on,scale=block_settings.get(mapping[m],(True,1.))
  if on:
   assert torch.equal(baked[ak],a)
   assert torch.equal(baked[bk],(src[bk].float()*scale).to(src[bk].dtype))
  else:assert ak not in baked and bk not in baked
 return baked

def select():
 p=read_plan();out=RUN/'render';saved=json.loads((out/'profile-scores.json').read_text());assert saved['complete'] and saved['protocol_sha256']==sha(PLAN)
 rows=saved['records'];src=load_file(SOURCE);engine,mapping=engine_for(src);native=profiler();conds=native.conditions(engine,False)
 result={'blocks':native.ablation_order(engine),'windows':native.windows(engine),'conds':[(cid,label) for cid,label,on in conds],'seeds':[181801,181802],'size':[832,1216],'prompt':p['prompts']['calibration_studio_target']['text'],'class_prompt':p['prompts']['calibration_studio_generic']['text'],'thumbs':{},'pics':{},'scores':{}}
 for role,kind in [('generic','class'),('target','trigger')]:
  result['pics'][kind]={};result['scores'][kind]={}
  for arm,c in p['profile_condition_map'].items():
   rs=sorted((r for r in rows if r['arm']==arm and r['role']==role),key=lambda r:r['seed']);assert [r['seed'] for r in rs]==result['seeds']
   result['pics'][kind][c['id']]=[native._small(Image.open(out/r['file'])) for r in rs]
   result['scores'][kind][c['id']]=[r['profile_similarity'] for r in rs]
 summary=native.summarise(result);blocks,notes=native.repair_settings(summary,down=.5);verdicts=native.verdicts(summary)
 baked=native_bake(src,blocks,mapping)
 with safe_open(SOURCE,framework='pt') as f:meta=f.metadata() or {}
 target=RUN/'adapters/fizgig.safetensors';assert not target.exists()
 save_file(baked,target,metadata={**meta,'fizgig_profile_comparison':json.dumps({'revision':p['fizgig']['sha'],'settings':blocks,'global_strength':1.0})})
 write(RUN/'fizgig-profile.json',summary)
 write(RUN/'fizgig-selection.json',{'protocol_sha256':sha(PLAN),'fizgig_revision':p['fizgig']['sha'],'conditions':18,'calibration_images':72,'reference_count':34,'block_settings':blocks,'notes':notes,'verdicts':verdicts,'output_sha256':sha(target),'source_sha256':sha(SOURCE),'retained_modules':len(baked)//2,'native_bake_verified':True,'scope':'Pinned upstream profile summary/verdict/recommendation and bake functions, shared renderer and common reference set; not manual Studio tuning or a benchmark of all Fizgig features.'})
 print('Fizgig recommended:',json.dumps({'settings':blocks,'notes':notes}),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['profile','evaluation','secondary','preset','select']);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 if args.command=='select':select()
 else:score(args.command,args.partial)
