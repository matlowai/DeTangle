"""Pinned inputs and shared helpers for the local factor/block repair comparison."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,os,sys
import numpy as np
HERE=Path(__file__).resolve().parent
# Published copy: every location comes from the environment (research/README.md); the as-run layout is kept under DS.
DS=Path(os.environ.get('DETANGLE_RESEARCH_ROOT','detangle-research')).resolve()
RUNS=DS/'runs'
ROOT=DS
RUN=RUNS/'2026-10-18-factor-and-block-repair'
VENDOR=HERE/'vendor/fizgig'  # Fizgig sources at the revision in protocol.json (not included; only the block profile needs them)
PLAN=HERE/'protocol.json'
SOURCE=Path(os.environ.get('DETANGLE_SOURCE_LORA','source-lora.safetensors'))  # the character LoRA (not distributed)
CAPTURE=Path(os.environ.get('DETANGLE_CAPTURE_DIR','captures'))  # Xc.safetensors / Xt.safetensors
HISTORICAL=Path(os.environ.get('DETANGLE_HISTORICAL_LORA','historical-surgery.safetensors'))
STABLE_A=RUNS/'2026-10-17-seed-equivalence/adapter.safetensors'
OLD_STUDY=Path(os.environ.get('DETANGLE_SCORER_STUDY','scorer'))  # provides score_study.prepare_evaluator (ArcFace + 34 reference photos; not included)
TEMPLATE=RUNS/'2026-10-17-seed-equivalence/historical-compatible/render/graphs/historical_trigger_beach_171018.json'
LABELS={'base':'Base','source':'Original LoRA','a_side':'A-side surgery · CPU float64','b_side':'B-side surgery · CPU float64','fizgig':'Fizgig profiler block repair','historical':'Historical A-side surgery'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def profiler():
 """Run the inspected upstream pure profiling functions without importing its model runtime."""
 names={'ablation_order','windows','conditions','_small','_dist','_mean','summarise','verdicts','repair_settings'}
 tree=ast.parse((VENDOR/'block_profile.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==names
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/'block_profile.py'),'exec'),ns)
 return SimpleNamespace(**{n:ns[n] for n in names})

def block_map():
 tree=ast.parse((VENDOR/'krea2_driver.py').read_text());nodes=[];ns={'Block':lambda id,label,modules:SimpleNamespace(id=id,label=label,modules=modules),'BlockGroup':lambda label,blocks:SimpleNamespace(label=label,blocks=blocks)}
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_BLOCK_MODULES','_IO'} for t in n.targets):nodes.append(n)
  if isinstance(n,ast.ClassDef) and n.name=='Krea2Driver':nodes += [x for x in n.body if isinstance(x,ast.FunctionDef) and x.name=='block_map']
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/'krea2_driver.py'),'exec'),ns)
 groups=ns['block_map'](SimpleNamespace(description=SimpleNamespace(n_blocks=28)))
 return groups

def engine_for(keys):
 groups=block_map();mapping={m:b.id for g in groups for b in g.blocks for m in b.modules}
 modules=[k.removeprefix('diffusion_model.').removesuffix('.lora_A.weight') for k in keys if k.endswith('.lora_A.weight')]
 assert len(modules)==256 and all(m in mapping for m in modules)
 have={mapping[m] for m in modules};assert len(have)==32 and 'io' not in have
 return SimpleNamespace(primary_block_ids=have,block_groups=lambda:groups),mapping

def graph(cell,plan):
 g=json.loads((HERE/'graph-template.json').read_text());g['pos']['inputs']['text']=plan['prompts'][cell['prompt']]['text'];g['ksampler']['inputs']['seed']=cell['seed'];g['save']['inputs']['filename_prefix']=cell['stage']+'/'+cell['id']
 name=cell['adapter']
 if name is None:
  g.pop('lora');g['ksampler']['inputs']['model']=['unet',0]
 else:
  g['lora']['inputs']['lora_name']=name;g['lora']['inputs']['strength_model']=cell['strength'];g['ksampler']['inputs']['model']=['lora',0]
 return g

def read_plan():
 p=json.loads(PLAN.read_text());assert p['graph_template_sha256']==sha(HERE/'graph-template.json')
 if VENDOR.exists():  # published copy: the Fizgig vendor tree is optional
  for n,h in p['fizgig']['files'].items():assert sha(VENDOR/n)==h
 return p
