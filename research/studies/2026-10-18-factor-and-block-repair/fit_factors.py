"""Fit A-only and B-only from identical source-input captures, on CPU float64."""
from common import *
import os,time,torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file
sys.path.insert(0,str(HERE.parents[2]/'src'))  # this repository's package
from delta_surgery_seed.core import solve,map_layers,response_metrics

def stats(a,b,bb,x):
 z=a.double()@x.double().T;b=b.double();bb=bb.double();v=b.T@b;vv=bb.T@bb;cross=b.T@bb;db=bb-b
 old=float(((v@z)*z).sum());new=float(((vv@z)*z).sum());dot=float(((cross@z)*z).sum());diff=float((((db.T@db)@z)*z).sum())
 return {'retained':max(new,0)**.5/old**.5 if old>0 else None,'relative_change':max(diff,0)**.5/old**.5 if old>0 else None,'cosine':dot/(old*new)**.5 if old*new>0 else None}

def main():
 p=read_plan();start=time.perf_counter();src=load_file(SOURCE);outputs={side:{k:v for k,v in src.items()} for side in ['a_side','b_side']};rows={s:{} for s in outputs};elapsed={s:0. for s in outputs}
 assert sha(SOURCE)==p['source_sha256']
 with safe_open(SOURCE,framework='pt') as f:meta=f.metadata() or {}
 with safe_open(CAPTURE/'Xc.safetensors',framework='pt') as fc,safe_open(CAPTURE/'Xt.safetensors',framework='pt') as ft:
  mapping=map_layers(src,fc.keys());assert len(mapping)==256 and set(fc.keys())==set(ft.keys())
  for i,(ak,cap) in enumerate(sorted(mapping.items()),1):
   bk=ak.removesuffix('lora_A.weight')+'lora_B.weight';a,b=src[ak],src[bk];xc,xt=fc.get_tensor(cap),ft.get_tensor(cap)
   t=time.perf_counter();ap,ai=solve(a,xc,xt,.3,.001,'zero','auto');ap=ap.to(a.dtype);outputs['a_side'][ak]=ap.contiguous()
   ai.update(class_fit=response_metrics(a,b,ap,xc),trigger_fit=response_metrics(a,b,ap,xt));rows['a_side'][ak]=ai;elapsed['a_side']+=time.perf_counter()-t
   t=time.perf_counter();zc=xc.double()@a.double().T;zt=xt.double()@a.double().T;bp,bi=solve(b,zc,zt,.3,.001,'zero','primal');bp=bp.to(b.dtype);outputs['b_side'][bk]=bp.contiguous()
   bi.update(class_fit=stats(a,b,bp,xc),trigger_fit=stats(a,b,bp,xt));rows['b_side'][bk]=bi;elapsed['b_side']+=time.perf_counter()-t
   if i%16==0:write(RUN/'fit-progress.json',{'layers_done':i,'layers_total':256,'wall_seconds':time.perf_counter()-start});print('Fit',i,'/256',flush=True)
 for side,tensors in outputs.items():
  edited='.lora_A.weight' if side=='a_side' else '.lora_B.weight'
  assert set(tensors)==set(src)
  for key,v in tensors.items():
   assert v.shape==src[key].shape and v.dtype==src[key].dtype and torch.isfinite(v).all()
   if not key.endswith(edited):assert torch.equal(v,src[key]),key
  path=RUN/'adapters'/f'{side}.safetensors';assert not path.exists()
  provenance={'method':side,'mu':.3,'ridge':.001,'center':'zero','precision':'CPUfloat64','source_sha256':p['source_sha256'],'input_capture_sha256':p['capture_sha256']}
  save_file(tensors,path,metadata={**meta,'factor_comparison':json.dumps(provenance)})
  write(RUN/(side+'-fit.json'),{**provenance,'output_sha256':sha(path),'layers':rows[side],'factor_fit_and_diagnostics_seconds':elapsed[side],'invariants':'Only the named factor changes; all other tensors, shapes, dtypes and original metadata retained.'})
 write(RUN/'fit-complete.json',{'wall_seconds':time.perf_counter()-start,'cpu_threads':torch.get_num_threads(),'factor_seconds':elapsed,'source_sha256':p['source_sha256'],'captures_sha256':p['capture_sha256']})
 print('Both factors fitted',elapsed,flush=True)
if __name__=='__main__':main()
