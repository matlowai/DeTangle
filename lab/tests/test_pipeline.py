import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
from PIL import Image,PngImagePlugin
from deltascope.dataset import validate,summarize,write_json,digest,check_score_integrity
from deltascope.cli import import_csv
from deltascope.scoring import score_dataset
from deltascope.export import export_dataset

class PipelineTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  for name,color in [('base.png',(0,0,0)),('edited.png',(255,0,0)),('ref.png',(128,128,128))]:Image.new('RGB',(16,16),color).save(self.root/name)
  self.data={'schema_version':1,'id':'test','title':'Test','baseline_condition':'base','conditions':[{'id':'base','label':'Base'},{'id':'edited','label':'Edited','settings':{'mu':.3,'checkpoint_path':'/private/weights/model.safetensors'}}],'cases':[{'id':'p','label':'Portrait','prompts':{'generic':'a person','target':'token, a person'}}],'metrics':[{'id':'identity','label':'Identity'}],'protocols':[{'id':'p1','matchable':True,'settings':{'steps':8}}],'reference_images':['ref.png'],'records':[]}
  for condition,image,values in [('base','base.png',[.1,.2]),('edited','edited.png',[.3,.7])]:
   for axis,value in zip(['generic','target'],values):self.data['records'].append({'id':condition+'-'+axis,'condition':condition,'case':'p','axis':axis,'seed':9007199254740993,'strength':1,'protocol':'p1','prompt':self.data['cases'][0]['prompts'][axis],'image':image,'sha256':digest(self.root/image),'scores':{'identity':{'value':value,'status':'ok'}}})
 def tearDown(self):self.temp.cleanup()
 def test_pairing_and_exact_large_seed(self):
  validate(self.data,self.root);rows,groups=summarize(self.data);r=next(r for r in rows if r['id']=='edited-generic');self.assertAlmostEqual(r['excess']['identity'],.2);self.assertEqual(r['seed'],'9007199254740993')
 def test_different_protocol_is_not_paired(self):
  self.data['protocols'].append({'id':'p2','settings':{'steps':16}});self.data['records'][-1]['protocol']='p2';rows,_=summarize(self.data);self.assertEqual(rows[-1]['baseline_status'],'absent');self.assertIsNone(rows[-1]['excess']['identity'])
 def test_unverified_protocol_is_not_paired(self):
  self.data['protocols'][0]['matchable']=False;rows,_=summarize(self.data);self.assertTrue(all(r['baseline_status']=='unverified_protocol' for r in rows))
 def test_missing_measurements_are_not_zero(self):
  self.data['records'][-1]['scores']['identity']={'status':'no_face','value':None};rows,groups=summarize(self.data);g=next(g for g in groups if g['condition']=='edited' and g['axis']=='target');self.assertEqual(g['measured'],0);self.assertIsNone(g['mean']);self.assertIsNone(rows[-1]['excess']['identity'])
 def test_ambiguous_baselines_rejected(self):
  other=copy.deepcopy(self.data['records'][0]);other.update(id='base-repeat',strength=2);self.data['records'].append(other);validate(self.data,self.root);rows,_=summarize(self.data);r=next(r for r in rows if r['id']=='edited-generic');self.assertEqual(r['baseline_status'],'ambiguous');self.assertIsNone(r['excess']['identity'])
 def test_duplicate_cell_and_prompt_mismatch_rejected(self):
  other=copy.deepcopy(self.data['records'][0]);other['id']='repeat';self.data['records'].append(other)
  with self.assertRaisesRegex(ValueError,'Duplicate render'):validate(self.data,self.root)
  self.data['records'].pop();self.data['records'][0]['prompt']='different'
  with self.assertRaisesRegex(ValueError,'prompt differs'):validate(self.data,self.root)
 def test_paths_cannot_escape(self):
  self.data['records'][0]['image']='../private.png'
  with self.assertRaises(ValueError):validate(self.data,self.root)
 def test_nonfinite_score_rejected(self):
  self.data['records'][0]['scores']['identity']['value']=float('nan')
  with self.assertRaisesRegex(ValueError,'finite'):validate(self.data,self.root)
 def test_cache_includes_changed_baseline_and_protocol(self):
  path=self.root/'dataset.json';write_json(path,self.data);score_dataset(path,'image_rmse');first=json.loads(path.read_text());self.assertAlmostEqual(first['records'][-1]['scores']['image_rmse']['value'],(1/3)**.5)
  db=self.root/'.adapter-review/scores.sqlite'
  with sqlite3.connect(db) as c:first_count=c.execute('SELECT count(*) FROM scores').fetchone()[0]
  score_dataset(path,'image_rmse')
  with sqlite3.connect(db) as c:self.assertEqual(c.execute('SELECT count(*) FROM scores').fetchone()[0],first_count)
  Image.new('RGB',(16,16),(255,0,0)).save(self.root/'base.png');score_dataset(path,'image_rmse');self.assertEqual(json.loads(path.read_text())['records'][-1]['scores']['image_rmse']['value'],0)
  changed=json.loads(path.read_text());changed['protocols'][0]['settings']['steps']=12;write_json(path,changed)
  with sqlite3.connect(db) as c:count=c.execute('SELECT count(*) FROM scores').fetchone()[0]
  score_dataset(path,'image_rmse')
  with sqlite3.connect(db) as c:self.assertGreater(c.execute('SELECT count(*) FROM scores').fetchone()[0],count)
 def test_export_excludes_references_paths_and_image_metadata(self):
  png=PngImagePlugin.PngInfo();png.add_text('workflow','/private/weights/model.safetensors');Image.new('RGB',(16,16)).save(self.root/'base.png',pnginfo=png)
  
  for r in self.data['records']:r['sha256']=digest(self.root/r['image'])
  path=self.root/'dataset.json';write_json(path,self.data);out=self.root/'public';export_dataset(path,out);text=(out/'dataset.json').read_text();self.assertNotIn('reference_images',text);self.assertNotIn('/private',text);self.assertNotIn('checkpoint_path',text);self.assertFalse((out/'ref.png').exists());self.assertEqual(len(list((out/'images').glob('*'))),2)
  for image in (out/'images').glob('*'):
   with Image.open(image) as im:self.assertNotIn('workflow',im.info);self.assertNotIn('exif',im.info)
  with self.assertRaises(FileExistsError):export_dataset(path,out)
 def test_csv_without_sampling_protocol_cannot_claim_pairing(self):
  csv=self.root/'records.csv';csv.write_text('condition,case,axis,prompt,seed,strength,image,score.identity\nbase,p,generic,a person,1,1,base.png,0.1\nedited,p,generic,a person,1,1,edited.png,0.4\n');out=self.root/'imported.json';import_csv(csv,out,'Imported','base');data=json.loads(out.read_text());rows,_=summarize(data);self.assertIsNone(rows[-1]['excess']['identity'])

 def test_stale_metric_does_not_become_valid_when_another_is_scored(self):
  path=self.root/'dataset.json';write_json(path,self.data)
  Image.new('RGB',(16,16),(0,255,0)).save(self.root/'edited.png')
  score_dataset(path,'image_rmse');updated=json.loads(path.read_text());r=updated['records'][-1]
  self.assertEqual(r['scores']['identity']['status'],'stale');self.assertIsNone(r['scores']['identity']['value']);self.assertEqual(r['scores']['image_rmse']['status'],'ok')
  with self.assertRaisesRegex(ValueError,'Stale'):export_dataset(path,self.root/'bad-export')
 def test_view_marks_changed_image_scores_stale(self):
  from deltascope.server import view_data
  path=self.root/'dataset.json';write_json(path,self.data);Image.new('RGB',(16,16),(0,255,0)).save(self.root/'edited.png');view,_=view_data(path)
  self.assertEqual(view['records'][-1]['scores']['identity']['status'],'stale');self.assertIsNone(view['records'][-1]['excess']['identity'])
 def test_changed_reference_invalidates_only_dependent_measurements(self):
  self.data['evaluators']={'identity':{'reference_assets':{'ref.png':digest(self.root/'ref.png')}}}
  path=self.root/'dataset.json';write_json(path,self.data)
  Image.new('RGB',(16,16),(255,255,255)).save(self.root/'ref.png');score_dataset(path,'image_rmse')
  changed=json.loads(path.read_text());r=changed['records'][-1]
  self.assertEqual(r['scores']['identity']['status'],'stale');self.assertIsNone(r['scores']['identity']['value']);self.assertEqual(r['scores']['image_rmse']['status'],'ok')
  with self.assertRaisesRegex(ValueError,'Stale'):export_dataset(path,self.root/'stale-reference-export')
 def test_reference_rescore_binds_new_reference_provenance(self):
  from unittest.mock import patch
  from deltascope.dataset import signature as make_signature
  self.data['evaluators']={'identity':{'reference_assets':{'ref.png':digest(self.root/'ref.png')}}}
  path=self.root/'dataset.json';write_json(path,self.data);Image.new('RGB',(16,16),(255,255,255)).save(self.root/'ref.png')
  class Fake:
   uses_baseline=False
   metric_id='identity'
   provenance={'reference_assets':{'ref.png':digest(self.root/'ref.png')}}
   signature=make_signature(provenance)
   def score(self,image,record):return {'value':.8,'status':'ok'}
  with patch('deltascope.evaluators.make',return_value=Fake()):score_dataset(path,'identity')
  fresh=json.loads(path.read_text());check_score_integrity(fresh,self.root);self.assertTrue(all(r['scores']['identity']['status']=='ok' for r in fresh['records']))
  fresh['reference_images']=[];check_score_integrity(fresh,self.root);self.assertTrue(all(r['scores']['identity']['status']=='stale' for r in fresh['records']))
 def test_strength_default_and_invalid_default_columns(self):
  del self.data['records'][0]['strength'];validate(self.data,self.root);self.assertEqual(self.data['records'][0]['strength'],1)
  self.data['default_columns']=[{'condition':'absent','strength':1}]
  with self.assertRaisesRegex(ValueError,'default column'):validate(self.data,self.root)
 def test_nested_export_paths_cannot_escape_filter(self):
  self.data['conditions'][1]['settings']['timesteps']=[{'path':str(self.root/'private')}];self.data['conditions'][1]['about']={'summary':'Example','internal_checkpoint':str(self.root/'private/model')};path=self.root/'dataset.json';write_json(path,self.data)
  with self.assertRaisesRegex(ValueError,'absolute local path'):export_dataset(path,self.root/'bad-nested')
  del self.data['conditions'][1]['settings']['timesteps'];write_json(path,self.data);export_dataset(path,self.root/'clean-about');text=(self.root/'clean-about/dataset.json').read_text();self.assertNotIn('internal_checkpoint',text)

if __name__=='__main__':unittest.main()
