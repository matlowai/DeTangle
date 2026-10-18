"""Loopback-only comparison server; explicit assets and one local scoring job."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import copy,json,mimetypes,secrets,shutil,threading
from .dataset import read,summarize,local_file,signature,check_score_integrity
from .scoring import score_dataset

WEB=Path(__file__).parent/'web'

def view_data(path):
    data,root=read(path);check_score_integrity(data,root);rows,stats=summarize(data);assets={}
    for r in rows:
        file=local_file(root,r['image']);key=signature(r['image'])[:24]+file.suffix.lower();assets[key]=file;r['image']='media/'+key
    result={k:copy.deepcopy(data[k]) for k in ['schema_version','id','title','description','baseline_condition','conditions','cases','metrics','protocols','lessons','default_columns','display_only','research_guide','deep_dive','method_library','figures'] if k in data}
    for f in result.get('figures',[]):
        file=local_file(root,f['image']);key=signature(f['image'])[:24]+file.suffix.lower();assets[key]=file;f['image']='media/'+key
    result.update(records=rows,summary=stats,reference_count=len(data.get('reference_images',[])))
    return result,assets

def serve(path,port=8890,model_dir=None):
    path=Path(path).resolve();read(path)
    job={'status':'idle'};lock=threading.Lock();token=secrets.token_urlsafe(24)
    cache={};data_lock=threading.Lock()
    def load_view(refresh=False):
        stamp=path.stat().st_mtime_ns
        with data_lock:
            if refresh or cache.get('stamp')!=stamp:
                data,assets=view_data(path);cache.update(stamp=stamp,data=data,assets=assets,asset_stamps={k:(p.stat().st_mtime_ns,p.stat().st_size) for k,p in assets.items()})
            return cache['data'],cache['assets']
    def run_job(metric):
        def progress(update):
            with lock:job.update(update)
        try:score_dataset(path,metric,model_dir,progress)
        except Exception as e:progress({'status':'error','message':str(e)})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):pass
        def allowed(self):return self.headers.get('Host') in {f'127.0.0.1:{port}',f'localhost:{port}'}
        def send_json(self,value,status=200):
            raw=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):self.get(True)
        def do_HEAD(self):self.get(False)
        def get(self,body):
            if not self.allowed():self.send_error(403);return
            route=urlsplit(self.path).path
            if route=='/api/dataset':
                cached,_=load_view(refresh=True);data=copy.deepcopy(cached);data['can_score']=not data.get('display_only',False);data['score_token']=token;data['evaluator_options']=['face_similarity','image_rmse'];self.send_json(data);return
            if route=='/api/job':
                with lock:snapshot=copy.deepcopy(job)
                self.send_json(snapshot);return
            if route in ['/','/index.html','/app.js','/styles.css']:file=WEB/('index.html' if route=='/' else route[1:])
            elif route.startswith('/media/'):
                _,assets=load_view();file=assets.get(route.removeprefix('/media/'))
                if file is None:self.send_error(404);return
                if (file.stat().st_mtime_ns,file.stat().st_size)!=cache['asset_stamps'][route.removeprefix('/media/')]:self.send_error(409,'Image changed; reload the dataset');return
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(file.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(file.stat().st_size));self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.end_headers()
            if body:
                with file.open('rb') as f:shutil.copyfileobj(f,self.wfile)
        def do_POST(self):
            if not self.allowed() or urlsplit(self.path).path!='/api/score':self.send_error(403);return
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{port}',f'http://localhost:{port}'}:self.send_error(403);return
            if self.headers.get('X-DeltaScope-Token')!=token:self.send_error(403);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1024:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(length));metric=payload.get('evaluator')
                if metric not in ['face_similarity','image_rmse']:raise ValueError('Custom plugins are launched explicitly from the CLI')
                data,_=read(path)
                if data.get('display_only'):raise ValueError('Built-in case studies use precomputed native-image measurements. Open your own dataset to score.')
                with lock:
                    if job['status']=='running':self.send_json({'error':'A scoring job is already running'},409);return
                    job.clear();job.update(status='running',done=0,total=len(data['records']),errors=0,metric=metric)
                threading.Thread(target=run_job,args=(metric,),daemon=True).start();self.send_json({'status':'started'},202)
            except (ValueError,TypeError,json.JSONDecodeError) as e:self.send_json({'error':str(e)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    port=server.server_address[1]
    print(f'DeltaScope: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
