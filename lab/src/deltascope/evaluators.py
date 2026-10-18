"""Optional evaluators. Model weights and reference photos are supplied by the user."""
from pathlib import Path
import contextlib,hashlib,importlib,io,json,math,os
from .dataset import digest,local_file,signature

class FaceSimilarity:
    uses_baseline=False
    metric_id='face_similarity'
    def __init__(self,data,root,model_dir):
        if not data.get('reference_images'):raise ValueError('Add your own reference_images to the dataset before scoring')
        if not model_dir:raise ValueError('Set --face-model-dir to an existing InsightFace model directory; no weights are downloaded automatically')
        directory=Path(model_dir).expanduser().resolve()
        required=['det_10g.onnx','w600k_r50.onnx']
        for name in required:
            if not (directory/name).is_file():raise ValueError('Missing required local model: '+name)
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
        import numpy as np,cv2,onnxruntime as ort
        from insightface import model_zoo
        from insightface.app.common import Face
        import importlib.metadata
        self.np=np;self.cv2=cv2;self.Face=Face;cv2.setNumThreads(1)
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        with contextlib.redirect_stdout(io.StringIO()):
            self.det=model_zoo.get_model(str(directory/required[0]),providers=['CPUExecutionProvider'],sess_options=options)
            self.rec=model_zoo.get_model(str(directory/required[1]),providers=['CPUExecutionProvider'],sess_options=options)
            self.det.prepare(ctx_id=-1,input_size=(640,640),det_thresh=.5);self.rec.prepare(ctx_id=-1)
        embeddings=[];refs=[];misses=[]
        reference_assets={value:digest(local_file(root,value)) for value in data['reference_images']}
        for value in data['reference_images']:
            path=local_file(root,value);faces=self.faces(path)
            if not faces:misses.append(value);continue
            face=max(faces,key=self.area);embeddings.append(face.normed_embedding);refs.append(digest(path))
        if not embeddings:raise ValueError('No faces detected in the reference images')
        if len(set(refs))!=len(refs):raise ValueError('Duplicate reference image bytes would change centroid weighting')
        self.centroid=np.stack(embeddings).mean(axis=0);norm=float(np.linalg.norm(self.centroid))
        if not math.isfinite(norm) or norm==0:raise ValueError('Reference centroid has zero or invalid norm')
        self.centroid/=norm
        self.provenance={'evaluator':'InsightFace ArcFace cosine to normalized mean of unit reference embeddings','selection':'largest bounding-box area','provider':'CPUExecutionProvider','detection_size':[640,640],'detection_threshold':.5,'reference_count':len(refs),'reference_no_face_count':len(misses),'reference_sha256':refs,'models':{n:digest(directory/n) for n in required},'versions':{'insightface':importlib.metadata.version('insightface'),'onnxruntime':ort.__version__,'numpy':np.__version__}}
        self.provenance['reference_assets']=reference_assets
        self.signature=signature(self.provenance)
    @staticmethod
    def area(f):return float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    def faces(self,path):
        image=self.cv2.imread(str(path))
        if image is None:raise ValueError('Image could not be decoded')
        boxes,kps=self.det.detect(image,max_num=0,metric='default');faces=[]
        for i,box in enumerate(boxes):
            f=self.Face(bbox=box[:4],kps=None if kps is None else kps[i],det_score=float(box[4]));self.rec.get(image,f);faces.append(f)
        return sorted(faces,key=self.area,reverse=True)
    def score(self,path,record):
        faces=self.faces(path);values=[float(f.normed_embedding@self.centroid) for f in faces]
        return {'value':values[0] if values else None,'status':'ok' if faces else 'no_face','details':{'faces':len(faces),'max_face_similarity':max(values) if values else None,'boxes':[f.bbox.tolist() for f in faces]}}

class ImageRMSE:
    uses_baseline=True
    metric_id='image_rmse'
    def __init__(self,data,root):
        from .dataset import summarize
        self.root=root;rows,_=summarize(data);self.rows={r['id']:r for r in rows};self.provenance={'evaluator':'RGB root mean square pixel difference from the exact matched baseline','units':'0..1','note':'Pixel difference is not perceptual quality or semantic preservation.'};self.signature=signature(self.provenance)
    def score(self,path,record):
        from PIL import Image,ImageChops,ImageStat
        r=self.rows[record['id']];base=self.rows.get(r['baseline_id'])
        if not base:return {'value':None,'status':'missing','details':{'reason':'No unique matching baseline'}}
        with Image.open(path) as a,Image.open(local_file(self.root,base['image'])) as b:
            if a.size!=b.size:return {'value':None,'status':'error','details':{'reason':'Image sizes differ; no automatic rescaling'}}
            values=ImageStat.Stat(ImageChops.difference(a.convert('RGB'),b.convert('RGB'))).rms
        return {'value':math.sqrt(sum(v*v for v in values)/3)/255,'status':'ok','details':{}}

def make(name,data,root,model_dir=None):
    if name=='face_similarity':return FaceSimilarity(data,root,model_dir)
    if name=='image_rmse':return ImageRMSE(data,root)
    if ':' not in name:raise ValueError('Unknown evaluator; plugins use module:factory')
    module,factory=name.split(':',1)
    evaluator=getattr(importlib.import_module(module),factory)(data,root)
    for attr in ['metric_id','signature','provenance','score']:
        if not hasattr(evaluator,attr):raise ValueError('Evaluator is missing '+attr)
    return evaluator
