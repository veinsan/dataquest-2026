#!/usr/bin/env python
import os,sys,json,time,gc,traceback,argparse,random,hashlib,inspect,re,importlib.metadata
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from TALENT.model.models.modernNCA import ModernNCA

P=argparse.ArgumentParser(); P.add_argument('--root',required=True); P.add_argument('--mode',choices=['runtime','screen','full','resume_test'],required=True); P.add_argument('--fold',type=int,required=True); a=P.parse_args()
ROOT=Path(a.root); MODE=a.mode; FOLD=a.fold; DEVICE=torch.device('cuda:0')
if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError(f'Worker must see exactly one GPU, got {torch.cuda.device_count()}')
CFG=json.load(open(ROOT/'runtime_contract.json')); CONFIGS=json.load(open(ROOT/'modernnca_configs_v12.json')); SEM=json.load(open(ROOT/'semantic_contract_v12.json'))
# Prove child process uses the same official TALENT source/config contract as the parent.
_model_file=Path(inspect.getfile(ModernNCA)).resolve()
_child_model_sha=hashlib.sha256(_model_file.read_bytes()).hexdigest()
if _child_model_sha != CFG['modernnca_model_sha256']:
    raise RuntimeError(f'Worker ModernNCA source mismatch: parent={CFG["modernnca_model_sha256"]}, child={_child_model_sha}')
try:
    _du=json.loads(importlib.metadata.distribution('TALENT').read_text('direct_url.json') or '{}')
    _child_commit=_du.get('vcs_info',{}).get('commit_id')
except Exception as e:
    raise RuntimeError('Worker cannot resolve TALENT Git provenance') from e
if _child_commit != CFG['talent_commit']:
    raise RuntimeError(f'Worker TALENT commit mismatch: parent={CFG["talent_commit"]}, child={_child_commit}')
_child_cfg_sha=hashlib.sha256(json.dumps(CONFIGS,sort_keys=True).encode()).hexdigest()
if _child_cfg_sha != CFG['config_manifest_sha256'] or len(CONFIGS)!=8 or [c['config_id'] for c in CONFIGS]!=list(range(8)):
    raise RuntimeError(f'Worker fixed-config contract mismatch: {_child_cfg_sha}')
_ctor=list(inspect.signature(ModernNCA.__init__).parameters)
_fwd_src=inspect.getsource(ModernNCA.forward)
if _ctor != ['self','d_in','d_num','d_out','dim','dropout','d_block','n_blocks','num_embeddings','temperature','sample_rate']:
    raise RuntimeError(f'Worker ModernNCA constructor drift: {_ctor}')
for _token in ['self.sample_rate','torch.cdist','fill_diagonal_','F.softmax','torch.mm']:
    if _token not in _fwd_src: raise RuntimeError(f'Worker ModernNCA behavior drift: missing {_token}')
TRAIN=pd.read_csv(ROOT/'v12_train_unique_raw.csv'); TEST=pd.read_csv(ROOT/'v12_test_unique_raw.csv')
SEED=42; BATCH=int(CFG['batch_size']); INFER=int(CFG['infer_batch']); MAXE=int(CFG['max_epoch']); PAT=int(CFG['patience'])
random.seed(SEED+FOLD); np.random.seed(SEED+FOLD); torch.manual_seed(SEED+FOLD); torch.cuda.manual_seed_all(SEED+FOLD)
OUT=ROOT/f'{MODE}_fold{FOLD}'; OUT.mkdir(parents=True,exist_ok=True)

def mem(tag):
    torch.cuda.synchronize(); d={'tag':tag,'allocated':int(torch.cuda.memory_allocated()),'reserved':int(torch.cuda.memory_reserved()),'peak':int(torch.cuda.max_memory_allocated())}; print('MEM',json.dumps(d),flush=True); return d

class Prep:
    def __init__(self): self.cont=SEM['continuous']; self.ord=SEM['ordinal']; self.nom=SEM['nominal']; self.special={k:set(map(float,v)) for k,v in SEM['special_codes'].items()}
    def fit(self,df):
        self.cont_med={}; self.ord_med={}
        C=[]
        for c in self.cont:
            s=pd.to_numeric(df[c],errors='coerce').astype(float); s=s.mask(s.isin(self.special.get(c,set())))
            self.cont_med[c]=float(s.median())
            C.append(s.fillna(self.cont_med[c]).to_numpy(float))
        self.cscale=StandardScaler().fit(np.column_stack(C)) if C else None
        for c in self.ord:
            s=pd.to_numeric(df[c],errors='coerce').astype(float); self.ord_med[c]=float(s.median())
        raw_missing=df[SEM['features']].isna().sum(axis=1).to_numpy(float)
        sp=np.zeros(len(df))
        for c,v in self.special.items(): sp += pd.to_numeric(df[c],errors='coerce').astype(float).isin(v).to_numpy(float)
        self.miss_scale=StandardScaler().fit((raw_missing+sp).reshape(-1,1))
        self.ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        self.ohe.fit(df[self.nom].astype(object).where(df[self.nom].notna(),'__MISSING__').astype(str))
        self.n_num=len(self.cont)+len(self.ord)+1
        return self
    def transform(self,df):
        nums=[]; flags=[]
        if self.cont:
            base=[]
            for c in self.cont:
                orig=pd.to_numeric(df[c],errors='coerce').astype(float); sf=orig.isin(self.special.get(c,set())); miss=orig.isna()|sf
                base.append(orig.mask(sf).fillna(self.cont_med[c]).to_numpy(float)); flags.append(miss.to_numpy(np.float32));
                if c in self.special: flags.append(sf.to_numpy(np.float32))
            nums.extend(self.cscale.transform(np.column_stack(base)).T)
        for c in self.ord: nums.append(pd.to_numeric(df[c],errors='coerce').astype(float).fillna(self.ord_med[c]).to_numpy(float))
        raw_missing=df[SEM['features']].isna().sum(axis=1).to_numpy(float); sp=np.zeros(len(df))
        for c,v in self.special.items(): sp += pd.to_numeric(df[c],errors='coerce').astype(float).isin(v).to_numpy(float)
        nums.append(self.miss_scale.transform((raw_missing+sp).reshape(-1,1)).ravel())
        N=np.column_stack(nums).astype(np.float32)
        C=self.ohe.transform(df[self.nom].astype(object).where(df[self.nom].notna(),'__MISSING__').astype(str)).astype(np.float32)
        if flags: C=np.column_stack([C]+flags).astype(np.float32)
        X=np.ascontiguousarray(np.column_stack([N,C]).astype(np.float32))
        assert np.isfinite(X).all()
        return X
    def manifest(self): return {'n_num':self.n_num,'ohe_dim':int(sum(map(len,self.ohe.categories_))),'total_nominal_categories':{c:int(len(v)) for c,v in zip(self.nom,self.ohe.categories_)}}

def build_model(d_in,d_num,cfg):
    mc={k:v for k,v in cfg['model'].items()}
    return ModernNCA(d_in=d_in,d_num=d_num,d_out=1,**mc).to(DEVICE).float()

def _encode_eval(model,x):
    if model.num_embeddings is not None and model.d_num>0:
        xn,xc=x[:,:model.d_num],x[:,model.d_num:]
        xn=model.num_embeddings(xn).flatten(1)
        x=torch.cat([xn,xc],dim=-1)
    x=model.encoder(x)
    if model.n_blocks>0: x=model.post_encoder(x)
    return x

def _encode_support(model,Xs,chunk=2048):
    zs=[]
    with torch.no_grad():
        for s in range(0,len(Xs),chunk):
            zs.append(_encode_eval(model,Xs[s:s+chunk]))
    return torch.cat(zs,dim=0)

def pred(model,Xq,Xs,ys,mean,std,chunk=INFER):
    # Exact is_train=False ModernNCA prediction, with support embeddings cached once.
    # This changes only execution efficiency: same encoder, Euclidean distance, temperature,
    # softmax weights, and full-support weighted target average as official forward().
    model.eval(); out=[]
    with torch.no_grad():
        Zs=_encode_support(model,Xs)
        cy=ys.reshape(-1,1)
        for s in range(0,len(Xq),chunk):
            q=torch.as_tensor(Xq[s:s+chunk],device=DEVICE,dtype=torch.float32)
            Zq=_encode_eval(model,q)
            d=torch.cdist(Zq,Zs,p=2)/model.T
            weights=torch.softmax(-d,dim=-1)
            z=torch.mm(weights,cy).reshape(-1)
            out.append(z.float().cpu().numpy())
        del Zs
    ans=np.concatenate(out)*std+mean
    if not np.isfinite(ans).all():
        raise FloatingPointError('Non-finite ModernNCA predictions')
    return ans

def train_early(Xtr,ytr,Xv,yv,cfg,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    mean=float(np.mean(ytr)); std=float(np.std(ytr)); std=std if std>0 else 1.0
    yt=(ytr-mean)/std; yvt=(yv-mean)/std
    X=torch.as_tensor(Xtr,device=DEVICE,dtype=torch.float32); Y=torch.as_tensor(yt,device=DEVICE,dtype=torch.float32)
    model=build_model(X.shape[1],PREP_NNUM,cfg); opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['lr']),weight_decay=float(cfg['training']['weight_decay']))
    best=np.inf; best_epoch=-1; best_state=None; stale=0; n=len(Xtr); allidx=torch.arange(n,device=DEVICE)
    hist=[]
    for ep in range(MAXE):
        model.train(); perm=torch.randperm(n,device=DEVICE); losses=[]
        for bi in perm.split(BATCH):
            mask=torch.ones(n,dtype=torch.bool,device=DEVICE); mask[bi]=False
            pr=model(X[bi],Y[bi],X[mask],Y[mask],True).reshape(-1); loss=torch.nn.functional.mse_loss(pr,Y[bi])
            if not torch.isfinite(loss): raise FloatingPointError(f'Non-finite HPO train loss epoch={ep}')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        pv=pred(model,Xv,X,Y,mean,std); rm=float(np.sqrt(mean_squared_error(yv,pv)))
        if not np.isfinite(rm): raise FloatingPointError(f'Non-finite validation RMSE epoch={ep}')
        hist.append({'epoch':ep,'rmse':rm,'train_loss':float(np.mean(losses))})
        if rm < best-1e-10:
            best=rm; best_epoch=ep; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>PAT: break
    model.load_state_dict(best_state); return model,best,best_epoch,mean,std,hist,X,Y

def train_fixed(Xtr,ytr,cfg,seed,epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    mean=float(np.mean(ytr)); std=float(np.std(ytr)); std=std if std>0 else 1.0; yt=(ytr-mean)/std
    X=torch.as_tensor(Xtr,device=DEVICE,dtype=torch.float32); Y=torch.as_tensor(yt,device=DEVICE,dtype=torch.float32)
    model=build_model(X.shape[1],PREP_NNUM,cfg); opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['lr']),weight_decay=float(cfg['training']['weight_decay']))
    n=len(Xtr)
    for ep in range(int(epochs)):
        model.train(); perm=torch.randperm(n,device=DEVICE)
        for bi in perm.split(BATCH):
            mask=torch.ones(n,dtype=torch.bool,device=DEVICE); mask[bi]=False
            pr=model(X[bi],Y[bi],X[mask],Y[mask],True).reshape(-1); loss=torch.nn.functional.mse_loss(pr,Y[bi])
            if not torch.isfinite(loss): raise FloatingPointError(f'Non-finite refit train loss epoch={ep}')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model,mean,std,X,Y

def embed(model,X,chunk=1024):
    out=[]; model.eval()
    with torch.no_grad():
        for s in range(0,len(X),chunk):
            x=torch.as_tensor(X[s:s+chunk],device=DEVICE,dtype=torch.float32)
            if model.num_embeddings is not None and model.d_num>0:
                xn,xc=x[:,:model.d_num],x[:,model.d_num:]; xn=model.num_embeddings(xn).flatten(1); x=torch.cat([xn,xc],-1)
            x=model.encoder(x)
            if model.n_blocks>0: x=model.post_encoder(x)
            out.append(x.float().cpu().numpy())
    return np.concatenate(out)

def geometry(model,Xv,Xtr,yv,ytr):
    # Seed-42 model only; diagnostic, never used for selection.
    Zs=embed(model,Xtr); Zq=embed(model,Xv); rows=[]
    Zs_t=torch.as_tensor(Zs,device=DEVICE); yt=torch.as_tensor(ytr,device=DEVICE,dtype=torch.float32)
    for s in range(0,len(Zq),128):
        q=torch.as_tensor(Zq[s:s+128],device=DEVICE); d=torch.cdist(q,Zs_t); w=torch.softmax(-d/float(model.T),dim=1)
        eff=(1/(w.square().sum(1)+1e-12)).cpu().numpy(); avd=(w*d).sum(1).cpu().numpy(); k=min(32,Zs.shape[0]); ix=d.topk(k,largest=False).indices
        topy=yt[ix].cpu().numpy(); same=(w* (yt[None,:]==torch.as_tensor(yv[s:s+len(q)],device=DEVICE)[:,None]).float()).sum(1).cpu().numpy()
        for j in range(len(q)): rows.append((int(yv[s+j]),float(eff[j]),float(avd[j]),float(same[j]),float(np.mean(topy[j]==1)),float(np.mean(topy[j]==2)),float(np.mean(topy[j]==3)),float(np.mean(topy[j]==4))))
    z=pd.DataFrame(rows,columns=['y','effective_neighbors','weighted_distance','same_target_mass','top32_p1','top32_p2','top32_p3','top32_p4'])
    return z

trmask=TRAIN['__fold__'].to_numpy()!=FOLD; vamask=~trmask
OUTER_TR=TRAIN.loc[trmask].reset_index(drop=True); OUTER_VA=TRAIN.loc[vamask].reset_index(drop=True); ytr=OUTER_TR['life_satisfaction'].to_numpy(float); yva=OUTER_VA['life_satisfaction'].to_numpy(float)

try:
    if MODE=='runtime':
        prep=Prep().fit(OUTER_TR); Xtr=prep.transform(OUTER_TR); Xv=prep.transform(OUTER_VA.iloc[:min(2048,len(OUTER_VA))]); PREP_NNUM=prep.n_num
        cfg=json.load(open(ROOT/'official_default_v12.json'))['modernNCA']; t=time.time(); model,mean,std,Xs,Ys=train_fixed(Xtr,ytr,cfg,42,1); pv=pred(model,Xv,Xs,Ys,mean,std)
        rec={'status':'success','fold':FOLD,'train_rows':len(Xtr),'probe_rows':len(Xv),'features':Xtr.shape[1],'n_num':PREP_NNUM,'seconds':time.time()-t,'finite':bool(np.isfinite(pv).all()),'peak_cuda':int(torch.cuda.max_memory_allocated()),'prep':prep.manifest(),'talent_commit':CFG['talent_commit'],'modernnca_model_sha256':CFG['modernnca_model_sha256'],'config_manifest_sha256':CFG['config_manifest_sha256']}; json.dump(rec,open(OUT/'runtime_proof.json','w'),indent=2); print(json.dumps(rec,indent=2))
    elif MODE in ('screen','full'):
        # Inner HPO: one fixed stratified split inside outer train, seed 42 only.
        sss=StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=20260908+FOLD); ii,iv=next(sss.split(np.zeros(len(OUTER_TR)),ytr.astype(int)))
        ITR=OUTER_TR.iloc[ii].reset_index(drop=True); IVA=OUTER_TR.iloc[iv].reset_index(drop=True); yi=ytr[ii]; yiv=ytr[iv]
        prep_i=Prep().fit(ITR); Xi=prep_i.transform(ITR); Xiv=prep_i.transform(IVA); PREP_NNUM=prep_i.n_num
        hrows=[]; bestrec=None
        for cfg in CONFIGS:
            t=time.time(); model,rm,ep,mean,std,hist,Xs,Ys=train_early(Xi,yi,Xiv,yiv,cfg,42)
            rec={'config_id':cfg['config_id'],'rmse':rm,'best_epoch':ep,'seconds':time.time()-t,'peak_cuda':int(torch.cuda.max_memory_allocated())}; hrows.append(rec); pd.DataFrame(hrows).to_csv(OUT/'hpo_metrics.csv',index=False)
            json.dump(hist,open(OUT/f'history_config{cfg["config_id"]}.json','w'))
            if bestrec is None or (rm, cfg['config_id']) < (bestrec['rmse'],bestrec['config_id']): bestrec=rec
            del model,Xs,Ys; gc.collect(); torch.cuda.empty_cache()
        if len(hrows)!=8 or sorted(int(r['config_id']) for r in hrows)!=list(range(8)):
            raise RuntimeError(f'Fixed HPO incomplete: completed {[r["config_id"] for r in hrows]}')
        sel=CONFIGS[int(bestrec['config_id'])]; epochs=int(bestrec['best_epoch'])+1
        # Full outer-training refit, two fixed seeds, average outer predictions.
        prep=Prep().fit(OUTER_TR); Xtr=prep.transform(OUTER_TR); Xv=prep.transform(OUTER_VA); PREP_NNUM=prep.n_num
        preds=[]; states=[]; geom=None
        for seed in (42,2026):
            model,mean,std,Xs,Ys=train_fixed(Xtr,ytr,sel,seed,epochs); p=pred(model,Xv,Xs,Ys,mean,std); preds.append(p)
            torch.save({'state':model.state_dict(),'mean':mean,'std':std,'config_id':int(bestrec['config_id']),'epochs':epochs,'n_num':PREP_NNUM},OUT/f'model_seed{seed}.pt')
            if seed==42:
                del Xs,Ys; torch.cuda.empty_cache(); geom=geometry(model,Xv,Xtr,yva,ytr); geom.to_csv(OUT/'geometry_rows.csv',index=False)
            else:
                del Xs,Ys
            del model; gc.collect(); torch.cuda.empty_cache()
        po=np.mean(np.vstack(preds),axis=0)
        if not np.isfinite(po).all(): raise FloatingPointError('Non-finite outer predictions')
        np.savez_compressed(OUT/'outer_predictions.npz',id=OUTER_VA.id.to_numpy(),y=yva,pred=po,fold=np.full(len(po),FOLD))
        meta={'status':'success','fold':FOLD,'selected_config_id':int(bestrec['config_id']),'selected_inner_rmse':float(bestrec['rmse']),'selected_best_epoch':int(bestrec['best_epoch']),'refit_epochs':epochs,'outer_rows':len(OUTER_TR),'valid_rows':len(OUTER_VA),'features':Xtr.shape[1],'n_num':PREP_NNUM,'prep':prep.manifest(),'talent_commit':CFG['talent_commit'],'modernnca_model_sha256':CFG['modernnca_model_sha256'],'config_manifest_sha256':CFG['config_manifest_sha256']}; json.dump(meta,open(OUT/'fold_summary.json','w'),indent=2)
        if MODE=='full':
            Xt=prep.transform(TEST); tp=[]
            for seed in (42,2026):
                ck=torch.load(OUT/f'model_seed{seed}.pt',map_location=DEVICE); model=build_model(Xtr.shape[1],PREP_NNUM,sel); model.load_state_dict(ck['state']); Xs=torch.as_tensor(Xtr,device=DEVICE,dtype=torch.float32); Ys=torch.as_tensor((ytr-ck['mean'])/ck['std'],device=DEVICE,dtype=torch.float32); tp.append(pred(model,Xt,Xs,Ys,ck['mean'],ck['std'])); del model,Xs,Ys; torch.cuda.empty_cache()
            _tp=np.mean(np.vstack(tp),axis=0)
            if not np.isfinite(_tp).all(): raise FloatingPointError('Non-finite full-fold test predictions')
            np.savez_compressed(OUT/'test_predictions.npz',id=TEST.id.to_numpy(),pred=_tp,fold=np.full(len(TEST),FOLD))
        print(json.dumps(meta,indent=2))
    elif MODE=='resume_test':
        src=ROOT/f'screen_fold{FOLD}'; meta=json.load(open(src/'fold_summary.json')); sel=CONFIGS[int(meta['selected_config_id'])]
        prep=Prep().fit(OUTER_TR); Xtr=prep.transform(OUTER_TR); Xt=prep.transform(TEST); PREP_NNUM=prep.n_num; tp=[]
        for seed in (42,2026):
            ck=torch.load(src/f'model_seed{seed}.pt',map_location=DEVICE); model=build_model(Xtr.shape[1],PREP_NNUM,sel); model.load_state_dict(ck['state']); Xs=torch.as_tensor(Xtr,device=DEVICE,dtype=torch.float32); Ys=torch.as_tensor((ytr-ck['mean'])/ck['std'],device=DEVICE,dtype=torch.float32); tp.append(pred(model,Xt,Xs,Ys,ck['mean'],ck['std'])); del model,Xs,Ys; torch.cuda.empty_cache()
        _tp=np.mean(np.vstack(tp),axis=0)
        if not np.isfinite(_tp).all(): raise FloatingPointError('Non-finite resume-test predictions')
        np.savez_compressed(OUT/'test_predictions.npz',id=TEST.id.to_numpy(),pred=_tp,fold=np.full(len(TEST),FOLD)); json.dump({'status':'success','fold':FOLD},open(OUT/'resume_summary.json','w'))
except Exception as e:
    fail={'status':'failed','mode':MODE,'fold':FOLD,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'cuda_allocated':int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,'cuda_reserved':int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else None,'cuda_peak':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None}
    json.dump(fail,open(OUT/'failure.json','w'),indent=2); print(json.dumps(fail,indent=2),file=sys.stderr); raise
