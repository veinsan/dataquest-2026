#!/usr/bin/env python
import os,sys,json,time,gc,traceback,argparse,random,hashlib,shutil,resource
from pathlib import Path
import numpy as np,pandas as pd,torch
from tabldm import TabLDMRegressor

ap=argparse.ArgumentParser()
ap.add_argument('--out',required=True)
ap.add_argument('--fold',type=int,required=True)
ap.add_argument('--stage',required=True)
ap.add_argument('--n-estimators',type=int,required=True)
ap.add_argument('--query-chunk-rows',type=int,required=True)
ap.add_argument('--predict-test',action='store_true')
ap.add_argument('--runtime-probe',action='store_true')
ap.add_argument('--safe-mode',action='store_true')
a=ap.parse_args()

OUT=Path(a.out); job=OUT/f'{a.stage}_fold{a.fold}'; job.mkdir(parents=True,exist_ok=True)
SEED=14000+a.fold*101+a.n_estimators*17
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(int(os.environ.get('TABLDM_CPU_THREADS','4')))

cfg=json.load(open(OUT/'worker_contract.json')); sem=json.load(open(OUT/'semantic_contract_v14.json'))
features=sem['features']; nominal=sem['nominal_categorical']

# Do not materialize the test table in screen/runtime workers.
train_cols=list(dict.fromkeys([cfg['id_col'],cfg['target'],'__fold__']+features))
tr=pd.read_csv(OUT/'train_unique_v14.csv',usecols=train_cols,low_memory=False)
folds=tr['__fold__'].astype(int).to_numpy()
y=tr[cfg['target']].astype(float).to_numpy()
train_mask=folds!=a.fold; valid_mask=folds==a.fold

def view(df):
    X=df[features].copy()
    for c in nominal:
        X[c]=X[c].astype('category')
    return X

A=view(tr.loc[train_mask])
valid_raw=tr.loc[valid_mask,[cfg['id_col']]+features].reset_index(drop=True)
if a.runtime_probe:
    valid_raw=valid_raw.iloc[:min(len(valid_raw),int(cfg['runtime_probe_rows']))].copy()

te=None
if a.predict_test:
    test_cols=list(dict.fromkeys([cfg['id_col']]+features))
    te=pd.read_csv(OUT/'test_unique_v14.csv',usecols=test_cols,low_memory=False)

off=Path(cfg['offload_root'])/f'{a.stage}_fold{a.fold}'
shutil.rmtree(off,ignore_errors=True); off.mkdir(parents=True,exist_ok=True)

# Conservative mode changes only inference storage/batching policy, never the model/data/seed.
inference_config=None
if a.safe_mode:
    common={
        'safety_factor':0.60,
        'offload':'disk',
        'disk_offload_dir':str(off),
        'disk_min_free_mb':1024.0,
        'disk_flush_mb':1024.0,
        'disk_cleanup':True,
        'use_async':False,
        'async_depth':1,
        'max_pinned_memory_mb':512.0,
    }
    inference_config={
        'COL_CONFIG':dict(common),
        'ROW_CONFIG':dict(common),
        'ICL_CONFIG':dict(common),
    }

def peak_rss_gib():
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024)

def chunked_predict(model,raw_df,chunk_rows,label):
    n=len(raw_df)
    if n==0:
        return np.empty(0,dtype=np.float64),0
    out=[]; calls=0
    for lo in range(0,n,chunk_rows):
        hi=min(n,lo+chunk_rows)
        Xq=view(raw_df.iloc[lo:hi])
        q=np.asarray(model.predict(Xq),dtype=np.float64).reshape(-1)
        if len(q)!=(hi-lo) or not np.isfinite(q).all():
            raise RuntimeError(f'Invalid {label} predictions for rows [{lo}:{hi})')
        out.append(q); calls+=1
        del Xq,q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if calls%4==0:
            gc.collect()
    p=np.concatenate(out)
    if len(p)!=n or not np.isfinite(p).all():
        raise RuntimeError(f'Invalid concatenated {label} predictions')
    return p,calls

started=time.time(); model=None
try:
    model=TabLDMRegressor(
        n_estimators=a.n_estimators,
        batch_size=cfg['batch_size'],
        kv_cache=False,
        model_path=cfg['checkpoint'],allow_auto_download=False,
        device='cuda',use_amp=cfg['use_amp'],use_fa3=False,
        offload_mode=cfg['offload_mode'],disk_offload_dir=str(off),
        random_state=SEED,n_jobs=int(cfg['n_jobs']),verbose=bool(a.runtime_probe),
        inference_config=inference_config,
    )
    model.fit(A,y[train_mask])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    p,nv_calls=chunked_predict(model,valid_raw,int(a.query_chunk_rows),'validation')
    np.savez_compressed(
        job/'valid_predictions.npz',
        id=valid_raw[cfg['id_col']].to_numpy(),
        pred=p
    )

    nt_calls=0
    if a.predict_test:
        pt,nt_calls=chunked_predict(model,te,int(a.query_chunk_rows),'test')
        np.savez_compressed(
            job/'test_predictions.npz',
            id=te[cfg['id_col']].to_numpy(),
            pred=pt
        )

    total_gib=float(torch.cuda.get_device_properties(0).total_memory/2**30)
    max_alloc=float(torch.cuda.max_memory_allocated()/2**30)
    rec={
        'status':'success','stage':a.stage,'fold':a.fold,'seed':SEED,
        'n_estimators':a.n_estimators,'train_rows':int(train_mask.sum()),
        'valid_rows_predicted':int(len(valid_raw)),
        'test_rows_predicted':int(len(te) if a.predict_test else 0),
        'query_chunk_rows':int(a.query_chunk_rows),
        'valid_predict_calls':int(nv_calls),'test_predict_calls':int(nt_calls),
        'safe_mode':bool(a.safe_mode),
        'features':int(len(features)),'nominal_features':int(len(nominal)),
        'runtime_sec':float(time.time()-started),
        'cuda_name':torch.cuda.get_device_name(0),
        'cuda_total_gib':total_gib,
        'max_cuda_alloc_gib':max_alloc,
        'cuda_peak_fraction':float(max_alloc/total_gib),
        'peak_rss_gib':peak_rss_gib(),
        'offload_dir':str(off)
    }
    json.dump(rec,open(job/'runtime.json','w'),indent=2)
    print(json.dumps(rec))
except Exception as e:
    rec={
        'status':'failure','stage':a.stage,'fold':a.fold,
        'n_estimators':a.n_estimators,'query_chunk_rows':int(a.query_chunk_rows),
        'safe_mode':bool(a.safe_mode),
        'runtime_sec':float(time.time()-started),
        'peak_rss_gib':peak_rss_gib(),
        'error':repr(e),'traceback':traceback.format_exc()
    }
    json.dump(rec,open(job/'failure.json','w'),indent=2)
    print(json.dumps(rec),file=sys.stderr)
    raise
finally:
    if model is not None: del model
    del A,valid_raw,tr
    if te is not None: del te
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
