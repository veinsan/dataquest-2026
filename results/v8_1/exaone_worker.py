
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

import numpy as np


def sha256_file(path,chunk=2**20):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        while block:=handle.read(chunk):digest.update(block)
    return digest.hexdigest()


def positive_probability(model,X):
    probabilities=np.asarray(model.predict_proba(X),dtype=np.float64)
    classes=np.asarray(model.classes_)
    match=np.flatnonzero(classes==1)
    assert len(match)==1,classes
    result=probabilities[:,int(match[0])]
    assert np.isfinite(result).all() and np.all((0<=result)&(result<=1))
    return result


parser=argparse.ArgumentParser()
parser.add_argument("--mode",choices=["preflight","run"],required=True)
parser.add_argument("--repo",required=True)
parser.add_argument("--revision",required=True)
parser.add_argument("--checkpoint",required=True)
parser.add_argument("--checkpoint-sha256",required=True)
parser.add_argument("--runtime-revision",required=True)
parser.add_argument("--cache",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--input")
parser.add_argument("--ensemble",type=int,default=1)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--max-vram-bytes",type=int,required=True)
args=parser.parse_args()

os.environ["HF_HOME"]=args.cache
os.environ.pop("EXAONETABULAR_CLASSIFIER_WEIGHTS",None)
import torch
from huggingface_hub import hf_hub_download
from exaonetabular import EXAONETabularClassifier

assert torch.cuda.is_available(),"EXAONE requires a Kaggle GPU"
torch.cuda.set_device(0)
checkpoint=hf_hub_download(repo_id=args.repo,filename=args.checkpoint,revision=args.revision)
checkpoint_sha=sha256_file(checkpoint)
assert checkpoint_sha==args.checkpoint_sha256,(checkpoint_sha,args.checkpoint_sha256)

distribution=metadata.distribution("exaonetabular")
direct=json.loads(distribution.read_text("direct_url.json") or "{}")
installed_commit=direct.get("vcs_info",{}).get("commit_id")
assert installed_commit==args.runtime_revision,(installed_commit,args.runtime_revision)

started=time.time()
model=EXAONETabularClassifier.from_pretrained(
    weights=checkpoint,device="cuda:0",ensemble_count=args.ensemble,seed=args.seed,
    revision=args.revision,max_vram_bytes=args.max_vram_bytes,
)
common={
    "package_version":distribution.version,
    "runtime_commit":installed_commit,
    "hf_revision":args.revision,
    "checkpoint":args.checkpoint,
    "checkpoint_path":str(checkpoint),
    "checkpoint_sha256":checkpoint_sha,
    "torch":torch.__version__,
    "gpu":torch.cuda.get_device_name(0),
}
if args.mode=="preflight":
    common["runtime_sec"]=time.time()-started
    Path(args.output).write_text(json.dumps(common,indent=2))
else:
    data=np.load(args.input,allow_pickle=False)
    X_train=np.asarray(data["X_train"],dtype=np.float32)
    y_train=np.asarray(data["y_train"],dtype=np.int64)
    X_valid=np.asarray(data["X_valid"],dtype=np.float32)
    X_test=np.asarray(data["X_test"],dtype=np.float32)
    assert X_train.ndim==X_valid.ndim==X_test.ndim==2
    assert X_train.shape[1]==X_valid.shape[1]==X_test.shape[1]
    assert set(np.unique(y_train).tolist())=={0,1}
    assert np.all(np.isfinite(X_train)|np.isnan(X_train))
    model.fit(X_train,y_train)
    p_valid=positive_probability(model,X_valid)
    p_test=positive_probability(model,X_test) if len(X_test) else np.empty(0,dtype=np.float64)
    selected=np.asarray(getattr(model,"selected_feature_indices_",[]),dtype=np.int64)
    np.savez(args.output,p_valid=p_valid,p_test=p_test,selected_feature_indices=selected,
             runtime_sec=np.array([time.time()-started],dtype=np.float64))
