"""Deterministic variable-size synthetic prior generation and HDF5 replay.

No synthetic task is a real clinical cohort. The file has an explicit
completion marker so a partial 200k run cannot masquerade as completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .maternal_scm import sample_task

DEFAULT_CONTEXT_BUDGETS = (16, 32, 64, 128, 256, 512)


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_replay(
    path, *, num_tables=200000, seed=20261107, prior="mixture",
    min_rows=32, max_rows=576, min_features=5, max_features=16,
    maternal_probability=.7, flush_every=250, limit=None,
):
    if not num_tables>0 or not 8<=min_rows<=max_rows or not 2<=min_features<=max_features<=32:
        raise ValueError("Invalid synthetic task dimensions")
    if not 0<=maternal_probability<=1:
        raise ValueError("maternal_probability must be between zero and one")
    if prior not in ("maternal","generic","mixture"):
        raise ValueError("Invalid prior")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    settings={
        "num_tables":int(num_tables),"seed":int(seed),"prior":prior,
        "min_rows":int(min_rows),"max_rows":int(max_rows),
        "min_features":int(min_features),"max_features":int(max_features),
        "maternal_probability":float(maternal_probability),
        "source":"synthetic","schema_version":2,
    }
    new=not path.exists()
    with h5py.File(path,"a") as f:
        if new:
            f.attrs["settings"]=json.dumps(settings,sort_keys=True)
            f.attrs["completed"]=0
            f.create_dataset("X",(num_tables,),dtype=h5py.vlen_dtype(np.dtype("float32")))
            f.create_dataset("y",(num_tables,),dtype=h5py.vlen_dtype(np.dtype("int64")))
            for k in ("n_rows","n_features","n_classes"):
                f.create_dataset(k,(num_tables,),dtype="i4")
            f.create_dataset("is_maternal",(num_tables,),dtype="i1")
        elif json.loads(f.attrs["settings"])!=settings:
            raise ValueError("Existing replay settings do not match; refusing unsafe resume")
        first=int(f.attrs["completed"])
        end=num_tables if limit is None else min(num_tables,first+int(limit))
        for i in range(first,end):
            rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
            rows=int(rng.integers(min_rows,max_rows+1))
            features=int(rng.integers(min_features,max_features+1))
            classes=int(rng.choice([2,3]))
            x,y,meta=sample_task(rows,features,classes,prior,rng=rng,
                                  maternal_probability=maternal_probability)
            f["X"][i]=x.ravel()
            f["y"][i]=y.ravel()
            f["n_rows"][i]=rows
            f["n_features"][i]=features
            f["n_classes"][i]=classes
            f["is_maternal"][i]=int(meta["family"]!="generic")
            if (i+1)%flush_every==0:
                f.attrs["completed"]=i+1
                f.flush()
        f.attrs["completed"]=end
        f.flush()
    return {**settings,"completed":end,"fully_generated":end==num_tables,
            "path":str(path),"sha256":file_sha256(path) if end==num_tables else None}


class HDF5TaskReplay:
    """Stream variable-size tasks without materializing the whole archive."""
    def __init__(self,path,require_complete=True):
        self.path=str(path)
        self.f=h5py.File(self.path,"r")
        self.settings=json.loads(self.f.attrs["settings"])
        self.completed=int(self.f.attrs["completed"])
        if require_complete and self.completed!=self.settings["num_tables"]:
            self.close()
            raise ValueError("Partial prior dump; complete generation before pretraining")
        if self.completed<=0:
            self.close()
            raise ValueError("Prior archive contains no completed tasks")
        self.n_rows=self.f["n_rows"][:self.completed]
        classes=self.f["n_classes"][:self.completed]
        self.by_class={c:np.where(classes==c)[0] for c in (2,3)}

    def __len__(self):
        return self.completed

    def read(self,index):
        if not 0<=index<self.completed:
            raise IndexError(index)
        n=int(self.f["n_rows"][index]);d=int(self.f["n_features"][index])
        return (np.asarray(self.f["X"][index],dtype=np.float32).reshape(n,d),
                np.asarray(self.f["y"][index],dtype=np.int64),
                int(self.f["n_classes"][index]))

    def sample_batch(self,rng,batch_size=4,context_budgets=DEFAULT_CONTEXT_BUDGETS,
                     feature_range=(5,16),query_rows=8):
        # Choose budget first. Sampling arbitrary row counts first would
        # suppress large contexts almost completely at useful batch sizes.
        eligible=[(c,b) for c,ids in self.by_class.items() for b in context_budgets
                  if np.any(self.n_rows[ids]>=b+query_rows)]
        if not eligible:
            raise ValueError("No suitable context budget in prior archive")
        n_classes,n_ctx=eligible[int(rng.integers(len(eligible)))]
        ids=self.by_class[n_classes]
        ids=ids[self.n_rows[ids]>=n_ctx+query_rows]
        indices=rng.choice(ids,size=batch_size,replace=True)
        tables=[self.read(int(i)) for i in indices]
        n_q=query_rows
        min_f=min(t[0].shape[1] for t in tables)
        low=min(feature_range[0],min_f)
        high=min(feature_range[1],min_f)
        n_features=int(rng.integers(low,high+1))
        xs=[];ys=[]
        for x,y,_ in tables:
            row_order=rng.permutation(len(y))[:n_ctx+n_q]
            col_order=rng.permutation(x.shape[1])[:n_features]
            xs.append(x[row_order][:,col_order])
            ys.append(y[row_order])
        return np.stack(xs),np.stack(ys),int(n_ctx),int(n_classes)

    def close(self):
        if self.f:
            self.f.close()
            self.f=None

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def main():
    p=argparse.ArgumentParser(description="Generate or resume variable-shape HDF5 synthetic prior")
    p.add_argument("--path",default="artifacts/priors_h5/maternal_mixture_200k.h5")
    p.add_argument("--num-tables",type=int,default=200000)
    p.add_argument("--seed",type=int,default=20261107)
    p.add_argument("--prior",choices=["maternal","generic","mixture"],default="mixture")
    p.add_argument("--min-rows",type=int,default=32)
    p.add_argument("--max-rows",type=int,default=576)
    p.add_argument("--min-features",type=int,default=5)
    p.add_argument("--max-features",type=int,default=16)
    p.add_argument("--limit",type=int,default=None,
                   help="Maximum new tasks this invocation; result remains marked partial")
    a=p.parse_args()
    print(json.dumps(create_replay(a.path,num_tables=a.num_tables,seed=a.seed,
          prior=a.prior,min_rows=a.min_rows,max_rows=a.max_rows,
          min_features=a.min_features,max_features=a.max_features,
          limit=a.limit),indent=2))


if __name__=="__main__":
    main()
