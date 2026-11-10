import numpy as np
import torch
from nanomaternalpfn.model import NanoMaternalPFN, ModelConfig, MissingnessFeatureEncoder
from nanomaternalpfn.priors.maternal_scm import sample_task
from nanomaternalpfn.priors.dump_hdf5 import dump
from nanomaternalpfn.data.deduplicate import profile_groups
from nanomaternalpfn.data.freeze_splits import freeze
from nanomaternalpfn.evaluation.calibration import multiclass_brier, ece
from nanomaternalpfn.train import pretrain
from nanomaternalpfn.classifier import NanoMaternalPFNClassifier

torch.set_num_threads(1)
CFG=ModelConfig(24,4,2,48,3)

def test_missingness_context_only():
    x=torch.tensor([[[1.,float('nan')],[3.,4.],[999.,8.]]])
    a=MissingnessFeatureEncoder.encode_inputs(x,2)
    b=MissingnessFeatureEncoder.encode_inputs(torch.cat((x[:,:2],torch.tensor([[[-999.,8.]]])),dim=1),2)
    assert torch.allclose(a[:,:2],b[:,:2])
    assert torch.isfinite(a).all()
    assert a[0,0,1,1]==1 and a[0,0,1,0]==0

def test_query_independence_and_permutation():
    torch.manual_seed(7)
    m=NanoMaternalPFN(CFG).eval()
    x=torch.randn(1,7,5)
    x[0,1,2]=float('nan')
    y=torch.tensor([[0,1,2,0]])
    with torch.no_grad():
        logits=m(x,y)
        alone=m(x[:,:5],y)
        context_perm=torch.tensor([2,0,3,1])
        shuffled=m(torch.cat([x[:,context_perm],x[:,4:]],dim=1),y[:,context_perm])
        feature_perm=torch.tensor([3,0,4,1,2])
        features=m(x[:,:,feature_perm],y)
    assert torch.allclose(logits[:,:1],alone,atol=1e-5)
    assert torch.allclose(logits,shuffled,atol=1e-5)
    assert torch.allclose(logits,features,atol=1e-5)

def test_synthetic_and_training(tmp_path):
    x,y,meta=sample_task(32,8,3,'maternal',seed=12)
    assert x.shape==(32,8) and y.min()==0 and y.max()==2
    assert meta['source']=='synthetic'
    assert dump(tmp_path/'prior.h5',3,32,8,3,seed=12).exists()
    ck=tmp_path/'tiny.pt'
    result=pretrain(ck,CFG,steps=2,n_rows=24,n_ctx=12,n_features=8,batch_size=2)
    assert result['parameter_count']>0 and np.isfinite(result['last_loss'])
    clf=NanoMaternalPFNClassifier(ck).fit(x[:12],y[:12])
    p=clf.predict_proba(x[12:15])
    assert p.shape==(3,3) and np.allclose(p.sum(axis=1),1)

def test_freeze_no_cross_split_duplicates(tmp_path):
    rng=np.random.default_rng(7)
    x=rng.normal(size=(80,4));x[60:]=x[:20]
    y=rng.integers(0,2,size=80)
    manifest=freeze(x,y,tmp_path/'frozen.json',budgets=(16,32),n_seeds=2)
    data=np.load(tmp_path/'frozen.npz')
    groups=profile_groups(x)
    assert not set(groups[data['train_idx']]) & set(groups[data['test_idx']])
    assert manifest['audit']['n_duplicate_rows']>=20

def test_calibration_sanity():
    p=np.array([[.8,.2],[.4,.6]])
    assert multiclass_brier([0,1],p)>=0
    assert 0<=ece([0,1],p)<=1

def test_binary_class_head():
    m=NanoMaternalPFN(CFG)
    x=torch.randn(1,6,4)
    y=torch.tensor([[0,1,0,1]])
    assert m(x,y,n_classes=2).shape==(1,2,2)
