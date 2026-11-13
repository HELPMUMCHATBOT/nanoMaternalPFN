"""Strict context-only model choice, tuned classical baselines and PFN experts."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class ClassSafeXGBoost(ClassifierMixin, BaseEstimator):
    """Explicitly encode arbitrary integer clinical target labels for XGB.

    SearchCV calls fit only on labeled context folds; original class identities
    are restored via classes_ for downstream alignment.
    """
    def __init__(self,n_estimators=200,max_depth=3,learning_rate=0.1,
                 min_child_weight=1,reg_lambda=1,random_state=20261113):
        self.n_estimators=n_estimators;self.max_depth=max_depth
        self.learning_rate=learning_rate;self.min_child_weight=min_child_weight
        self.reg_lambda=reg_lambda;self.random_state=random_state

    def get_params(self,deep=True):
        return {k:getattr(self,k) for k in ('n_estimators','max_depth','learning_rate',
             'min_child_weight','reg_lambda','random_state')}

    def set_params(self,**params):
        for k,v in params.items():
            if k not in self.get_params():raise ValueError('Unexpected XGB search parameter '+k)
            setattr(self,k,v)
        return self

    def fit(self,x,y):
        from xgboost import XGBClassifier
        import numpy as np
        self.classes_,encoded=np.unique(y,return_inverse=True)
        if len(self.classes_)<2:raise ValueError('XGBoost requires two observed classes')
        self.model_=XGBClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,
             learning_rate=self.learning_rate,min_child_weight=self.min_child_weight,
             reg_lambda=self.reg_lambda,random_state=self.random_state,n_jobs=1,
             tree_method='hist',objective='binary:logistic' if len(self.classes_)==2 else 'multi:softprob',
             eval_metric='logloss' if len(self.classes_)==2 else 'mlogloss',verbosity=0)
        self.model_.fit(x,encoded)
        return self

    def predict_proba(self,x):return self.model_.predict_proba(x)
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(axis=1)]


CLASSICAL=('logistic','random_forest','xgboost','lightgbm')
SINGLE=('tiny_generic','tiny_mixture','base_generic','base_mixture',
        'v1_tiny_generic','v1_tiny_mixture','v1_base_generic','v1_base_mixture')
EXPERTS=('v1_tiny_mixture','tiny_mixture','tiny_generic')
EXPERT_RULES=('uniform','context_cv_softmax','context_cv_winner','fixed_v2_mixture')


def checkpoint(name,v1_dir,v2_dir):
    if name not in SINGLE:raise ValueError('Unknown PFN checkpoint name: '+name)
    path=Path(v1_dir if name.startswith('v1_') else v2_dir)
    return path/((name[3:] if name.startswith('v1_') else name)+'.best.inference.pt')


def build_classical(name,seed=20261113):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if name=='logistic':
        from sklearn.linear_model import LogisticRegression
        return Pipeline([('imp',SimpleImputer(add_indicator=True)),('scale',StandardScaler()),
                         ('est',LogisticRegression(max_iter=1500,random_state=seed))]),{
            'est__C':[.01,.1,1,10], 'est__class_weight':[None,'balanced']}
    if name=='random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return Pipeline([('imp',SimpleImputer(add_indicator=True)),
             ('est',RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=1))]),{
                 'est__max_depth':[3,6,None], 'est__min_samples_leaf':[1,3,8],
                 'est__max_features':['sqrt',.75]}
    if name=='xgboost':
        return Pipeline([('imp',SimpleImputer(add_indicator=True)),
                ('est',ClassSafeXGBoost(random_state=seed))]),{
            'est__max_depth':[2,3,5], 'est__learning_rate':[.03,.1],
            'est__min_child_weight':[1,5], 'est__reg_lambda':[1,5]}
    if name=='lightgbm':
        from lightgbm import LGBMClassifier
        return Pipeline([('imp',SimpleImputer(add_indicator=True)),
             ('est',LGBMClassifier(n_estimators=200,verbosity=-1,random_state=seed,n_jobs=1))]),{
                 'est__num_leaves':[7,15], 'est__min_child_samples':[2,5,20],
                 'est__learning_rate':[.03,.1]}
    raise ValueError('Not a classical baseline: '+name)


def tuned_baseline(name,x,y,seed=20261113,n_iter=8):
    from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
    model,space=build_classical(name,seed)
    unique,counts=np.unique(y,return_counts=True)
    if len(unique)<2 or min(counts)<2:raise ValueError('Insufficient class labels for inner CV tuning')
    folds=min(3,int(min(counts)))
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    # Only labeled context enters the search; frozen test observations are never passed to fit.
    search=RandomizedSearchCV(model,space,n_iter=n_iter,cv=cv,scoring='neg_log_loss',
                              random_state=seed,n_jobs=1,refit=True,error_score='raise')
    search.fit(x,y)
    return search.best_estimator_,{'inner_cv_folds':folds,'best_params':search.best_params_,
                                    'inner_cv_neg_log_loss':float(search.best_score_)}


class ControlledFeatureAdapter:
    """Maintain context-only imputation at transfer for ablated checkpoints."""
    def __init__(self,base):
        self.base=base
    def _transform(self,x):
        x=np.asarray(x,dtype=np.float32)
        return np.where(np.isnan(x),self.context_means_[None,:],x).astype(np.float32)
    def fit(self,x,y):
        with np.errstate(all='ignore'):
            sums=np.nansum(x,axis=0)
            n=np.sum(~np.isnan(x),axis=0)
            self.context_means_=np.divide(sums,n,out=np.zeros_like(sums,dtype=float),where=n>0).astype(np.float32)
        self.base.fit(self._transform(x),y)
        self.classes_=self.base.classes_
        return self
    def predict_proba(self,x):return self.base.predict_proba(self._transform(x))


def make_expert(name,v1_dir,v2_dir,override_path=None):
    from nanomaternalpfn.classifier import NanoMaternalPFNClassifier
    path=Path(override_path) if override_path is not None else checkpoint(name,v1_dir,v2_dir)
    if not path.is_file():raise FileNotFoundError('Actual research checkpoint unavailable: '+str(path))
    base=NanoMaternalPFNClassifier(path,device='cpu',query_batch_size=16)
    if override_path is not None:
        import torch
        meta=torch.load(path,map_location='cpu',weights_only=True)
        if meta.get('feature_mode')=='imputed_no_indicator':
            return ControlledFeatureAdapter(base)
    return base


class ContextOnlyExpertEnsemble:
    def __init__(self,expert_names,v1_dir,v2_dir,rule='context_cv_softmax',seed=20261113):
        if rule not in EXPERT_RULES:raise ValueError('Unrecognized expert selection rule')
        self.expert_names=tuple(expert_names)
        if len(self.expert_names)<2:raise ValueError('At least two distinct experts needed')
        self.v1_dir=v1_dir;self.v2_dir=v2_dir;self.rule=rule;self.seed=seed
        self.experts_=[]
    def fit(self,x,y):
        from sklearn.model_selection import StratifiedKFold
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y)
        self.classes_=np.unique(y)
        self.experts_=[make_expert(n,self.v1_dir,self.v2_dir) for n in self.expert_names]
        self.weights_=np.full(len(self.experts_),1/len(self.experts_))
        self.cv_nll_=[];self.cv_status_='not_used'
        if self.rule=='fixed_v2_mixture':
            if 'tiny_mixture' not in self.expert_names:raise ValueError('fixed expert not in pool')
            self.weights_=np.array([float(n=='tiny_mixture') for n in self.expert_names])
        elif self.rule!='uniform':
            counts=np.unique(y,return_counts=True)[1]
            folds=min(4,int(counts.min()))
            if folds>=2:
                cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=self.seed)
                oof=np.full((len(self.experts_),len(y),len(self.classes_)),np.nan)
                for train,valid in cv.split(x,y):
                    for i,expert in enumerate(self.experts_):
                        expert.fit(x[train],y[train])
                        p=expert.predict_proba(x[valid])
                        for j,c in enumerate(expert.classes_):
                            oof[i,valid,np.where(self.classes_==c)[0][0]]=p[:,j]
                if np.isfinite(oof).all():
                    idx=np.searchsorted(self.classes_,y)
                    nll=-np.log(np.clip(oof[:,np.arange(len(y)),idx],1e-7,1)).mean(axis=1)
                    self.cv_nll_=nll.tolist();self.cv_status_='context_only_cross_fitted'
                    if self.rule=='context_cv_winner':
                        self.weights_=np.eye(len(nll))[np.argmin(nll)]
                    else:
                        logits=-6*(nll-nll.min());w=np.exp(logits-logits.max());self.weights_=w/w.sum()
            else:self.cv_status_='insufficient_context_labels_fallback_uniform'
        for expert in self.experts_:expert.fit(x,y)
        self.diagnostics_={'rule':self.rule,'expert_names':list(self.expert_names),
                           'weights':self.weights_.tolist(),'context_cv_nll':self.cv_nll_,
                           'context_cv_status':self.cv_status_}
        return self
    def predict_proba(self,x):
        probs=[]
        for expert in self.experts_:
            out=np.zeros((len(x),len(self.classes_)))
            p=expert.predict_proba(x)
            for j,c in enumerate(expert.classes_):out[:,np.where(self.classes_==c)[0][0]]=p[:,j]
            probs.append(out)
        result=np.tensordot(self.weights_,np.asarray(probs),axes=1)
        return result/result.sum(axis=1,keepdims=True)
