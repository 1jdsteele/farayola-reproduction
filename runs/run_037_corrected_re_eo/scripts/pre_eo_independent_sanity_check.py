"""Noncanonical standalone pre-EO check, no AIF360 dependency.

Uses the analytic Kamiran-Calders race/label reweighting formula directly.
This DOES NOT run EO, and its results are NOT the official Run 037.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
D = pd.read_csv(ROOT/'data/compas_farayola_numeric.csv')
REF = pd.read_csv(ROOT/'provenance/run_010_reweighing_per_seed_results_A.csv').set_index('seed')
X = D.drop(columns='two_year_recid')
y = D.two_year_recid
rows=[]
for seed in range(101,111):
    xtr, xte, ytr, yte = train_test_split(X,y,test_size=.3,random_state=seed,shuffle=True,stratify=X.race)
    g, yy = xtr.race.to_numpy(), ytr.to_numpy()
    w = np.zeros(len(yy), dtype=float)
    for group in (0,1):
        for label in (0,1):
            mask = (g==group) & (yy==label)
            w[mask] = (g==group).sum() * (yy==label).sum() / (len(yy)*mask.sum())
    base = LogisticRegression(solver='lbfgs',max_iter=1000).fit(xtr,ytr)
    weighted = LogisticRegression(solver='lbfgs',max_iter=1000).fit(xtr,ytr,sample_weight=w)
    before,after=base.predict(xte),weighted.predict(xte)
    cm = confusion_matrix(yte,after,labels=[0,1]).ravel()
    refcm = REF.loc[seed,['tn','fp','fn','tp']].to_numpy(int)
    rows.append({'seed':seed,'pre_eo_predictions_changed':int(np.sum(before!=after)),
                 'weighted_accuracy':float(np.mean(after==yte)),
                 're010_cm_matches':bool(np.array_equal(cm,refcm)),
                 'weight_min':float(w.min()), 'weight_max':float(w.max())})
r=pd.DataFrame(rows)
out=ROOT/'metadata/pre_eo_independent_sanity_check.csv'
out.parent.mkdir(exist_ok=True)
r.to_csv(out,index=False)
print('Independent noncanonical pre-EO check — no AIF360, no EO')
print('Python',sys.version.split()[0],'scikit-learn',sklearn.__version__,'scipy',scipy.__version__)
print(r.to_string(index=False))
print('Changed predictions in',int((r.pre_eo_predictions_changed>0).sum()),'/10 seeds;',
      'mean',r.pre_eo_predictions_changed.mean(),'/1584')
print('Matches frozen Re010 confusion matrices:',int(r.re010_cm_matches.sum()),'/10 seeds')
print('Saved',out)
