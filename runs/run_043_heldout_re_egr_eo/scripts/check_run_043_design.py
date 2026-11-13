"""Offline preflight for Run 043; no AIF360/Fairlearn model fitting required."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from weighted_oracle import ExternalWeightOracleLR, features_fingerprint

ROOT=Path(__file__).resolve().parents[1]
METRICS=('accuracy','statistical_parity_difference','disparate_impact',
         'equal_opportunity_difference','predictive_equality_difference')
EXPECTED='6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e'


def main():
    path=ROOT/'data/compas_farayola_numeric.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED
    df=pd.read_csv(path)
    assert df.shape==(5278,12)
    assert df.race.value_counts().to_dict()=={0:3175,1:2103}
    assert df.two_year_recid.value_counts().to_dict()=={0:2795,1:2483}
    x=df.drop(columns='two_year_recid'); y=df.two_year_recid
    prov=ROOT/'provenance'
    references={key:pd.read_csv(prov/f'run_{key}_{stem}_per_seed_results_A.csv').set_index('seed')
      for key,stem in [('011','egr'),('014','re_egr'),('016','egr_eo'),('022','re_egr_eo')]}
    for key,frame in references.items():
        assert sorted(frame.index.tolist())==list(range(101,111)),f'Missing {key} reference seed'
    for seed in range(101,111):
        train,outer=train_test_split(np.arange(len(df)),test_size=.3,random_state=seed,
                                     shuffle=True,stratify=x.race)
        strata=(x.iloc[outer].race.astype(str).to_numpy()+'_'+y.iloc[outer].astype(str).to_numpy())
        cal,test=train_test_split(outer,test_size=.5,random_state=seed+10000,
                                  shuffle=True,stratify=strata)
        assert (len(train),len(cal),len(test))==(3694,792,792)
        assert len(set(train)|set(cal)|set(test))==len(df)
        assert not (set(train)&set(cal) or set(train)&set(test) or set(cal)&set(test))
        for split,ids in [('train',train),('cal',cal),('test',test)]:
            counts=df.iloc[ids].groupby(['race','two_year_recid']).size()
            assert all(counts.get((g,l),0)>0 for g in (0,1) for l in (0,1)),(seed,split)
        for a,b in [('011','014'),('016','022')]:
            for metric in METRICS:
                assert np.isclose(references[a].loc[seed,metric],
                                  references[b].loc[seed,metric],atol=1e-12,rtol=0)
        print(f'Seed {seed}: valid disjoint 3694/792/792; frozen EGR and EO controls match')

    # Verify cloneability and the precise oracle weight transformation introduced
    # in Run 042. We do NOT falsely claim EGR itself was fitted here.
    frame=pd.DataFrame({'a':np.arange(12.),'b':np.arange(12.)%2})
    y=np.array([0]*6+[1]*6)
    internal=np.array([.8,1.0,1.1,1.3,1.0,.9,1.1,1.2,1.0,.8,1.1,1.0])
    ext=(.8,1.2)*6
    sha=features_fingerprint(frame)
    simple=LogisticRegression(solver='lbfgs',max_iter=1000).fit(frame,y,sample_weight=internal)
    unity=clone(ExternalWeightOracleLR((1.,)*12,sha)).fit(frame,y,sample_weight=internal)
    assert np.array_equal(simple.predict(frame),unity.predict(frame))
    assert np.allclose(simple.coef_,unity.model_.coef_,atol=1e-12,rtol=0)
    assert not unity.FIT_TRACES[-1]['changed_from_internal']
    weighted=clone(ExternalWeightOracleLR(ext,sha)).fit(frame,y,sample_weight=internal)
    assert weighted.FIT_TRACES[-1]['changed_from_internal']
    assert weighted.FIT_TRACES[-1]['ratio_std']>0
    assert np.isclose(weighted.FIT_TRACES[-1]['combined_weight_mass'],
                      weighted.FIT_TRACES[-1]['internal_weight_mass'],atol=1e-8,rtol=0)
    try:
        clone(ExternalWeightOracleLR(ext,sha)).fit(frame.iloc[::-1].reset_index(drop=True),y,
                                                   sample_weight=internal)
    except AssertionError as exc:
        assert 'reordered/changed' in str(exc)
    else:raise AssertionError('External-oracle row-alignment guard failed')
    print('\nRUN 043 DESIGN CHECK PASSED (no EGR/EO optimization executed locally).')

if __name__=='__main__':main()
