"""Reproducible descriptive audit; no final-model training or test-label use."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/final_evidence'
OUT.mkdir(exist_ok=True)
train = pd.read_csv(ROOT / 'data/train_final.csv')
test = pd.read_csv(ROOT / 'data/test_final.csv')
sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
dictionary = pd.read_csv(ROOT / 'data/data_dictionary.csv').set_index('kolom')
u = train.drop_duplicates('id').reset_index(drop=True)
t = test.drop_duplicates('id').reset_index(drop=True)
features = [c for c in test if c not in ('id', 'household_id')]
y = u.life_satisfaction.to_numpy()
facts = {'train_shape': train.shape, 'test_shape': test.shape, 'sample_shape': sample.shape,
         'unique_train': len(u), 'unique_test': len(t),
         'train_conflicting_id_rows': int(train.groupby('id').nunique(dropna=False).gt(1).any(axis=1).sum()),
         'test_conflicting_id_rows': int(test.groupby('id').nunique(dropna=False).gt(1).any(axis=1).sum()),
         'id_overlap': len(set(u.id) & set(t.id)),
         'sample_test_id_overlap': len(set(sample.id) & set(t.id)),
         'sample_equals_test_ids': sample.id.equals(test.id),
         'class_counts_unique': u.life_satisfaction.value_counts().sort_index().to_dict(),
         'class_counts_rows': train.life_satisfaction.value_counts().sort_index().to_dict(),
         'dictionary_only': sorted(set(dictionary.index)-set(train)),
         'train_only': sorted(set(train)-set(dictionary.index)),
         'raw_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')}}
uh = pd.util.hash_pandas_object(u[features], index=False)
th = pd.util.hash_pandas_object(t[features], index=False)
facts['distinct_id_identical_predictor_rows'] = int(uh.duplicated().sum())
facts['train_test_identical_predictor_hashes'] = len(set(uh) & set(th))
facts['household_train_unique'] = int(u.household_id.nunique())
facts['household_overlap_train_test'] = len(set(u.household_id) & set(t.household_id))
facts['id_range_train'] = [int(u.id.min()),int(u.id.max())]
facts['id_order_spearman'] = float(spearmanr(np.arange(len(u)),u.id).statistic)

def family(c):
    for name, pat in [('survey','year|month|quarter|survey|pseudo|record|sample|status_recode|imputation|respondent'),
                      ('mental','anxiety|depression|mental|therapy'), ('food','food|meals'),
                      ('economic_employment','income|poverty|work|employer|sick_leave|home_ownership|years_in_current'),
                      ('functional','difficulty|disability|glasses|hearing_aid|mobility'),
                      ('healthcare','insurance|plan|care|medical|prescription|medication|vaccine|doctor|hospital|emergency'),
                      ('physical','health|ever|cancer|height|weight|bmi|immune'),
                      ('demographic_household','age|sex|marit|family|household|child|parent|education|citizen|born|military')]:
        if re.search(pat,c): return name
    return 'other'

rows=[]
for c in features:
    a,b=u[c],t[c]
    cats=a.fillna('__NaN__').astype(str)
    tab=pd.crosstab(cats,u.life_satisfaction)
    cond=float(np.average(entropy(tab.to_numpy(),base=2,axis=1),weights=tab.sum(axis=1)))
    counts=a.value_counts(dropna=False)
    av=a.fillna(-123456).value_counts(normalize=True)
    bv=b.fillna(-123456).value_counts(normalize=True)
    tv=float(av.sub(bv,fill_value=0).abs().sum()/2)
    defined=re.findall(r'(?:^|,\s*)(\d+)=',str(dictionary.loc[c,'nilai'])) if c in dictionary.index else []
    codes=set(map(float,defined))
    outside=int((a.notna() & ~a.isin(codes)).sum()) if codes else None
    rows.append({'feature':c,'family':family(c),'cardinality':int(a.nunique()),
                 'missing_train':float(a.isna().mean()),'missing_test':float(b.isna().mean()),
                 'min':float(a.min()),'max':float(a.max()),'categorical_TV':tv,
                 'conditional_entropy_bits_in_sample':cond,'dictionary_unlisted_observed':outside,
                 'values_counts':json.dumps({str(k):int(v) for k,v in counts.items()}) if len(counts)<150 else 'continuous/high-cardinality',
                 'semantic_type':'continuous_or_count' if a.nunique()>32 else 'coded_state_preserve',
                 'special_7_8_9_count_not_always_sentinel':int(a.isin([7,8,9]).sum())})
profile=pd.DataFrame(rows);profile.to_csv(OUT/'feature_profile.csv',index=False)
facts['mean_missing'] = float(u[features].isna().mean().mean())
facts['row_missing_quantiles'] = u[features].isna().sum(axis=1).quantile([0,.01,.25,.5,.75,.99,1]).to_dict()
missing=u[features].isna().astype(np.int8)
corr=missing.corr();pairs=[]
for i,c in enumerate(features):
    for d in features[i+1:]:
        if abs(corr.loc[c,d])>.03: pairs.append((c,d,float(corr.loc[c,d])))
pd.DataFrame(pairs,columns=['feature1','feature2','missingness_correlation']).sort_values('missingness_correlation',ascending=False).to_csv(OUT/'comissingness.csv',index=False)
byclass=[]
for k,g in u.groupby('life_satisfaction'):
    for c in features:
        byclass.append({'target':k,'feature':c,'missing_rate':g[c].isna().mean(),
                        'codes_789_rate_not_always_sentinel':g[c].isin([7,8,9]).mean()})
pd.DataFrame(byclass).to_csv(OUT/'class_conditional_states.csv',index=False)

relationships=[]
for a,b in [('survey_year','data_year'),('interview_month','interview_quarter'),('food_security_4cat','food_security_3cat'),
            ('children_in_family','children_in_household'),('marital_status_detailed','marital_status'),
            ('smoking_status','ever_smoked'),('ecigarette_status','ever_used_ecigarette'),('cancer_types_count','cancer_ever')]:
    q=u[[a,b]].dropna();ct=pd.crosstab(q[a],q[b]);mp=ct.idxmax(axis=1)
    relationships.append({'source':a,'destination':b,'support':len(q),'mode_accuracy':float(ct.max(axis=1).sum()/len(q)),
                          'mapping':json.dumps(mp.to_dict()),'fill_opportunities':int((u[b].isna()&u[a].notna()).sum())})
pd.DataFrame(relationships).to_csv(OUT/'redundancy.csv',index=False)

# Historical OOF is for audit only, never an input to the final notebook.
oof=pd.read_csv(ROOT/'results/v11/oof_v11.csv')
z=u.merge(oof.drop(columns='life_satisfaction'),on='id',validate='one_to_one')
assert len(z)==len(u)
actual=z.life_satisfaction.to_numpy();pred=z.recovery_class.to_numpy();score=z.recovery_score.to_numpy()
cm=pd.crosstab(actual,pred).reindex(index=[1,2,3,4],columns=[1,2,3,4],fill_value=0)
cm.to_csv(OUT/'recovery_confusion.csv')
loss=(actual-pred)**2
families={'1_to_2':(actual==1)&(pred==2),'2_to_1':(actual==2)&(pred==1),
          'false_tail':(actual<=2)&(pred>=3),'missed_tail':(actual>=3)&(pred<=2),
          '3_4':(actual>=3)&(pred>=3)&(actual!=pred)}
facts['recovery']={'qwk':cohen_kappa_score(actual,pred,weights='quadratic'),
                   'rmse':float(np.sqrt(np.mean((actual-score)**2))),
                   'weighted_error_sum':int(loss.sum()),'transitions':{
                       name:{'rows':int(mask.sum()),'weighted_loss':int(loss[mask].sum()),'weighted_fraction':float(loss[mask].sum()/loss.sum())}
                       for name,mask in families.items()}}
for name,mask,labels in [('1v2',actual<=2,actual==2),('tail',np.ones(len(z),bool),actual>2),('3v4',actual>=3,actual==4)]:
    facts['recovery']['auc_'+name]=float(roc_auc_score(labels[mask],score[mask]))
    facts['recovery']['entropy_'+name]=float(entropy(np.bincount(labels[mask].astype(int)),base=2))
sub=[]
z['residual']=actual-score;z['weighted_error']=loss;z['false_tail']=families['false_tail']
z['missing_quartile']=pd.qcut(z[features].isna().sum(axis=1),4,duplicates='drop').astype(str)
for c in ['survey_year','data_year','interview_quarter','interview_month','sex','age','education_level','marital_status',
          'home_ownership','general_health_status','depression_frequency','food_security_4cat','income_poverty_category','missing_quartile']:
    for v,g in z.groupby(c,dropna=False):
        sub.append({'feature':c,'value':str(v),'n':len(g),'target_mean':g.life_satisfaction.mean(),
                    'tail_rate':(g.life_satisfaction>2).mean(),'score_mean':g.recovery_score.mean(),
                    'residual_mean':g.residual.mean(),'weighted_error_mean':g.weighted_error.mean(),
                    'false_tail_rate':g.false_tail.mean()})
pd.DataFrame(sub).to_csv(OUT/'train_oof_subgroups.csv',index=False)

# Conditional associations: raw labels within identical burden states, replicated across
# fixed historical folds. These test associations, not incremental model performance.
burden=['general_health_status','depression_frequency','food_security_4cat','worry_about_medical_costs']
z['burden_group']=z[burden].fillna(-1).astype(str).agg('|'.join,axis=1)
protective=[]
for c,good in [('marital_status',1),('home_ownership',1),('has_investment_income',1),('worked_last_week',1),('has_paid_sick_leave',1)]:
    for f,g in z.groupby('fold'):
        g=g[g[c].notna()].copy();g['protective']=g[c].eq(good)
        stats=g.groupby(['burden_group','protective']).agg(n=('id','size'),tail=('life_satisfaction',lambda s:(s>2).mean()),
                                                        mean=('life_satisfaction','mean'),residual=('residual','mean'))
        if True not in stats.index.get_level_values(1) or False not in stats.index.get_level_values(1):continue
        yes=stats.xs(True,level=1);no=stats.xs(False,level=1)
        both=yes.join(no,lsuffix='_yes',rsuffix='_no').dropna();both=both[(both.n_yes>=20)&(both.n_no>=20)]
        weights=np.minimum(both.n_yes,both.n_no)
        protective.append({'feature':c,'fold':f,'matched_strata':len(both),'overlap_weight':float(weights.sum()),
                           **{m+'_yes_minus_no':float(np.average(both[m+'_yes']-both[m+'_no'],weights=weights)) for m in ['tail','mean','residual']}})
pd.DataFrame(protective).to_csv(OUT/'protective_matched_associations.csv',index=False)

# Predeclared bounded shift probe: 12k respondents per domain, 3 folds, 150 trees.
# IDs are audited separately; they cannot enter final model features from this probe.
rng=np.random.default_rng(20260910)
a=u.iloc[rng.choice(len(u),12000,replace=False)];b=t.iloc[rng.choice(len(t),12000,replace=False)]
domains=pd.concat([a,b],ignore_index=True);dy=np.repeat([0,1],12000)
shift=[]
for name,cols in [('raw',features),('design',['survey_year','data_year','pseudo_stratum','pseudo_psu','survey_weight','interview_quarter','interview_month']),('id_only',['id'])]:
    pp=np.zeros(len(domains))
    for f,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20260910).split(domains,dy)):
        model=LGBMClassifier(n_estimators=150,num_leaves=15,learning_rate=.05,min_child_samples=80,
                            reg_lambda=5,verbosity=-1,n_jobs=4,random_state=20260910+f)
        model.fit(domains.iloc[tr][cols],dy[tr]);pp[va]=model.predict_proba(domains.iloc[va][cols])[:,1]
    shift.append({'view':name,'rows':len(domains),'auc':roc_auc_score(dy,pp)})
facts['adversarial_probe']=shift

# Inspect every ZIP entry byte-wise; compare backup against current primary results.
archives=[]
for p in [*ROOT.glob('sum/*.zip'),*ROOT.glob('.fr-*/*.zip')]:
    with zipfile.ZipFile(p) as f:
        for i in f.infolist():
            if i.is_dir():continue
            payload=f.read(i.filename);q=ROOT/i.filename
            archives.append({'archive':str(p.relative_to(ROOT)),'entry':i.filename,'bytes':len(payload),
                             'sha256':hashlib.sha256(payload).hexdigest(),
                             'same_as_working_file':q.exists() and hashlib.sha256(q.read_bytes()).digest()==hashlib.sha256(payload).digest()})
pd.DataFrame(archives).to_csv(OUT/'archive_inventory.csv',index=False)
(OUT/'raw_audit.json').write_text(json.dumps(facts,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)))
print(json.dumps(facts,indent=2,default=str))
