import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .evaluation import evaluate
PRE_RUN=['build_duration','checkout_duration','previous_failure_count','retry_count','code_changes_count','trigger_type','total_tests']
# Post-run columns NEVER in PRE_RUN: failed_tests, deployment_error_count
def train_models(org, dataset_path='datasets/sample_pipeline_runs.csv'):
    df=pd.read_csv(dataset_path).sort_values('run_id'); n_test=max(10,int(len(df)*0.2)); train, test=df.iloc[:-n_test], df.iloc[-n_test:]
    y_train=(train['execution_status']=='Failure').astype(int); y_test=(test['execution_status']=='Failure').astype(int)
    pre=ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), [c for c in PRE_RUN if c!='trigger_type']), ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), ['trigger_type'])])
    models={'random_forest': RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1), 'logistic': LogisticRegression(max_iter=1000), 'gradient_boosting': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=42)}
    results={}
    for name, clf in models.items():
        pipe=Pipeline([('pre', pre), ('clf', clf)]); pipe.fit(train[PRE_RUN], y_train); prob=pipe.predict_proba(test[PRE_RUN])[:,1]; results[name]=evaluate(y_test.values, prob); joblib.dump({'model':pipe,'version':name+'-v1','metrics':results[name]}, f'trained_models/{org}_{name}.joblib')
    best=max(results, key=lambda k: results[k]['f1'])
    return {'organization_id':org,'train_size':len(train),'test_size':len(test),'results':results,'selected':best}
