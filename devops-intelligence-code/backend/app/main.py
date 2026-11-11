from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .ml.training import train_models
app = FastAPI(title='DevOps Intelligence API', version='2.4.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'], allow_methods=['*'], allow_headers=['*'])
class PredictIn(BaseModel):
    organization_id: str
    build_duration: float
    checkout_duration: float
    previous_failure_count: int
    retry_count: int
    code_changes_count: int
    trigger_type: str = 'push'
    total_tests: int = 100
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/api/models/train')
def train(org: str, dataset_id: str): return train_models(org, dataset_id)
@app.post('/api/models/predict')
def predict(inp: PredictIn):
    import joblib, numpy as np, os
    path=f'trained_models/{inp.organization_id}_random_forest.joblib'
    
    if not os.path.exists(path): raise HTTPException(409,'No model trained for this organization. Train a model first.')
    d=joblib.load(path); x=np.array([[inp.build_duration, inp.checkout_duration, inp.previous_failure_count, inp.retry_count, inp.code_changes_count, inp.trigger_type, inp.total_tests]], dtype=object)
    prob=float(d['model'].predict_proba(x)[0][1])
    return {'predicted':'Failure' if prob>=0.5 else 'Success','failure_probability':prob,'success_probability':1-prob,'risk':'LOW' if prob<0.35 else 'MEDIUM' if prob<=0.65 else 'HIGH','model_version':d['version']}
