import os,uuid,json,joblib,numpy as np,pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from src.database.db import Session,Tx,init_db
from src.utils.config import MODEL_DIR
from src.models.deep import ae_error
from src.ensemble.fusion import scale_score,combine,category
app=FastAPI(title="FinTech Fraud Detection API"); bundle=None
class Input(BaseModel):
 transaction_id:str|None=None
 amount:float=Field(gt=0)
 step:int=0
 transaction_type:str="TRANSFER"
 oldbalanceOrg:float|None=None
 newbalanceOrig:float|None=None
def load():
 global bundle
 p=os.path.join(MODEL_DIR,"bundle.joblib")
 if os.path.exists(p): bundle=joblib.load(p)
@app.on_event("startup")
def startup(): init_db(); load()
@app.get("/health")
def health(): return {"status":"ok","model_loaded":bundle is not None}
def make_x(t):
 if bundle is None: load()
 if bundle is None: raise HTTPException(503,"Train first: python scripts/train_all.py")
 row={f:0. for f in bundle["features"]}
 for k,v in {"amount":t.amount,"amount_log1p":np.log1p(t.amount),"amount_vs_history":t.amount,
 "balance_error_orig":abs((t.oldbalanceOrg or 0)-t.amount-(t.newbalanceOrig or 0))}.items():
  if k in row: row[k]=v
 key="type_"+t.transaction_type
 if key in row: row[key]=1.
 return bundle["scaler"].transform(pd.DataFrame([row],columns=bundle["features"]))
@app.post("/predict")
def predict(t:Input):
 x=make_x(t); probs=[float(m.predict_proba(x)[0,1]) for m in bundle["models"].values()]
 sup=float(np.mean(probs)); iso=float(scale_score(-bundle["iso"].score_samples(x),*bundle["iso_range"])[0])
 ae=float(scale_score(ae_error(bundle["ae"],x),*bundle["ae_range"])[0])
 score=float(combine([sup],[(iso+ae)/2])[0]); risk=category(score); tid=t.transaction_id or str(uuid.uuid4())
 db=Session()
 try:
  db.add(Tx(transaction_id=tid,amount=t.amount,step=t.step,risk_score=score,risk_category=risk,
  explanation=f"supervised={sup:.3f}; isolation={iso:.3f}; autoencoder={ae:.3f}")); db.commit()
 except Exception: db.rollback()
 finally: db.close()
 return {"transaction_id":tid,"model_probabilities":probs,"supervised_score":sup,
 "isolation_anomaly_score":iso,"autoencoder_anomaly_score":ae,"ensemble_risk_score":score,"risk_category":risk}
@app.get("/transactions")
def transactions(limit:int=100):
 db=Session()
 try:
  rows=db.query(Tx).order_by(Tx.id.desc()).limit(min(limit,500)).all()
  return [{"transaction_id":r.transaction_id,"amount":r.amount,"step":r.step,"risk_score":r.risk_score,
  "risk_category":r.risk_category,"explanation":r.explanation} for r in rows]
 finally: db.close()
@app.get("/alerts")
def alerts(): return [x for x in transactions() if x["risk_category"]!="LOW RISK"]
@app.get("/models")
def models_status(): return {"loaded":bundle is not None,"models":list(bundle["models"]) if bundle else []}
@app.get("/metrics")
def metrics():
 p="artifacts/reports/metrics.json"
 return json.load(open(p)) if os.path.exists(p) else {"message":"Train models first"}
@app.get("/explain/{tid}")
def explain(tid:str):
 db=Session()
 try:
  r=db.query(Tx).filter_by(transaction_id=tid).first()
  if not r: raise HTTPException(404,"Not found")
  return {"transaction_id":tid,"risk_score":r.risk_score,"risk_category":r.risk_category,"explanation":r.explanation}
 finally: db.close()
