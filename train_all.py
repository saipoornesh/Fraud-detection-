import os,json,joblib,numpy as np
from pathlib import Path
from src.data.dataset_loader import load_transactions
from src.data.feature_engineering import build_features
from src.data.split import split_scale
from src.models.classical import models
from src.models.deep import train_lstm,train_ae,ae_error,lstm_score
from src.ensemble.fusion import scale_score,best_threshold,combine
from src.evaluation.metrics import evaluate
from src.utils.config import DATA_PATH,MODEL_DIR
df=load_transactions(DATA_PATH); X,y=build_features(df)
A,B,C,yt,yv,yte,scaler=split_scale(X.reset_index(drop=True),y.reset_index(drop=True))
Path(MODEL_DIR).mkdir(parents=True,exist_ok=True); Path("artifacts/reports").mkdir(parents=True,exist_ok=True)
mods={}; pv=[]; pt=[]; report={}
for name,m in models().items():
 m.fit(A,yt); v=m.predict_proba(B)[:,1]; t=m.predict_proba(C)[:,1]
 pv.append(v); pt.append(t); mods[name]=m; report[name]=evaluate(yte,t,best_threshold(yv,v))
 print(name,report[name]["f1"])
from sklearn.ensemble import IsolationForest
iso=IsolationForest(n_estimators=120,contamination="auto",random_state=42).fit(A)
iv=-iso.score_samples(B); it=-iso.score_samples(C); ilo,ihi=np.quantile(iv,[.05,.95])
isv=scale_score(iv,ilo,ihi); ist=scale_score(it,ilo,ihi)
ae=train_ae(A[yt==0]); av=ae_error(ae,B); at=ae_error(ae,C); alo,ahi=np.quantile(av,[.05,.95])
asv=scale_score(av,alo,ahi); ast=scale_score(at,alo,ahi)
lstm=train_lstm(A,yt); lv=lstm_score(lstm,B); lt=lstm_score(lstm,C)
sv=np.mean(pv,axis=0); st=np.mean(pt,axis=0)
if lstm is not None: sv=.85*sv+.15*lv; st=.85*st+.15*lt
ev=combine(sv,(isv+asv)/2); et=combine(st,(ist+ast)/2); threshold=best_threshold(yv,ev)
report["ensemble"]=evaluate(yte,et,threshold)
joblib.dump({"models":mods,"scaler":scaler,"features":list(X.columns),"iso":iso,"ae":ae,"lstm":lstm,
"iso_range":[float(ilo),float(ihi)],"ae_range":[float(alo),float(ahi)],"threshold":float(threshold)},Path(MODEL_DIR)/"bundle.joblib")
json.dump(report,open("artifacts/reports/metrics.json","w"),indent=2)
print("Saved model bundle and evaluation metrics.")
