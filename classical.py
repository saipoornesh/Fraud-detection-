from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
def models():
 return {"logistic_regression":LogisticRegression(max_iter=1000,class_weight="balanced"),
 "random_forest":RandomForestClassifier(n_estimators=150,class_weight="balanced_subsample",n_jobs=-1,random_state=42),
 "xgboost":XGBClassifier(n_estimators=150,max_depth=5,learning_rate=.06,eval_metric="logloss",n_jobs=-1,random_state=42),
 "lightgbm":LGBMClassifier(n_estimators=150,class_weight="balanced",verbosity=-1,random_state=42)}
