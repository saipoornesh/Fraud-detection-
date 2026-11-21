import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
def evaluate(y,p,t=.5):
 pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
 "recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),
 "roc_auc":float(roc_auc_score(y,p)) if len(set(y))>1 else None,
 "pr_auc":float(average_precision_score(y,p)) if len(set(y))>1 else None,
 "false_positive_rate":float(fp/(fp+tn)) if fp+tn else 0.0,"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}
