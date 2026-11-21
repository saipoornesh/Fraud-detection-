import numpy as np
from sklearn.metrics import f1_score
def scale_score(x,lo,hi):
 return np.zeros(len(x)) if hi<=lo else np.clip((np.asarray(x)-lo)/(hi-lo),0,1)
def best_threshold(y,p):
 return max((f1_score(y,p>=t,zero_division=0),t) for t in np.linspace(.05,.95,37))[1]
def combine(supervised,anomaly): return np.clip(.8*np.asarray(supervised)+.2*np.asarray(anomaly),0,1)
def category(s): return "HIGH RISK" if s>=.7 else ("MEDIUM RISK" if s>=.3 else "LOW RISK")
