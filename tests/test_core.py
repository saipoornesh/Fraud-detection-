import pandas as pd
from src.data.feature_engineering import build_features
from src.ensemble.fusion import category,combine
def test_features():
 d=pd.DataFrame({"step":[1,2,3],"amount":[10,20,15],"nameOrig":["a","a","b"],"isFraud":[0,1,0]})
 X,y=build_features(d); assert len(X)==3 and len(y)==3 and "amount_log1p" in X
def test_fusion():
 assert category(.1)=="LOW RISK" and category(.5)=="MEDIUM RISK" and category(.9)=="HIGH RISK"
 assert abs(combine([.5],[.5])[0]-.5)<1e-8
