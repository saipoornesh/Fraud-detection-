import numpy as np, pandas as pd
def build_features(df):
    d=df.copy()
    acct="nameOrig" if "nameOrig" in d else ("account_id" if "account_id" in d else None)
    d=d.sort_values(([acct] if acct else [])+["step"],kind="stable")
    if acct:
        g=d.groupby(acct,sort=False)["amount"]
        d["account_tx_count_before"]=g.cumcount()
        d["account_mean_amount_before"]=g.transform(lambda s:s.shift().expanding().mean()).fillna(0)
        d["previous_amount"]=g.shift().fillna(0)
    else:
        d["account_tx_count_before"]=np.arange(len(d))
        d["account_mean_amount_before"]=d["amount"].shift().expanding().mean().fillna(0)
        d["previous_amount"]=d["amount"].shift().fillna(0)
    d["amount_log1p"]=np.log1p(d["amount"])
    d["amount_vs_history"]=d["amount"]/(d["account_mean_amount_before"]+1)
    d["balance_error_orig"]=0.0
    if {"oldbalanceOrg","newbalanceOrig"}<=set(d):
        d["balance_error_orig"]=(d.oldbalanceOrg-d.amount-d.newbalanceOrig).abs()
    if "type" in d: d=pd.get_dummies(d,columns=["type"],dtype=float)
    y=d["isFraud"].astype(int)
    X=d.drop(columns=[c for c in ["isFraud","nameOrig","nameDest","account_id","transaction_id"] if c in d])
    X=X.select_dtypes(include=[np.number,"bool"]).astype(float).replace([np.inf,-np.inf],0).fillna(0)
    return X,y
