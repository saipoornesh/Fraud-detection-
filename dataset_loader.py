import pandas as pd
def load_transactions(path):
    df=pd.read_csv(path).drop_duplicates().copy()
    df=df.rename(columns={"Class":"isFraud","fraud":"isFraud","label":"isFraud"})
    if "amount" not in df or "isFraud" not in df: raise ValueError("CSV needs amount and isFraud/Class/label columns")
    df["amount"]=pd.to_numeric(df["amount"],errors="coerce").fillna(0).clip(lower=0)
    df["isFraud"]=pd.to_numeric(df["isFraud"],errors="coerce").fillna(0).astype(int)
    if "step" not in df: df["step"]=range(len(df))
    return df
