import numpy as np,pandas as pd,os
os.makedirs("data/raw",exist_ok=True); r=np.random.default_rng(42); n=5000
fraud=(r.random(n)<.04).astype(int); amount=r.lognormal(3.5,.9,n)*(1+fraud*4)
df=pd.DataFrame({"step":np.arange(n),"type":r.choice(["PAYMENT","TRANSFER","CASH_OUT","CASH_IN"],n),
"amount":amount,"nameOrig":[f"C{x}" for x in r.integers(0,500,n)],"oldbalanceOrg":r.lognormal(7,1,n),
"newbalanceOrig":r.lognormal(7,1,n),"nameDest":[f"M{x}" for x in r.integers(0,800,n)],"isFraud":fraud})
df.to_csv("data/raw/transactions.csv",index=False); print("Synthetic demo data saved.")
