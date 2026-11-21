from sklearn.preprocessing import StandardScaler
def split_scale(X,y):
    n=len(X); nt=max(1,int(.2*n)); nv=max(1,int(.2*n)); end=n-nt-nv
    if end<2: raise ValueError("Need at least 10 rows")
    Xtr,Xv,Xte=X.iloc[:end],X.iloc[end:n-nt],X.iloc[n-nt:]
    ytr,yv,yte=y.iloc[:end],y.iloc[end:n-nt],y.iloc[n-nt:]
    scaler=StandardScaler(); A=scaler.fit_transform(Xtr); B=scaler.transform(Xv); C=scaler.transform(Xte)
    return A,B,C,ytr.to_numpy(),yv.to_numpy(),yte.to_numpy(),scaler
