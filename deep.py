import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
class LSTM(nn.Module):
 def __init__(self,n):
  super().__init__(); self.net=nn.LSTM(n,32,batch_first=True); self.out=nn.Linear(32,1)
 def forward(self,x):
  z,_=self.net(x); return self.out(z[:,-1]).squeeze(-1)
class Autoencoder(nn.Module):
 def __init__(self,n):
  super().__init__(); self.enc=nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Linear(32,8)); self.dec=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,n))
 def forward(self,x): return self.dec(self.enc(x))
def train_lstm(X,y,epochs=4,window=5):
 seq=[]; lab=[]
 for i in range(window-1,len(X)): seq.append(X[i-window+1:i+1]); lab.append(y[i])
 if len(seq)<4 or len(set(lab))<2: return None
 m=LSTM(X.shape[1]); opt=torch.optim.Adam(m.parameters()); loss=nn.BCEWithLogitsLoss()
 ds=DataLoader(TensorDataset(torch.tensor(np.asarray(seq),dtype=torch.float32),torch.tensor(lab,dtype=torch.float32)),batch_size=128,shuffle=True)
 m.train()
 for _ in range(epochs):
  for a,b in ds: opt.zero_grad(); l=loss(m(a),b); l.backward(); opt.step()
 return m
def train_ae(X,epochs=6):
 if len(X)<4:return None
 m=Autoencoder(X.shape[1]); x=torch.tensor(np.asarray(X),dtype=torch.float32); opt=torch.optim.Adam(m.parameters())
 for _ in range(epochs):
  opt.zero_grad(); loss=((m(x)-x)**2).mean(); loss.backward(); opt.step()
 return m
def ae_error(m,X):
 if m is None:return np.zeros(len(X))
 m.eval()
 with torch.no_grad():
  x=torch.tensor(np.asarray(X),dtype=torch.float32); return ((m(x)-x)**2).mean(1).numpy()
def lstm_score(m,X,window=5):
 out=np.full(len(X),.5)
 if m is None or len(X)<window:return out
 seq=np.stack([X[i-window+1:i+1] for i in range(window-1,len(X))])
 m.eval()
 with torch.no_grad(): out[window-1:]=torch.sigmoid(m(torch.tensor(seq,dtype=torch.float32))).numpy()
 return out
