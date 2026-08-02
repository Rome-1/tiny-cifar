import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
n=B[0];k=1<<n;D=73;K=10
C=np.frombuffer(B,np.float16,k,1).astype(np.float32)
c=np.unpackbits(np.frombuffer(B[1+2*k:],np.uint8),bitorder="little")[:D*K*n].reshape(-1,n)
W=C[(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)].reshape(D,K)
g=np.random.default_rng(1)
R=g.standard_normal((8,48),dtype=np.float32)*0.14433756729740646
t=g.standard_normal(8,dtype=np.float32)*0.1
def feats(x):
 n=len(x)
 w=np.lib.stride_tricks.sliding_window_view(x,(4,4),(1,2))[:,::2,::2]
 c=w.transpose(0,1,2,4,5,3).reshape(n,15*15,48)
 c=(c-c.mean(2,keepdims=True))/np.sqrt(c.var(2,keepdims=True)+.01)
 h=np.maximum(c@R.T-t,0).reshape(n,15,15,8)
 return h.reshape(n,3,5,3,5,8).mean((2,4)).reshape(n,-1)
def predict(x):
 x=x.astype(np.float32)/255
 o=np.empty(len(x),np.int64)
 for i in range(0,len(x),500):
  z=x[i:i+500];f=feats(z)+feats(z[:,:,::-1])
  o[i:i+500]=np.argmax(f@W[:-1]+W[-1],1)
 return o
