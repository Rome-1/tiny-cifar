import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
n=B[0];k=1<<n;D=2049
C=np.frombuffer(B,np.float16,k,1).astype(np.float32)
c=np.unpackbits(np.frombuffer(B[1+2*k:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=C[(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)].reshape(D,10)
g=np.random.default_rng(1)
R=(g.standard_normal((192,2048),dtype=np.float32)*0.07216878364870323).astype(np.float32)
b=g.standard_normal(2048,dtype=np.float32)*0.1
def predict(x):
 x=x.astype(np.float32)
 f=x.reshape((-1,8,4,8,4,3)).mean((2,4)).reshape(len(x),-1)/255
 o=np.empty(len(x),np.int64)
 for i in range(0,len(x),4096):
  h=np.maximum(f[i:i+4096]@R+b,0)
  o[i:i+4096]=np.argmax(h@W[:-1]+W[-1],1)
 return o
