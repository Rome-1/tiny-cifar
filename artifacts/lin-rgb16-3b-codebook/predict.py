import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
n=B[0];k=1<<n;D=769
C=np.frombuffer(B,np.float16,k,1).astype(np.float32)
c=np.unpackbits(np.frombuffer(B[1+2*k:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=C[(c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)].reshape(D,10)
def predict(x):
 x=x.astype(np.float32)
 f=x.reshape((-1,16,2,16,2,3)).mean((2,4)).reshape(len(x),-1)/255
 return np.argmax(f@W[:-1]+W[-1],1)
