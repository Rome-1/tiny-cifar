import numpy as np,pathlib,struct
B=(pathlib.Path(__file__).parent/"w").read_bytes()
n=B[0];s,z=struct.unpack_from("<ff",B,1)
D=65
c=np.unpackbits(np.frombuffer(B[9:],np.uint8),bitorder="little")[:D*10*n].reshape(-1,n)
W=((c.astype(np.uint32)<<np.arange(n,dtype=np.uint32)).sum(1)*s+z).reshape(D,10)
def predict(x):
 x=x.astype(np.float32)
 f=x.mean(3).reshape((-1,8,4,8,4)).mean((2,4)).reshape(len(x),-1)/255
 return np.argmax(f@W[:-1]+W[-1],1)
