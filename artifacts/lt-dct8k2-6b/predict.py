import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,64,1).astype(np.float32)
W=C[(np.unpackbits(np.frombuffer(B,np.uint8,1448,129),bitorder="little")[:11580].reshape(-1,6).astype(np.uint32)<<np.arange(6,dtype=np.uint32)).sum(1)].reshape(193,10)
D=np.cos(np.outer(np.arange(8)*2+1,np.arange(2))*.19634954)
def predict(x):
 return np.argmax(((D.T@x.reshape(-1,4,8,4,8,3).transpose(0,1,3,5,2,4)@D).reshape(len(x),-1))@W[:-1]+W[-1],1)
