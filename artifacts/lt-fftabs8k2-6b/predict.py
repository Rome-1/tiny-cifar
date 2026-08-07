import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,64,1).astype(np.float32)
W=C[(np.unpackbits(np.frombuffer(B,np.uint8,1448,129),bitorder="little")[:11580].reshape(-1,6).astype(np.uint32)<<np.arange(6,dtype=np.uint32)).sum(1)].reshape(193,10)
def predict(x):
 return np.argmax((abs(np.fft.rfft2(x.reshape(-1,4,8,4,8,3),axes=(2,4))[:,:,:2,:,:2]).reshape(len(x),-1))@W[:-1]+W[-1],1)
