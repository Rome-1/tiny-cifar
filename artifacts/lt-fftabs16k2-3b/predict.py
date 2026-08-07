import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,8,1).astype(np.float32)
W=C[(np.unpackbits(np.frombuffer(B,np.uint8,184,17),bitorder="little")[:1470].reshape(-1,3).astype(np.uint32)<<np.arange(3,dtype=np.uint32)).sum(1)].reshape(49,10)
def predict(x):
 return np.argmax((abs(np.fft.rfft2(x.reshape(-1,2,16,2,16,3),axes=(2,4))[:,:,:2,:,:2]).reshape(len(x),-1))@W[:-1]+W[-1],1)
