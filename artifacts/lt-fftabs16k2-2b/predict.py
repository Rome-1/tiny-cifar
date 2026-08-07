import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,4,1).astype(np.float32)
W=C[(np.frombuffer(B,np.uint8,123,9)[:,None]>>np.array([0,2,4,6])&3).reshape(-1)[:490]].reshape(49,10)
def predict(x):
 return np.argmax((abs(np.fft.rfft2(x.reshape(-1,2,16,2,16,3),axes=(2,4))[:,:,:2,:,:2]).reshape(len(x),-1))@W[:-1]+W[-1],1)
