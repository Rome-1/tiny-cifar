import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,28).astype(np.float32)
W=C[(np.frombuffer(B,np.uint8,4704,56)[:,None]>>np.array([0,4])&15)].reshape(784,12)
def predict(x):
 v=np.log1p(np.add.reduceat(abs(np.fft.rfft(x[:,:15974].reshape(len(x),49,326)*1.)),(14.3*10**np.linspace(0,1.094,16)-14.3).astype(int),2)).reshape(len(x),-1)
 return np.argmax(v@W+C[16:],1)
