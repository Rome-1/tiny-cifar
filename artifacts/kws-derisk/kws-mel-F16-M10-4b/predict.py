import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,28).astype(np.float32)
W=C[(np.frombuffer(B,np.uint8,960,56)[:,None]>>np.array([0,4])&15)].reshape(160,12)
def predict(x):
 v=np.log1p(np.add.reduceat(abs(np.fft.rfft(x.reshape(len(x),16,1000)*1.)),(43.8*10**np.linspace(0,1.094,10)-43.8).astype(int),2)).reshape(len(x),-1)
 return np.argmax(v@W+C[16:],1)
