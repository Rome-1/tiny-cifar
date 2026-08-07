import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,16).astype(np.float32)
W=C[(np.frombuffer(B,np.uint8,480,32)[:,None]>>np.array([0,2,4,6])&3)].reshape(160,12)
def predict(x):
 v=np.log1p(np.add.reduceat(abs(np.fft.rfft(x.reshape(len(x),16,1000)*1.)),(2**np.linspace(0,8.966,10)).astype(int),2)).reshape(len(x),-1)
 return np.argmax(v@W+C[4:],1)
