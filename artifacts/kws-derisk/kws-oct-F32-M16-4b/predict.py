import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,28).astype(np.float32)
W=C[(np.frombuffer(B,np.uint8,3072,56)[:,None]>>np.array([0,4])&15)].reshape(512,12)
def predict(x):
 v=np.log1p(np.add.reduceat(abs(np.fft.rfft(x.reshape(len(x),32,500)*1.)),(2**np.linspace(0,7.966,16)).astype(int),2)).reshape(len(x),-1)
 return np.argmax(v@W+C[16:],1)
