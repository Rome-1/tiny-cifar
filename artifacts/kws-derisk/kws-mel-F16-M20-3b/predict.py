import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,20).astype(np.float32)
W=C[(np.unpackbits(np.frombuffer(B,np.uint8,1440,40),bitorder="little")[:11520].reshape(-1,3).astype(np.uint32)<<np.arange(3,dtype=np.uint32)).sum(1)].reshape(320,12)
def predict(x):
 v=np.log1p(np.add.reduceat(abs(np.fft.rfft(x.reshape(len(x),16,1000)*1.)),(43.8*10**np.linspace(0,1.094,20)-43.8).astype(int),2)).reshape(len(x),-1)
 return np.argmax(v@W+C[8:],1)
