import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,20).astype(np.float32)
W=C[(np.unpackbits(np.frombuffer(B,np.uint8,2304,40),bitorder="little")[:18432].reshape(-1,3).astype(np.uint32)<<np.arange(3,dtype=np.uint32)).sum(1)].reshape(512,12)
def predict(x):
 v=np.log1p(np.add.reduceat(abs(np.fft.rfft(x.reshape(len(x),32,500)*1.)),(21.9*10**np.linspace(0,1.094,16)-21.9).astype(int),2)).reshape(len(x),-1)
 return np.argmax(v@W+C[8:],1)
