import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,20).astype(np.float32)
W=C[(np.unpackbits(np.frombuffer(B,np.uint8,441,40),bitorder="little")[:3528].reshape(-1,3).astype(np.uint32)<<np.arange(3,dtype=np.uint32)).sum(1)].reshape(98,12)
g=np.random.default_rng(0)
R=g.standard_normal((2,9),dtype=np.float32)*9**-.5
t=g.standard_normal(2,dtype=np.float32)*.5
def predict(x):
 v=np.log1p(np.add.reduceat(abs(np.fft.rfft(x.reshape(len(x),16,1000)*1.)),(43.8*10**np.linspace(0,1.094,16)-43.8).astype(int),2)).reshape(len(x),-1)
 c=np.lib.stride_tricks.sliding_window_view(v.reshape(len(x),16,16),(3,3),(1,2)).reshape(len(x),14*14,9)
 c=(c-c.mean(2,keepdims=True))/np.sqrt(c.var(2,keepdims=True)+.01)
 h=np.maximum(c@R.T-t,0).reshape(len(x),7,2,7,2,2).mean((2,4)).reshape(len(x),-1)
 return np.argmax(h@W+C[8:],1)
