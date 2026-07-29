import numpy as np
B=open(__file__[:-10]+"w","rb").read()
C=np.frombuffer(B,np.float16,4,1).astype(np.float32)
W=C[(np.frombuffer(B,np.uint8,183,9)[:,None]>>np.array([0,2,4,6])&3).reshape(-1)[:730]].reshape(73,10)
g=np.random.default_rng(1)
R=g.standard_normal((8,48),dtype=np.float32)*48**-.5
t=g.standard_normal(8,dtype=np.float32)*0.1
def f(x):
 c=np.lib.stride_tricks.sliding_window_view(x,(4,4),(1,2))[:,::2,::2].transpose(0,1,2,4,5,3).reshape(len(x),15*15,48)
 c=(c-c.mean(2,keepdims=True))/np.sqrt(c.var(2,keepdims=True)+.01)
 h=np.maximum(c@R.T-t,0).reshape(len(x),3,5,3,5,8)
 return h.mean((2,4)).reshape(len(x),-1)
def predict(x):
 return np.concatenate([np.argmax((f(z)+f(z[:,:,::-1]))@W[:-1]+W[-1],1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//500))])
