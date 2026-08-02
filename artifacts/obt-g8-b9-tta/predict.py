import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
E=np.eye(10,10,0,int)
def g(x):
 v=x.reshape(-1,8,4,8,4,3).mean((2,4)).reshape(len(x),-1)
 return E[B[18+(v[:,B[:9]]>B[9:18])@(1<<np.arange(9))]]
def predict(x):
 return sum(g(np.roll(x,d,(1,2))[:,:,::m])for m in(1,-1)for d in [(0,0),(4,0),(-4,0),(0,4),(0,-4)]).argmax(1)
