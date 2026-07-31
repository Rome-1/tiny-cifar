import numpy as np
B=np.frombuffer(open(__file__[:-10]+"w","rb").read(),np.uint8)
def predict(x):
 v=x.reshape(-1,8,4,8,4,3).mean((2,4)).reshape(len(x),-1)
 return B[18+(v[:,B[:9]]>B[9:18])@(1<<np.arange(9))]
