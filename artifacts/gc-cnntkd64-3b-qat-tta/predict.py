import numpy as np
B=open(__file__[:-10]+"w","rb").read()
b=3;k=8;c=[16,32,48,64];o=257
M=np.maximum
C=np.frombuffer(B,np.float16,16*k,1).astype(np.float32).reshape(16,k)
S=np.frombuffer(B,np.float16,181,o).astype(np.float32)
q=(np.unpackbits(np.frombuffer(B,np.uint8,2746,619),bitorder="little")[:21966].reshape(-1,3).astype(np.uint32)<<np.arange(3,dtype=np.uint32)).sum(1)
SP=[(c[0],(c[0],3,3,3)),(1,(c[0],))]
for i in range(3):SP+=[(1,(c[i],3,3)),(1,(c[i],)),(c[i+1],(c[i+1],c[i])),(1,(c[i+1],))]
SP+=[(10,(10,c[3])),(1,(10,))]
P=[];p=0;g=0
for j,(m,s) in enumerate(SP):
 n=int(np.prod(s));P.append((C[j,q[p:p+n]].reshape(m,-1)*S[g:g+m,None]).reshape(s));p+=n;g+=m
def sw(x):
 return np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(1,1),(1,1),(0,0))),(3,3),(1,2))
def mp(x):
 n,h,w,d=x.shape
 return x.reshape(n,h//2,2,w//2,2,d).max((2,4))
def fw(x):
 v=sw(x);h=mp(M(v.reshape(len(x),32,32,-1)@P[0].reshape(c[0],-1).T+P[1],0))
 for i in range(3):
  j=2+4*i
  h=M((sw(h)*P[j]).sum((4,5))+P[j+1],0)
  h=M(h@P[j+2].T+P[j+3],0)
  if i<2:h=mp(h)
 return h.mean((1,2))@P[14].T+P[15]
def T(z):
 return fw(z)+fw(z[:,:,::-1])
def predict(x):
 return np.concatenate([np.argmax(sum(T(np.roll(z,d,(1,2)))for d in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]),1)for z in np.array_split(x.astype(np.float32)/255,-(-len(x)//250))])
