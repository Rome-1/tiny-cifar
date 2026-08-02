import numpy as np,pathlib
B=(pathlib.Path(__file__).parent/"w").read_bytes()
b=B[0];k=1<<b;c=[16, 32, 48, 64];o=1+2*16*k
C=np.frombuffer(B,np.float16,16*k,1).astype(np.float32).reshape(16,k)
S=np.frombuffer(B,np.float16,181,o).astype(np.float32)
q=np.unpackbits(np.frombuffer(B[o+2*181:],np.uint8),bitorder="little")[:7322*b].reshape(-1,b)
q=(q.astype(np.uint32)<<np.arange(b,dtype=np.uint32)).sum(1)
SP=[(c[0],(c[0],3,3,3)),(1,(c[0],))]
for i in range(3):SP+=[(1,(c[i],3,3)),(1,(c[i],)),(c[i+1],(c[i+1],c[i])),(1,(c[i+1],))]
SP+=[(10,(10,c[3])),(1,(10,))]
P=[];p=0;g=0
for j,(m,s) in enumerate(SP):
 n=int(np.prod(s));P.append((C[j,q[p:p+n]].reshape(m,-1)*S[g:g+m,None]).reshape(s));p+=n;g+=m
def sw(x):
 x=np.pad(x,((0,0),(1,1),(1,1),(0,0)))
 return np.lib.stride_tricks.sliding_window_view(x,(3,3),(1,2))
def mp(x):
 n,h,w,d=x.shape
 return x.reshape(n,h//2,2,w//2,2,d).max((2,4))
def fw(x):
 v=sw(x);h=np.maximum(v.reshape(v.shape[0],32,32,-1)@P[0].reshape(c[0],-1).T+P[1],0);h=mp(h)
 for i in range(3):
  j=2+4*i
  h=np.maximum((sw(h)*P[j]).sum((4,5))+P[j+1],0)
  h=np.maximum(h@P[j+2].T+P[j+3],0)
  if i<2:h=mp(h)
 return h.mean((1,2))@P[14].T+P[15]
def predict(x):
 o=np.empty(len(x),np.int64)
 for i in range(0,len(x),250):
  z=x[i:i+250].astype(np.float32)/255
  o[i:i+250]=np.argmax(fw(z)+fw(z[:,:,::-1]),1)
 return o
