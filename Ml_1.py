import numpy as np
n=int(input("Enter Dimensions:"))
print(f"n={n}")

x=np.zeros((n,1))
A=np.zeros((n,n))
B=np.zeros((n,n))

print("Enter values for vector x")
for i in range(n):
    x[i][0]=int(input())
print(f"x=\n{x}")

print("Enter values for vector A")
for i in range(n):
    for j in range(n):
        A[i][j]=int(input())
print(f"A=\n{A}")

print("Enter values for vector B")
for i in range(n):
    for j in range(n):
        B[i][j]=int(input())
print(f"B=\n{B}")

# Transpose A^t and B^t
At=np.transpose(A)
print(f"A'=\n{At}")

Bt=np.transpose(B)
print(f"B'=\n{Bt}")

# AB
AB=np.dot(A,B)
print(f"AB=\n{AB}")

# (AB)^t=B^tA^t
AB_t=np.transpose(AB)
print(f"AB^t=\n{AB_t}")
BtAt=np.dot(Bt,At)
print(f"B^tA^t=\n{BtAt}")
print(f"(AB)^t == B^tA^t\n{AB_t==BtAt}")

#Determinant of A and B
detA=(np.linalg.det(A)).round(0)
print(f"|A|={detA}")
detB=(np.linalg.det(B)).round(0)
print(f"|B|={detB}")

if (detA==0 and detB==0):
    print("A and B are Singular Matrix")
elif (detA==0 and detB!=0):
    print("A is Singular and B is Non-Singular Matrix.")
elif (detA!=0 and detB==0):
    print("A is Non-Singular and B is Singular Matrix.")
elif (detA!=0 and detB!=0):
    print("A is Non-Singular and B is Non-Singular Matrix.")

#Inverse of Matrix
if(detA!=0):
    Ainv=np.linalg.inv(A).round(2)
    print(f"Inverse of A=\n{Ainv}")
else:
    print("Inverse of A is not Possible")
if(detB!=0):
    Binv=np.linalg.inv(B).round(2)
    print(f"Inverse of B=\n{Binv}")
else:
    print("Inverse of B is not Possible")

I=np.identity(n)
print(f"Identity Matrix I=\n{I}")

# AAinv=I
AAinv=np.dot(A,Ainv).round(0)
print(f"A*Ainv=\n{AAinv}")
AinvA=np.dot(Ainv,A).round(0)
print(f"Ainv*A=\n{AinvA}")

#Trace of A and B , Ainv, Binv , A' and B'
traceA=np.trace(A)
traceB=np.trace(B)
traceAinv=np.trace(Ainv)
traceBinv=np.trace(Binv)
traceAt=np.trace(At)
traceBt= np.trace(Bt)
print(f"tr(A)={traceA}")
print(f"tr(B)={traceB}")
print(f"tr(Ainv)={traceAt}")
print(f"tr(Binv)={traceBt}")
print(f"tr(A')={traceAt}")
print(f"tr(B')={traceBt}")

# BA
BA=np.dot(B,A)
print(f"BA=\n{BA}")

# verification tranceAB = traceBA
traceAB=np.trace(AB)
traceBA=np.trace(BA)
print(f"tr(AB)==tr(BA)\n{traceAB==traceBA}")

#%% B'A, AB' A'B, BA'

BtA=np.dot(Bt,A)
print(f"B'A=\n{BtA}")
ABt=np.dot(A,Bt)
print(f"AB'=\n{AB}")
AtB=np.dot(At,B)
print(f"A'B=\n{AB}")
BAt=np.dot(B,At)
print(f"BA'=\n{BAt}")

# trace(A'B) == trace(B'A)==trace(AB')==trace(BA')
traceAtB=np.trace(AtB)
traceABt=np.trace(ABt)
traceBAt=np.trace(BAt)
traceBtA=np.trace(BtA)
print(f"tr(A'B)={traceAtB}")
print(f"tr(AB')={traceABt}")
print(f"tr(BA')={traceBAt}")
print(f"tr(B'A)={traceBtA}")

# y=Ax ; y belongs to R^n
y=np.dot(A,x)
print(f"y=Ax\n{y}")
#Retrieve x from y
xnew=np.dot(Ainv,y).round(0)
print(f"Retrieved x=\n{xnew}")
print(f"x==retrieved x?\n{x==xnew}")

#Inner Product of x and y and verify that they are orthogonal or not
inner_product=sum(np.multiply(x,y))
print(f"inner product of x and y={inner_product}")
if inner_product==0:
    print("x and y are orthogonal")
else:
    print("x and y are not orthogonal")

# norm of x
xnorm=0
for i in x:
    xnorm=xnorm+(i**2)
xnorm=np.sqrt(xnorm)
print(f"||x||={xnorm}")

normalised_x=x/xnorm
print(f"normalised x=\n{normalised_x}")

# norm of y
ynorm=0
for i in x:
    ynorm=ynorm+(i**2)
ynorm=np.sqrt(ynorm)
print(f"||y||={ynorm}")

normalised_y=y/ynorm
print(f"normalised y=\n{normalised_y}")

# CS inequality for x and y
cs_inequality=abs(np.dot(x.transpose(),y))<=xnorm*ynorm
print(f"CS inequality={cs_inequality}")

#verification x'y=y'x
xt=x.transpose()
yt=y.transpose()
print(f"x'y==y'x?\n{np.dot(xt,y)==np.dot(yt,x)}")
print(f"x'Ax==y'Ay?\n{np.dot(np.dot(yt,A),x) == np.dot(np.dot(xt,A),y)}")

# Eigen Values and Vectors
eig_A=np.linalg.eig(A)
lamA=eig_A[0]
uA=eig_A[1]
print(f"Eigen Values of A=\n{lamA}")
print(f"Eigen Vector of A=\n{uA}")

eig_B=np.linalg.eig(B)
lamB=eig_B[0]
uB=eig_B[1]
print(f"Eigen Values of B=\n{lamB}")
print(f"Eigen Vector of B=\n{uB}")

# Verification EVD for A and B
print(f"EVD for A=\n{A==np.dot(np.dot(uA,lamA*I).round(0),np.linalg.inv(uA)).round(0)}")
print(f"EVD for B=\n{B==np.dot(np.dot(uB,lamB*I).round(0),np.linalg.inv(uB)).round(0)}")














