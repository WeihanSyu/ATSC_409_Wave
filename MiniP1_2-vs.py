import numpy as np
import matplotlib.pylab as plt
#Removing all the first 4 functions from our primary matrices by just setting a variable equal to the functions
#At specific parameter (N,h,Amax, etc.) made our main matrix functions and calculations soooo much faster.
#Except we have to manually switch parameters again and again for different cases.

# d: depth, measured from surface
# h: approximate depth of the bottom of the ocean's surface layer
# Ah: The eddy viscosity or mixing coefficient
# derAh: the finite difference derivative matrix for mixing coefficients
# IoMatrix: the light intensity matrix on right side of Ax = b equation
# All together: Ax = b >>> Tmatrix[T] = IoMatrix; solve for T for all depths

def dRange(N):      
    d = np.arange(0,(200+1/N),(200/N))   #step size = final value - initial value / N subdivisions 
    return d
d = dRange(1000)

def dh(N,h):      # Could have just changed dRange(N) to d here since we already decided to define d above
    i = next(x[0] for x in enumerate(dRange(N)) if x[1] > h)
    #First index in array d that is greater than h, note it counts 0 as first index
    #So if element 22 in d is greater than h, that means d[21], i = 21
    return i 
dh = dh(1000,10)

def Ah(Adepth,Amax,Adip,h,N): #for d > h
    A = np.zeros(N-dh)  #Last element gives us Ah at d(final - 1) which is exactly what we need
    for i in range((N-dh)):
        A[i] = Adepth + (Amax - Adepth - Adip*(d[i+dh]-h))*np.exp(-0.5*(d[i+dh]-h))
    return A
Ah = Ah(0.0001,0.01,0.0015,10,1000)

def derAh(Adepth,Amax,Adip,h,N): #for d > h
    derAh = np.zeros(N-dh)
    for i in range((N-dh)):
        derAh[i] = -0.5*Amax*np.exp(-0.5*(d[i+dh]-h)) + 0.5*Adepth*np.exp(-0.5*(d[i+dh]-h)) - Adip*np.exp(-0.5*(d[i+dh]-h)) + 0.5*Adip*(d[i+dh]-h)*np.exp(-0.5*(d[i+dh]-h))
    return derAh
derAh = derAh(0.0001,0.01,0.0015,10,1000)
                                                                                                                           
def IoMatrix(beta,alpha,albedo,N,Amax,Cp):
    I = np.zeros((1,N-1)).reshape(N-1,1)
    Io = beta*100*(1-albedo)
    for i in range(dh-1):   #Cause our matrix starts at d(1/N) m depth not d(0) since our b.c. is T(d(0)) = -1
        I[i] = (1/(Amax*(N/200)*Cp))*(Io*np.exp(-alpha*d[i+2])-Io*np.exp(-alpha*d[i+1]))
    I[0] = I[0] - (-1)
    for i in range((dh-1),N-1):
        I[i] = (1/Cp)*(Io*np.exp(-alpha*d[i+2])-Io*np.exp(-alpha*d[i+1]))
    I[N-2] = I[N-2] -(-2)*(derAh[N-dh-1] + Ah[N-dh-1]*(N/200))
    return I

def TMatrix(N):
    T = np.zeros((N-1,N-1))
    for i in range(dh-1):
        T[i,i] = -2
        T[i,i+1] = 1
    for i in range((dh-2),-1,-1):
        if i-1 > -1:
            T[i,i-1] = 1
    for i in range((dh-1),N-2):
        T[i,i] = -((derAh[i-(dh-1)]) + (2*(N/200)*Ah[i-(dh-1)]))
        T[i,i+1] = (derAh[i-(dh-1)]) + ((N/200)*Ah[i-(dh-1)])
    for i in range(N-3,(dh-2),-1):
        T[i,i-1] = Ah[i - dh+1]*(N/200)
    T[N-2,N-2] = -((derAh[N-dh-1]) + (2*(N/200)*Ah[N-dh-1]))
    T[N-2,N-3] = Ah[N-dh-1]*(N/200)
    return T
#Base
A = TMatrix(1000)
b = IoMatrix(0.5,0.1,0.1,1000,0.01,4E6)
Ai = np.linalg.inv(A)
T = np.dot(Ai,b)
T = np.append(-1,T)
T = np.append(T,-2)

plt.figure()
plt.xlabel('Temperature [degC]')
plt.ylabel('Depth [m]')
plt.title('Arctic Ocean Temperature vs Depth plot (Base case)')
plt.plot(T,d)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])

#No ice cover
b = IoMatrix(1,0.1,0.1,1000,0.01,4E6)
T = np.dot(Ai,b)
T = np.append(-1,T)
T = np.append(T,-2)

plt.figure()
plt.xlabel('Temperature [degC]')
plt.ylabel('Depth [m]')
plt.title('Arctic Ocean Temperature vs Depth plot (No ice)')
plt.plot(T,d)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
'''
#Base but stronger Adepth
A = TMatrix(1000)
b = IoMatrix(0.5,0.1,0.1,1000,0.01,4E6)
Ai = np.linalg.inv(A)
T = np.dot(Ai,b)
T = np.append(-1,T)
T = np.append(T,-2)

plt.figure()
plt.xlabel('Temperature [degC]')
plt.ylabel('Depth [m]')
plt.title('Arctic Ocean Temperature vs Depth plot (Base but 10x Adepth)')
plt.plot(T,d)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
'''
#Make decrease in Io very small
A = TMatrix(1000)
b = IoMatrix(1,0.001,0.001,1000,0.01,4E6)
Ai = np.linalg.inv(A)
T = np.dot(Ai,b)
T = np.append(-1,T)
T = np.append(T,-2)

plt.figure()
plt.xlabel('Temperature [degC]')
plt.ylabel('Depth [m]')
plt.title('Arctic Ocean Temperature vs Depth plot (Io small decrease)')
plt.plot(T,d)
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
'''
#Deepening mixing depth
A = TMatrix(1000)
b = IoMatrix(0.5,0.1,0.1,200,0.01,50,4E6)
Ai = np.linalg.inv(A)
T = np.dot(Ai,b)
T = np.append(-1,T)
T = np.append(T,-2)

plt.figure()
plt.xlabel('Temperature [degC]')
plt.ylabel('Depth [m]')
plt.title('Arctic Ocean Temperature vs Depth plot (Deepening mixing layer)')
plt.plot(T,dRange(200))
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
'''


plt.show()

