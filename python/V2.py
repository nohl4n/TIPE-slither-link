import matplotlib.pyplot as plt
import numpy as np
import random as r
import math as m


def init_Map(n,m):
    L =[[False]*m]*n
    M = np.array(L)
    return M

#V1 GENE
def random_case (n,m):
    return r.randint(0,n-1),r.randint(0,m-1)

def random_neighbord (i,j):
    s = r.randint(1,4)
    if s == 1:
        return i,j-1
    if s == 2:
        return i,j+1
    if s == 3:
        return i+1,j
    if s == 4:
        return i-1,j
        

def generate_1 (M,n,m,i,j):

    if not M.any():
        i,j = random_case(n,m)
        M[i][j] = True
        return generate_1 (M,n,m,i,j)
    
    while ((i>=0 and i<n) and (j>=0 and j<m)) and M[i][j]:
        i,j = random_neighbord(i,j)
        
    print(i,j)
    
    if (i<0 or i>n-1) or (j<0 or j>m-1):
        return M
    else:
        M[i][j] = True
        return generate_1 (M,n,m,i,j)

#2
def noise (n,m,r):
    L= [[False]*m]*n
    M = np.array(L)
    H= []
    for _ in range (int((n+m)*r)):
        i,j = random_case (n,m)
        H.append([i,j])
        M[i][j] = True
    
    return M,H

def sign(x):
    if x<0:
        return -1
    else:
        return 1
        

def way (i1,j1,i2,j2):
    W=[]
    i,j = i1,j1
    for _ in range (abs(i1-i2) + abs(j1-j2)):
        s = r.randint(0,1)
        if s==1:
            if i != i2:
                i += sign(i2-i)
            
            else:
                j += sign(j2-j)
        
        else:
            if j != j2:
                j += sign(j2-j)
            
            else:
                i += sign(i2-i)
        
        W.append([i,j])
    
    return W
    
def generate_2(n,m,r):
    M,H = noise(n,m,r)
    h = len(H)
    for k in range(h-1):
        i1,j1 = H[k][0], H[k][1]
        i2,j2 = H[k+1][0], H[k+1][1]
        W = way(i1,j1,i2,j2)
        for c in W:
            M[c[0]][c[1]] = True
        
    return M
    
#3
def Breakable(M,i,j):

    C = Cycle(M,i,j)
    n = len(C)
    precedent=C[0]
    var = 0
    
    for k in range(1,n+1):
        if (precedent and (not C[k%n])) or ((not precedent) and C[k%n]):
            var +=1
            
        precedent = C[k%n]
    
    return var <= 2 and Cross(M,i,j)
    
def Cross(M,i,j):
    
    n,m = len(M) , len(M[0])

    if i==0 and j==0:
        C =np.array([M[i][j+1],M[i+1][j]])
                    
    elif i==n-1 and j==0:
        C =np.array([M[i-1][j],M[i][j+1]])
        
    elif i==0 and j==m-1:
        C =np.array([M[i][j-1],M[i+1][j]])
                              
    elif i==n-1 and j==m-1:
        C =np.array([M[i][j-1],M[i-1][j]])
                    
    elif i==0:
        C =np.array([M[i][j-1],M[i+1][j],M[i][j+1]])
                    
    elif j==0:

        C =np.array([M[i-1][j],M[i][j+1],M[i+1][j]])
                    
    elif i==n-1:
        C =np.array([M[i][j-1],M[i-1][j],M[i][j+1]])
                    
    elif j==m-1:
        C =np.array([M[i-1][j],M[i][j-1],M[i+1][j]])
    
    else:
        C =np.array([M[i-1][j],M[i][j+1],M[i+1][j],M[i][j-1]])
        
    c=len(C)
    return (not C[0]) or (not C[1]) or (not C[2%n]) or (not C[3%n])
    
def Cycle(M,i,j):

    n,m = len(M) , len(M[0])
    
    if i==0 and j==0:
        C =np.array([M[i][j+1],M[i+1][j+1],M[i+1][j]])
                    
    elif i==n-1 and j==0:
        C =np.array([M[i-1][j],M[i-1][j+1],M[i][j+1]])
        
    elif i==0 and j==m-1:
        C =np.array([M[i][j-1],M[i+1][j-1],M[i+1][j]])
                              
    elif i==n-1 and j==m-1:
        C =np.array([M[i][j-1],M[i-1][j-1],M[i-1][j]])
                    
    elif i==0:
        C =np.array([M[i][j-1],M[i+1][j-1],M[i+1][j],M[i+1][j+1],M[i][j+1]])
                    
    elif j==0:
        C =np.array([M[i-1][j],M[i-1][j+1],M[i][j+1],M[i+1][j+1],M[i+1][j]])
                    
    elif i==n-1:
        C =np.array([M[i][j-1],M[i-1][j-1],M[i-1][j],M[i-1][j+1],M[i][j+1]])
                    
    elif j==m-1:
        C =np.array([M[i-1][j],M[i-1][j-1],M[i][j-1],M[i+1][j-1],M[i+1][j]])
    
    else:
        C =np.array([M[i-1][j-1],M[i-1][j],M[i-1][j+1],M[i][j+1],M[i+1][j+1], M[i+1][j], M[i+1][j-1],M[i][j-1]])
        
    return C

def bordure(nd,nf,md,mf):
    B=[]
    for j in range(md,mf):
        B.append([nd,j])
    for i in range(nd+1,nf):
        B.append([i,mf-1])
    for j in range(mf-2,md-1,-1):
        B.append([nf-1,j])
    for i in range(nf-1,nd-1,-1):
        B.append([i,md])
    return B

def grignotage(n,m):

    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0,n,0,m)
    b=len(B)
    r.shuffle(B)
    
    for i in range(r.randint(0,b-1)):
        M[B[i][0]][B[i][1]] = False
    p = min(n,m)
    
    for k in range(1,p//2):
        B= bordure(k,n-k,k,m-k)
        r.shuffle(B)
        b=len(B)
        for l in range(r.randint(4*b//5,b-1)):
            if Breakable(M,B[l][0],B[l][1]):
                M[B[l][0]][B[l][1]] = False
    
    return show(M)
    
    

def show (M):
    fig, ax = plt.subplots()
    img = ax.imshow(M)
    plt.show()
