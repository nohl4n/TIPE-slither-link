import matplotlib.pyplot as plt
import numpy as np
import random as r
import copy as c
import sys
from Show import show

#_____________AUGMENTE_LA_RECURSION_DEPTH____________________

sys.setrecursionlimit(10000)


#_______________________________METHODE_RECURSIVE__________________________________

def grignotage_rec(n,m):


    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0,n,0,m)
    b=len(B)
    r.shuffle(B)
    a = r.randint(1,b-1)
    
    for k in range(a):
        if In_able(M,B[k][0],B[k][1]):
            generate_rec(M,B[k][0],B[k][1],n*m)
    
    return M
    
def grignotage_rec2(n,m):

    L =[[True]*m]*n
    M = np.array(L)
    generate_rec(M,n//2,m//2,n*m)
    
    return M
    
def inverse(M):
    n,m = len(M) , len(M[0])
    for i in range(n):
        for j in range(m):
            if M[i][j]:
                M[i][j] = False
            else:
                M[i][j] = True

def generate_rec (M,i,j,n):
    if n>0:
    
        M[i][j] = False
    
        C = Cross_ind(M,i,j)
        r.shuffle(C)
        c = len (C)

    
        for k in range(r.randint(1,c-1)):
            if In_able(M,C[k][0],C[k][1]):
                generate_rec(M,C[k][0],C[k][1],n-1)


def Cross_ind(M,i,j):
    
    n,m = len(M) , len(M[0])

    if i==0 and j==0:
        C =[[i,j+1],[i+1,j]]
                    
    elif i==n-1 and j==0:
        C =[[i-1,j],[i,j+1]]
        
    elif i==0 and j==m-1:
        C =[[i,j-1],[i+1,j]]
                              
    elif i==n-1 and j==m-1:
        C =[[i,j-1],[i-1,j]]
                    
    elif i==0:
        C = [[i,j-1],[i+1,j],[i,j+1]]
                    
    elif j==0:

        C =[[i-1,j],[i,j+1],[i+1,j]]
                    
    elif i==n-1:
        C =[[i,j-1],[i-1,j],[i,j+1]]
                    
    elif j==m-1:
        C =[[i-1,j],[i,j-1],[i+1,j]]
    
    else:
        C = [[i-1,j],[i,j+1],[i+1,j],[i,j-1]]
        
    return C

#_______________________________METHODE_NOISE+WAY__________________________________

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
    
#____tRAveEau________
def way2 (i1,j1,i2,j2,n,m):
    W=[]
    i,j = i1,j1
    while i!=i2 and j!=j2 :
        di=abs(i2-i)
        dj=abs(j2-j)
        a=1
        b=0.7
        p= r.random()
        
        if p<= (a+dj)/(di+dj+2*a):
            p = r.random()
            if p<=b:
                if ((sign(j2-j)==1 and j<m-1) or (sign(j2-j)==-1 and j>0)):
                    j+= sign(j2-j)
                else:
                    j-= sign(j2-j)
            else:
                if ((sign(j2-j)==1 and j>0) or (sign(j2-j)==-1 and j<m-1)):
                    j-= sign(j2-j)
                else:
                    j+= sign(j2-j)
            
        else:
            p = r.random()
            if p<=b:
                if ((sign(i2-i)==1 and i<n-1) or (sign(i2-i)==-1 and i>0)):
                    i+= sign(i2-i)
                else:
                    i-= sign(i2-i)
            else:
                if ((sign(i2-i)==1 and i>0) or (sign(i2-i)==-1 and i<n-1)):
                        i-= sign(i2-i)
                else:
                    i+= sign(i2-i)
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
    
#3______________________________METHODE_GRIGNOTAGE_CARRE___________________________

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

def grignotage(M,n,m):

#    L =[[True]*m]*n
#    M = np.array(L)
    B= bordure(0,n,0,m)
    b=len(B)
    r.shuffle(B)
    
    for i in range(r.randint(0,b-1)):
        if In_able(M,B[i][0],B[i][1]):
            M[B[i][0]][B[i][1]] = False
    p = min(n,m)
    
    for k in range(1,p//2):
        B= bordure(k,n-k,k,m-k)
        r.shuffle(B)
        b=len(B)
        for l in range(r.randint(4*b//5,b-1)):
            if In_able(M,B[l][0],B[l][1]):
                M[B[l][0]][B[l][1]] = False
    
    return M

def Crack(n,m):
    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0,n,0,m)
    
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1,b//5)):
        i,j = r.randint(1,n-1), r.randint(1,m-1)
        W = way (B[k][0],B[k][1],i,j)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if In_able(M,w[0],w[1]):
                M[w[0]][w[1]]= False
                

    return M
    
def Crack2(n,m):
    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0,n,0,m)
    
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1,b//5)):
        i,j = r.randint(1,n-1), r.randint(1,m-1)
        W = way2 (B[k][0],B[k][1],i,j,n,m)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if In_able(M,w[0],w[1]):
                M[w[0]][w[1]]= False
                

    return M
#______________Methode_Maze_____________

def init_maze(n,m):
    M=[[0]*m]*n
    M = np.array(M)
    Hist = []
    compteur = 1
    
    for i in range(1,n,2):
        for j in range(1,m,2):
            M[i][j] = compteur
            Hist.append([i,j])
            compteur += 1
            
    return M,Hist
    
def init_maze2(n,m,proba=0.1):
    
    M=[[0]*(2*m)]*(2*n)
    M=np.array(M)
    
    for i in range(1,2*n,2):
        for j in range(1,2*m,2):
            rand = r.random()
            if rand < proba:
                M[i][j] = 1
                
    show(M)
    M = M.tolist()
    print(M)
                
    for k in range(2*n,n,-1):
        supr_l(M,r.randint(0,k-1))
        show(M)
        
    for k in range(2*m,m,-1):
        supr_col(M,r.randint(0,k-1))
        show(M)
            
    return M
    
def Histo(M):
    n,m = len(M),len(M[0])
    Hist = []
    
    for i in range(n):
        for j in range(m):
            if M[i][j] != 0:
                Hist.append([i,j])
    return Hist    
    
    
    
def supr_col(M,ind):
    n,m = len(M),len(M[0])
    
    for i in range(n):
        for j in range(ind,m-1):
        
            M[i][j] , M[i][j+1] = M[i][j+1] , M[i][j]
            
    for i in range(n):
        print(M)
        M[i].pop()

   
    return M
    
def supr_l(M,ind):
    n= len(M)
    
    for i in range(ind,n-1):
        
        M[i],M[i+1] = M[i+1],M[i]
    
    M.pop()
                
    return M
    
    
    
    
def link(M,i,j):
    n,m= len(M),len(M[0])
    L = []
    val = M[i][j]
    if M[i][j] + M[(i+2)%n][j] != val and M[i][j] + M[(i+2)%n][j] < 2*val:
        L.append([i+2,j])
        
    if M[i][j] + M[(i-2)%n][j] != val and M[i][j] + M[(i-2)%n][j] < 2*val:
        L.append([i-2,j])
    
    if M[i][j] + M[i][(j+2)%m] != val and M[i][j] + M[i][(j+2)%m] < 2*val:
        L.append([i,j+2])
    
    if M[i][j] + M[i][(j-2)%m] != val and M[i][j] + M[i][(j-2)%m] < 2*val:
        L.append([i,j-2])
        
    return L
    
def generate_maze(n,m):
    
    M,Hist= init_maze(n,m)

    h = len(Hist)
    r.shuffle(Hist)
    S=[]
    
    while h != 0:
        ind = r.randint(0,h-1)
        Hist[ind][0],Hist[-1][0] = Hist[-1][0],Hist[ind][0]
        Hist[ind][1],Hist[-1][1] = Hist[-1][1],Hist[ind][1]
        i,j = Hist[-1][0],Hist[-1][1]
        L = link(M,i,j)
        
        if len(L) == 0:
            Hist.pop()
            h-=1
            
        else:
            r.shuffle(L)
            
            i_p , j_p = L[0][0],L[0][1]
            val = M[i][j]
            Hist.append([(i+i_p)//2,(j+j_p)//2])
            h+=1
            M[(i+i_p)//2][(j+j_p)//2] = val
            
            homog(M,i_p,j_p,val)

            
    return M
    
def homog(M,i,j,val):
    M[i][j] = val
    Cross = Cross_ind(M,i,j)
    for i_p,j_p in Cross:
        if M[i_p][j_p] != 0 and M[i_p][j_p] != val:
            homog(M,i_p,j_p,val)
