from copy import deepcopy
import numpy as np
from _able import Cross
from V2 import Cross_ind

def Rule1(M,NA):
    n,m = len(M) , len(M[0])
    i=0
    res = True
    
    while i<n and res:
        j=0
        while j<m and res:
            C = Cross(M,i,j)
            c=len(C)
            p=0
            if M[i][j]:
                p= 4-c
                for k in range(c):
                    if not C[k]:
                        p+=1
                print(i,j,NA[i][j],p)
                if NA[i][j]!=p:
                    res = False
    
            if not M[i][j]:
                for k in range(c):
                    if C[k]:
                        p+=1
                print(i,j,NA[i][j],p)
                if NA[i][j]!=p:
                    res = False
            j+=1
        i+=1
    
    return res

def Rule1_ind(M,i,j,NA):
    res = True
    C = Cross(M,i,j)
    c=len(C)
    p=0
    if M[i][j]:
        p= 4-c
        for k in range(c):
            if not C[k]:
                p+=1
        
            print(i,j,NA[i][j],p)
        
            if NA[i][j]!=p:
                res = False

        if not M[i][j]:
            for k in range(c):
                if C[k]:
                    p+=1
            print(i,j,NA[i][j],p)
            if NA[i][j]!=p:
                res = False
                
def NA(M):
    n,m = len(M) , len(M[0])
    L =[[0]*m]*n
    NA = np.array(L)
    
    for i in range (n):
        for j in range(m):
            C = Cross(M,i,j)
            c=len(C)
            n=0
            if M[i][j]:
                n= 4-c
                for k in range(c):
                    if not C[k]:
                        n+=1
                NA[i][j]=n
                
            if not M[i][j]:
                for k in range(c):
                    if C[k]:
                        n+=1
                NA[i][j]=n
    
    return NA
    
#___________________EQUIVALENT_2_COULEUR_____________________
    
#________________

def Hist(M,i,j,L):
    L.append([i,j])
    C = Cross_ind(M,i,j)
    
    for ind in C:
        if M[ind[0]][ind[1]] == M[i][j] and not ([ind[0],ind[1]] in L):
            Hist(M,ind[0],ind[1],L)

    
def Couche_ext(M):
    n,m = len(M),len(M[0])
    Couche = [[False]*(m+2)]*(n+2)
    Couche = np.array(Couche)
    
    for i in range(n+2):
        for j in range(m+2):
            if i>0 and i<=n and j>0 and j<=m:
                Couche[i][j] = M[i-1][j-1]
    
    return Couche
    
def Verif(M):
    n,m = len(M),len(M[0])

    i,j=0,0
    found = False
    
    while i<n and not found:
        j=0
        while j<m and not found:
            if M[i,j]:
                found = True
                i,j = i-1,j-1
            j+=1
        i+=1

    if found == False:
        return True
    else:
        
            
        L_int=[]
        Hist(M,i,j,L_int)
        
        L_ext=[]
        Couche= Couche_ext(M)
        Hist(Couche,0,0,L_ext)
        
        return len(L_int) + len(L_ext) == (n+2)*(m+2)
        

#________________

def in_pas_diag(M):
    
    res =True
    n,m = len(M),len(M[0])
    
    i,j=0,0
    
    while i<n-1 and res == True:
        j=0
        while j<m-1 and res == True:
            Carre = [[M[i][j],M[i][j+1]],[M[i+1][j],M[i+1][j+1]]]
            if Carre == [[True,False],[False,True]] or Carre == [[False,True],[True,False]]:
                res = False
            
            j+=1
        i+=1
        
    return res
   
        
        
