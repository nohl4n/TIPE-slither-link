import numpy as np
from Show import show_n
from copy import deepcopy
from _able import Cross
from Rules import Verif
from Rules import in_pas_diag

def List_3x3():

    L = []
    for i in range (512):
    
        s = list(format(i,'09b'))
    
        for p in range (9):
            if s[p] == '0':
                s[p] = False
            else:
                s[p] = True

        M = np.array([[s[8],s[7],s[6]],[s[1],s[0],s[5]],[s[2],s[3],s[4]]])
        
        
        L.append(M)
        
    return L
    
def List_nxm(n,m):
    L = []
    f = '0'
    f += str(n*m)
    f += 'b'
    
    for k in range(2**(n*m)):
        s = list(format(k,f))
    
        for p in range (n*m):
            if s[p] == '0':
                s[p] = False
            else:
                s[p] = True
                
        M = [[False]*m]*n
        M=np.array(M)
        somme = 0
    
        for i in range(n):
            for j in range(m):
                M[i,j] = s[somme]
                somme += 1
                
        L.append(M)
    
    return L
    
def filtre_equiv(L):
    Hist = []
    L_filtre = []
    for M in L:
        Ml = M.tolist()
        if not Ml in Hist:
            Hist.append(Ml)
            Hist.append(np.rot90(M,k=1).tolist())
            Hist.append(np.rot90(M,k=2).tolist())
            Hist.append(np.rot90(M,k=3).tolist())
            Mt = np.transpose(M)
            Hist.append(Mt.tolist())
            Hist.append(np.rot90(Mt,k=1).tolist())
            Hist.append(np.rot90(Mt,k=2).tolist())
            Hist.append(np.rot90(Mt,k=3).tolist())
            L_filtre.append(M)
            
    return L_filtre

def filtre_puzzle(L):
    L_filtre =[]
    for M in L:
        if Verif(M):
            L_filtre.append(M)
    return L_filtre
    
def filtre_diag(L):
    L_filtre =[]
    for M in L:
        if in_pas_diag(M):
            L_filtre.append(M)
    return L_filtre
    
def filtre_Point(L,marge):
    L_filtre =[]
    for M in L:
        if Point(M) <= marge:
            L_filtre.append(M)
    return L_filtre

def List_3x3_able():
    L = []
    for i in range (512):
    
        s = list(format(i,'09b'))
    
        for p in range (9):
            if s[p] == '0':
                s[p] = False
            else:
                s[p] = True

        M = np.array([[s[8],s[7],s[6]],[s[1],s[0],s[5]],[s[2],s[3],s[4]]])
        precedent=s[8]
        var = 0
        k = 1
        
        while k<9 and var<=2 :
            if precedent != s[k]:
                var +=1

            precedent = s[k]
            k+=1

        is_Cross = (s[0]==s[1]) or (s[0]==s[3]) or (s[0]==s[5]) or (s[0]==s[7])
        
        if var <= 2 and is_Cross :
            L.append(M)
    
    return L
    
def List_3x3_puzzle():

    L = []
    for i in range (512):
    
        s = list(format(i,'09b'))
    
        for p in range (9):
            if s[p] == '0':
                s[p] = False
            else:
                s[p] = True

        M = np.array([[s[8],s[7],s[6]],[s[1],s[0],s[5]],[s[2],s[3],s[4]]])
        
        
        if Verif(M):
            L.append(M)
        
    return L
    
def nb_in(M):
    n,m = len(M),len(M[0])
    nb = 0
    
    for i in range(n):
        for j in range(m):
            if M[i,j]:
                nb +=1
                
    return nb
    
def List_in(L):
    L_in = []
    for M in L:
        L_in.append(nb_in(M))
        
    return L_in
    
def Sort(L,L_in):
    n= len(L)
    maxi = max(L_in)
    
    L_sort = []
    
    for i in range(maxi+1):
        for j in range (n):
            if L_in[j] == i:
                L_sort.append(L[j])
        
    return np.array(L_sort)
    
def Tri(L):
    return Sort(L,List_in(L))
    

def Ligne(M,i):
    return M[i]

def Colonne(M,j):
    n= len(M)
    C=[]
    for i in range(n):
        C.append(M[i][j])
        
    return C
    
def Moyenne(L,M):
    n= len(M)
    moy=0
    for Bool in L:
        if Bool:
            moy+=1/n
        else:
            moy-=1/n
    
    return moy
    
def Point(M):
    n,m = len(M),len(M[0])
        
    pts = 0
    
    for i in range(n):
        pts += abs((Moyenne(Ligne(M,i),M))**2)
    
    for j in range(m):
        pts += abs((Moyenne(Colonne(M,j),M))**2)
        
    pts = pts/(n+m)
    
    return pts
    

