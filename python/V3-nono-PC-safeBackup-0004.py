import numpy as np
from Show import show_n
from copy import deepcopy
from _able import Cross
from Rules import Verif

def List_3x3():
    

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
    
    
