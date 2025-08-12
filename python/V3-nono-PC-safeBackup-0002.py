import numpy as np
from Show import show_n

def List_3x3


def List_3x3_2():
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

def verif(M):
    
    # pas de touche in en diag
    
    res =True
    n,m = len(M),len(M[0])
    
    i,j=0,0
    
    while i<n-1 and res == True:
        j=0
        while j<m-1 and res == True:
            Carre = Carre(M,i,j)
            j+=1
        i+=1
            
            if Carre == np.array([[True],[False]],[False,True]]) or Carre == np.array([[False],[True]],[True,False]]):
                res = False
                
    



def Carre(M,i,j):
    n,m = len(M),len(M[0])
    
    if i>=(n-1) and j>=(m-1):
        return np.array([[M[i][j],M[i][j+1]],
                        [M[i+1][j],M[i+1][j+1]]])

