from copy import deepcopy

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
    
___________________EQUIVALENT_2_COULEUR_____________________
    
def verif(M):
    
    # pas de touche in en diag
    
    res1 =True
    n,m = len(M),len(M[0])
    
    i,j=0,0
    
    while i<n-1 and res == True:
        j=0
        while j<m-1 and res == True:
            Carre = Carre(M,i,j)
            if Carre == np.array([[True],[False]],[False,True]]) or Carre == np.array([[False],[True]],[True,False]]):
                res = False
            
            j+=1
        i+=1
            
    # Toutes les in sont liés
    
    while i<n and not M[i][j]:
        j=0
            while j<m and not M[i][j]:
                j+=1
            i+=1
    
    M_copy = deepcopy(M)
    killer (M_copy,i,j)
                
    
    
    while i<n and not M[i][j]:
        j=0
            while j<m and not M[i][j]:
                j+=1
            i+=1
    
    res2 = not M[i][j]
    
    return res1,res2

def Carre(M,i,j):
    n,m = len(M),len(M[0])
    
    if i>=(n-1) and j>=(m-1):
        return np.array([[M[i][j],M[i][j+1]],
                        [M[i+1][j],M[i+1][j+1]]])

def killer(M,i,j):
    M[i][j] = False
    
    C = Cross(M,i,j)
    n =len(C)
    
    for k in range(n):
        if C[k]:
            return killer(M,i,j)
            
def in_en_diag(M):
    
    
    
    res1 =True
    n,m = len(M),len(M[0])
    
    i,j=0,0
    
    while i<n-1 and res == True:
        j=0
        while j<m-1 and res == True:
            Carre = Carre(M,i,j)
            if Carre == np.array([[True],[False]],[False,True]]) or Carre == np.array([[False],[True]],[True,False]]):
                res = False
            
            j+=1
        i+=1
