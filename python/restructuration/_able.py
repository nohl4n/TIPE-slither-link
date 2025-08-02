import numpy as np

def Out_able(M,i,j):

    C = Cycle(M,i,j)
    n = len(C)
    precedent=C[n-1]
    var = 0
    
    for k in range(n):
        if precedent != C[k]:
            var +=1
        precedent = C[k]
        
    if (n==5 or n==3) and var ==2:
        var +=3
        
    C=Cross(M,i,j)
    c=len(C)
    is_Cross = (not C[0]) or (not C[1%c]) or (not C[2%c]) or (not C[3%c])
    
    
    return var <= 2 and (is_Cross or n<6) and M[i][j]
    
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
        
    return C
    
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
