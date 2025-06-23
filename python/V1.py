import matplotlib.pyplot as plt
import random as r

#rule of the game

def Rule_1 (M):
    G,L,C = M
    n,m = len(G[0]),len(G)
    i,j= 0,0
    res = True
    
    while i < m and res == True:
        j=0
        while j < n and res == True:
        
            if G[i][j] != None and G[i][j] != L[i][j] + L[i+1][j] + C[i][j] + C[i][j+1]:
                res = False
            j+=1
        i+=1
    return res
    
def two_connexion (M):
    G,L,C = M
    n,m = len(G),len(G[0])
    i,j= 0,0
    res = True
    
    while i < n+1 and res == True:
        j=0
        while j < m+1 and res == True:
        
            if (i==0):
            
                if j==0 and L[i][j]+C[i][j] == 1:
                    res = False
                    
                elif j==m and L[i][j-1]+C[i][j] == 1:
                    res = False
                    
                elif j<m and (L[i][j-1]+ L[i][j] +C[i][j])%2 == 1:
                    res = False
                    
            elif (j==0) :
            
                if i==0 and L[i][j]+C[i][j] == 1:
                    res = False
                    
                if i==n and L[i][j] +C[i-1][j] == 1:
                    res = False
                    
                elif i<m and (L[i][j] + C[i-1][j] +C[i][j])%2 != 0:
                    res = False
            
            elif (i==n and j==m) and L[i][j-1] + C[i-1][j] == 1:
                res = False
            
            elif (i<n and j<m) and (L[i][j-1]+L[i][j] + C[i-1][j] +C[i][j] != 2  and L[i][j-1]+L[i][j] + C[i-1][j] +C[i][j] != 0):
                res = False
                
            j+=1
        i+=1
    return res
                
                

def Rule_2_1 (M):
    # verifier que chaque point à 0 ou 2 connexions
    if not two_connexion(M):
        return False
    else:
        G,L,C = M
        n,m = len(G),len(G[0])
        G_fill = [[False]*n]*m
    
    
    
    return G

# générer Grid

def grid (n,m):
    G=[[None]*n]*m
    L=[[0]*n]*(m+1)
    C=[[0]*(n+1)]*m
    
    return G,L,C
    
# générer un cycle V1

def unicité_map (M):
     return M

def map (M):
    G,L,C = M
    

# representer grid
    
def draw_grid(M):

    G,L,C = M
    
    m,n = len(G[0]),len(G)
    fig, ax = plt.subplots(figsize=(n,m))

    # Tracer les points
    for i in range(n+1):
        for j in range(m+1):
            ax.plot(j, n-i, 'ko', markersize=8)

    # Ajouter les indices
    for i in range(n):
        for j in range(m):
            if G[-i-1][j] is not None:
                ax.text(j+0.5, n-i-0.5, str(G[-i-1][j]),
                        fontsize=16, ha='center', va='center')

    # Tracer les arêtes horizontales
    for i in range(n+1):
        for j in range(m):
            if L[-i-1][j] == 1:
                ax.plot([j, j+1], [n-i, n-i], 'k', linewidth=4)
            elif L[-i-1][j] == -1:
                ax.text(j+0.5, n-i, '×', color='red', fontsize=18, ha='center', va='center')

    # Tracer les arêtes verticales
    for i in range(n):
        for j in range(m+1):
            if C[-i-1][j] == 1:
                ax.plot([j, j], [n-i, n-(i+1)], 'k', linewidth=4)
            elif C[-i-1][j] == -1:
                ax.text(j, n-i-0.5, '×', color='red', fontsize=18, ha='center', va='center')

    ax.set_xlim(-0.5, m+1-0.5)
    ax.set_ylim(-0.5, n+1-0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()
