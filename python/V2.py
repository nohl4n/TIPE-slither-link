import matplotlib.pyplot as plt
import numpy as np
import random as r
import math as m



def init_Map(n,m):
    L =[[True]*m]*n
    M = np.array(L)
    return M

#_______________________________METHODE_RECURSIVE__________________________________

def grignotage_rec(n,m):

    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0,n,0,m)
    b=len(B)
    r.shuffle(B)
    a = r.randint(1,b-1)
    
    for k in range(a):
        if Breakable(M,B[k][0],B[k][1]):
            generate_rec(M,B[k][0],B[k][1],n*m)
    
    return M

def generate_rec (M,i,j,n):
    if n>0:
    
        M[i][j] = False
    
        C = Cross_ind(M,i,j)
        r.shuffle(C)
        c = len (C)

    
        for k in range(r.randint(1,c-1)):
            if Breakable(M,C[k][0],C[k][1]):
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

def Breakable(M,i,j):

    C = Cycle(M,i,j)
    n = len(C)
    precedent=C[0]
    var = 0
    
    for k in range(1,n+1):
        if (precedent and (not C[k%n])) or ((not precedent) and C[k%n]):
            var +=1
            
        precedent = C[k%n]
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
        if Breakable(M,B[i][0],B[i][1]):
            M[B[i][0]][B[i][1]] = False
    p = min(n,m)
    
    for k in range(1,p//2):
        B= bordure(k,n-k,k,m-k)
        r.shuffle(B)
        b=len(B)
        for l in range(r.randint(4*b//5,b-1)):
            if Breakable(M,B[l][0],B[l][1]):
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
            if Breakable(M,w[0],w[1]):
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
            if Breakable(M,w[0],w[1]):
                M[w[0]][w[1]]= False
                

    return M
    
#___________________________________________________________________________


def fissure(n,m):
    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0,n,0,m)
    
    r.shuffle(B)
    M[B[0][0]][B[0][1]] = False
    
#___________________Valeur_Carré_____________________________________________

def Nombre_arrête(M):
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
                        
def show_NA(NA):
    
    m,n = len(NA[0]),len(NA)
    fig, ax = plt.subplots(figsize=(n,m))

    # Tracer les points
    for i in range(n+1):
        for j in range(m+1):
            ax.plot(j, n-i, 'ko', markersize=8)

    # Ajouter les indices
    for i in range(n):
        for j in range(m):
            if NA[-i-1][j] is not None:
                ax.text(j+0.5, n-i-0.5, str(NA[-i-1][j]),
                        fontsize=16, ha='center', va='center')


    ax.set_xlim(-0.5, m+1-0.5)
    ax.set_ylim(-0.5, n+1-0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    
    fig, ax = plt.subplots()
    img = ax.imshow(M)
    
    plt.show()
    
    
    
    

def show_M (M):
    fig, ax = plt.subplots()
    img = ax.imshow(M)
    plt.show()
    
def show(M,NA):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    img1 = ax1.imshow(M)
    img2 = ax2.imshow(NA)
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt

def plot_M_and_NA(M, NA):
    """
    Affiche une grille où :
    - la couleur de fond est déterminée par la matrice booléenne `M`
    - les valeurs de 0 à 3 sont affichées en texte dans chaque cellule selon `NA`
    
    Paramètres :
    - M : np.array booléen de forme (m, n)
    - NA : np.array de même forme (m, n) contenant 0–3, None ou np.nan
    """
    if M.shape != NA.shape:
        raise ValueError("Les matrices M et NA doivent avoir la même forme.")
    
    fig, ax = plt.subplots()

    # Affichage de M en couleur (True -> rouge, False -> blanc)
    ax.imshow(M, cmap=plt.cm.Reds, vmin=0, vmax=1)

    # Affichage des valeurs de NA dans la grille
    for i in range(NA.shape[0]):
        for j in range(NA.shape[1]):
            val = NA[i, j]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ax.text(j, i, str(int(val)), va='center', ha='center', color='black', fontsize=14)

    # Personnalisation des axes et de la grille
    ax.set_xticks(np.arange(NA.shape[1]))
    ax.set_yticks(np.arange(NA.shape[0]))
    ax.set_xticks(np.arange(-.5, NA.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, NA.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.show()
    
Bite = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0]])
 
def test():
    img   = plt.imread('matrice_d_amour.png')
    img   = img[:, :, 0]
    G = [[False]*100]*100
    
    for i in range(100):
        for j in range(100):
            if img[i][j] == 1:
                G[i][j] = True
            else:      
                G[i][j]= False
    return G
