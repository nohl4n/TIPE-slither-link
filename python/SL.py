"""
====================
| Slitherlink GAME |
====================

Un programme complet pour générer,  résoudre et visualiser des puzzles Slitherlink.
Le Slitherlink est un puzzle logique où le but est de connecter des points pour former
une boucle unique fermée,  avec des indices numériques indiquant le nombre d'arêtes
à dessiner autour de chaque cellule.

Table des matières :

    I - Generation_puzzle
        a) Grignotage recursif
        b) bruit chemin
        c) Grignotage non recursif
        d) Generation type Labyrinthe
        
    II - Etude de dénombrement
        
    III - Verifier_puzzle
        a) Règles
        b) Règle equivalente en jeux 2 colorables
        
    VI - Solve_puzzle
        a) Brut_force (a faire)
        b) 
    
    V - Show_puzzle
    
    
Auteur: SAUCET Nohlan
PROJET TIPE 2025 - 2026
"""


# _____________BIBLIOTHEQUE

import matplotlib.pyplot as plt
import numpy as np
import random as r
import copy as c
import sys

#_____________AUGMENTE_LA_RECURSION_DEPTH

sys.setrecursionlimit(10000)


"""
~~~~~~~~~~~~~~~~~~~
I - Generation_puzzle
~~~~~~~~~~~~~~~~~~~
"""




#__________________Grignotage_récursif
    
def Croix(M,i,j):
    
    n,m = len(M) , len(M[0])

    if i==0 and j==0:
        C =np.array([M[i][j+1],M[i+1][j],False,False])
                    
    elif i==n-1 and j==0:
        C =np.array([M[i-1][j],M[i][j+1],False,False])
        
    elif i==0 and j==m-1:
        C =np.array([M[i][j-1],M[i+1][j],False,False])
                              
    elif i==n-1 and j==m-1:
        C =np.array([M[i][j-1],M[i-1][j],False,False])
                    
    elif i==0:
        C =np.array([M[i][j-1],M[i+1][j],M[i][j+1],False])
                    
    elif j==0:

        C =np.array([M[i-1][j],M[i][j+1],M[i+1][j],False])
                    
    elif i==n-1:
        C =np.array([M[i][j-1],M[i-1][j],M[i][j+1],False])
                    
    elif j==m-1:
        C =np.array([M[i-1][j],M[i][j-1],M[i+1][j],False])
    
    else:
        C =np.array([M[i-1][j],M[i][j+1],M[i+1][j],M[i][j-1]])
        
    return C
    
def Contour(M,i,j):

    n,m = len(M) , len(M[0])
    
    if i==0 and j==0:
        C =np.array([False,False,False,M[i][j+1],M[i+1][j+1],M[i+1][j],False,False])
                    
    elif i==n-1 and j==0:
        C =np.array([False,M[i-1][j],M[i-1][j+1],M[i][j+1],False,False,False,False])
        
    elif i==0 and j==m-1:
        C =np.array([False,False,False,False,False,M[i][j-1],M[i+1][j-1],M[i+1][j]])
                              
    elif i==n-1 and j==m-1:
        C =np.array([M[i-1][j-1],M[i-1][j],False,False,False,False,False,M[i][j-1]])
                    
    elif i==0:
        C =np.array([False,False,False,M[i][j-1],M[i+1][j-1],M[i+1][j],M[i+1][j+1],M[i][j+1]])
                    
    elif j==0:
        C =np.array([False,M[i-1][j],M[i-1][j+1],M[i][j+1],M[i+1][j+1],M[i+1][j],False,False])
                    
    elif i==n-1:
        C =np.array([M[i-1][j-1],M[i-1][j],M[i-1][j+1],M[i][j+1],False,False,False,M[i][j-1]])
                    
    elif j==m-1:
        C =np.array([M[i-1][j-1],M[i-1][j],False,False,False,M[i+1][j],M[i+1][j-1],M[i][j-1]])
    
    else:
        C =np.array([M[i-1][j-1],M[i-1][j],M[i-1][j+1],M[i][j+1],M[i+1][j+1], M[i+1][j], M[i+1][j-1],M[i][j-1]])
        
    return C
    
def modifiable(M,i,j):
    """Verifie si une case peut être à la fois exterieur et interieur"""
    
    Contour_M = Contour(M,i,j)
    Croix_M = Croix(M,i,j)
    
    # la case est connecté à l'interieur et à l'exterieur
    bool1 = (False in Croix_M and True in Croix_M)
    
    precedent = Contour_M[7]
    var = 0
    
    for k in range(8):
        if precedent != Contour_M[k]:
            var +=1
        precedent = Contour_M[k]
    
    # le contour n'as qu'une seule variation de couleur
    bool2 = (var == 2)
    
    return bool1 and bool2

def grignotage_rec(n, m):

    """
    Génère un puzzle Slitherlink par grignotage récursif.
        
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
            
    Returns:
    np.array: Matrice booléenne représentant le puzzle
    """

    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0, n, 0, m)
    b=len(B)
    r.shuffle(B)
    a = r.randint(1, b - 1)
    
    for k in range(a):
        if modifiable(M, B[k][0], B[k][1]):
            generate_rec(M, B[k][0], B[k][1], n*m//2)
    
    return M
    
def grignotage_rec_centre(n, m):

    """
    Autre méthode de grignotage récursif commençant au centre.
        
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        
    Returns:
        np.array: Matrice booléenne représentant le puzzle
    """

    L =[[True]*m]*n
    M = np.array(L)
    generate_rec(M, n//2, m//2, (n*m)//2)
    
    return M
    
def inverse(M):
    """
    Inverse les valeurs booléennes d'une matrice.
        
    Args:
        M (np.array): Matrice à inverser
        
    Returns:
        np.array: Matrice inversée
    """
    n, m = len(M) ,  len(M[0])
    for i in range(n):
        for j in range(m):
            if M[i][j]:
                M[i][j] = False
            else:
                M[i][j] = True

def generate_rec (M, i, j, n):

    """
    Fonction récursive pour le grignotage.
    
    Args:
        M (np.array): Matrice du puzzle
        i (int): Index de ligne
        j (int): Index de colonne
        n (int): Nombre d'itérations restantes
    """

    if n>0:
    
        M[i][j] = not(M[i][j])
    
        C = Croix_ind(M, i, j)
        r.shuffle(C)
        c = len (C)

    
        for k in range(r.randint(1, c - 1)):
            if modifiable(M, C[k][0], C[k][1]):
                generate_rec(M, C[k][0], C[k][1], n - 1)
                
def generate_rec_list(M,i,j,L,n):
    
    l = len(L)

    if n > 0 and l > 0 :
    
        ind = r.randint(0, l - 1)
        L[-1], L[ind] = L[ind], L[-1] 
        i, j = L[ind][0], L[ind][1]
        
        


def Croix_ind(M, i, j):

    """
    Retourne les indices des cellules voisines (en croix).
    
    Args:
        M (np.array): Matrice du puzzle
        i (int): Index de ligne
        j (int): Index de colonne
        
    Returns:
        list: Liste des indices des voisins
    """
    
    n, m = len(M) ,  len(M[0])

    if i==0 and j==0:
        C =[[i, j + 1], [i + 1, j]]
                    
    elif i==n - 1 and j==0:
        C =[[i - 1, j], [i, j + 1]]
        
    elif i==0 and j==m - 1:
        C =[[i, j - 1], [i + 1, j]]
                              
    elif i==n - 1 and j==m - 1:
        C =[[i, j - 1], [i - 1, j]]
                    
    elif i==0:
        C = [[i, j - 1], [i + 1, j], [i, j + 1]]
                    
    elif j==0:

        C =[[i - 1, j], [i, j + 1], [i + 1, j]]
                    
    elif i==n - 1:
        C =[[i, j - 1], [i - 1, j], [i, j + 1]]
                    
    elif j==m - 1:
        C =[[i - 1, j], [i, j - 1], [i + 1, j]]
    
    else:
        C = [[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]]
        
    return C

#_______________Bruit_Chemins

def bruit (n, m, r):

    """
    Génère du bruit aléatoire dans une matrice.
    
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        r (float): Intensité du bruit
        
    Returns:
        tuple: (Matrice bruitée,  Historique des positions modifiées)
    """
    
    L = [[False]*m]*n
    M = np.array(L)
    H = []
    
    for _ in range (int((n + m)*r)):
        i, j = random_case (n,  m)
        H.append([i,  j])
        M[i][j] = True
    
    return M, H

def sign(x):
    """Retourne le signe d'un nombre."""
    return  - 1 if x < 0 else 1
        

def chemin (i1,  j1,  i2,  j2):

    """
    Génère un chemin entre deux points.
    
    Args:
        i1,  j1 (int): Point de départ
        i2,  j2 (int): Point d'arrivée
        
    Returns:
        list: Liste des points du chemin
    """
    
    W=[]
    i,  j = i1,  j1
    
    for _ in range (abs(i1  -  i2)  +  abs(j1  -  j2)):
        s = r.randint(0,  1)
        if s == 1:
            if i != i2:
                i  += sign(i2  -  i)
            
            else:
                j  += sign(j2  -  j)
        
        else:
            if j != j2:
                j  += sign(j2  -  j)
            
            else:
                i  += sign(i2  -  i)
        
        W.append([i, j])
    
    return W
    
#____tRAveEau________
def chemin_complexe (i1,  j1,  i2,  j2,  n,  m):

    """
    Génère un chemin plus complexe entre deux points.
    
    Args:
        i1,  j1 (int): Point de départ
        i2,  j2 (int): Point d'arrivée
        n,  m (int): Dimensions de la grille
        
    Returns:
        list: Liste des points du chemin
    """
    
    W = []
    i,  j = i1,  j1
    
    while i != i2 and j != j2 :
        di = abs(i2  -  i)
        dj = abs(j2  -  j)
        a = 1
        b = 0.7
        p = r.random()
        
        if p <= (a  +  dj) / (di  +  dj  +  2*a):
            p = r.random()
            if p <= b:
                if ((sign(j2  -  j) == 1 and j < m - 1) or 
                    (sign(j2  -  j) ==  - 1 and j > 0)):
                    j += sign(j2 - j)
                else:
                    j -= sign(j2 - j)
            else:
                if ((sign(j2  -  j) == 1 and j > 0) or 
                    (sign(j2  -  j) ==  - 1 and j < m - 1)):
                    j  -= sign(j2 - j)
                else:
                    j  += sign(j2 - j)
            
        else:
            p = r.random()
            if p <= b:
                if ((sign(i2  -  i) == 1 and i < n - 1) or
                    (sign(i2  -  i) ==  - 1 and i > 0)):
                    i  += sign(i2  -  i)
                else:
                    i  -= sign(i2  -  i)
            else:
                if ((sign(i2  -  i) == 1 and i > 0) or
                    (sign(i2  -  i) ==  - 1 and i < n - 1)):
                    i  -= sign(i2  -  i)
                else:
                    i  += sign(i2  -  i)
        W.append([i,  j])
    
    return W
    

    
def generate_bruit_chemin(n,  m,  r):

    """
    Génère un puzzle avec la méthode bruit  +  chemin.
        
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        r (float): Intensité du bruit
            
    Returns:
        np.array: Matrice du puzzle
    """
    
    M, H = bruit(n,  m,  r)
    h = len(H)
    for k in range(h - 1):
        i1,  j1 = H[k][0],  H[k][1]
        i2,  j2 = H[k  +  1][0],  H[k  +  1][1]
        W = chemin(i1,  j1,  i2,  j2)
        for c in W:
            M[c[0]][c[1]] = True
        
    return M
    
def generate_chemins(n, m):

    """
    Génère des chemins sur un puzzle.
    
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        
    Returns:
        np.array: Matrice du puzzle
    """
    
    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0, n, 0, m)
    
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1, b//5)):
        i, j = r.randint(1, n - 1),  r.randint(1, m - 1)
        W = chemin (B[k][0], B[k][1], i, j)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if modifiable(M, w[0], w[1]):
                M[w[0]][w[1]]= False
                

    return M
    
def generate_chemins_complexes(n, m):
    L =[[True]*m]*n
    M = np.array(L)
    B= bordure(0, n, 0, m)
    
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1, b//5)):
        i, j = r.randint(1, n - 1),  r.randint(1, m - 1)
        W = chemin_complexe (B[k][0], B[k][1], i, j, n, m)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if modifiable(M, w[0], w[1]):
                M[w[0]][w[1]]= False
                

    return M
    
#3_________________METHODE_GRIGNOTAGE_CARRE

def bordure(nd, nf, md, mf):

    """
    Génère les indices des cellules de bordure.
    
    Args:
        nd, nf (int): Début et fin des lignes
        md, mf (int): Début et fin des colonnes
        
    Returns:
        list: Liste des indices de bordure
    """
    
    B=[]
    for j in range(md, mf):
        B.append([nd, j])
    for i in range(nd + 1, nf):
        B.append([i, mf - 1])
    for j in range(mf - 2, md - 1,  - 1):
        B.append([nf - 1, j])
    for i in range(nf - 1, nd - 1,  - 1):
        B.append([i, md])
    return B

def grignotage_carré(M, n, m):

    """
    Génère un puzzle par grignotage carré.
        
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        M (np.array, optional): Matrice existante
        
    Returns:
        np.array: Matrice du puzzle
    """

#    L =[[True]*m]*n
#    M = np.array(L)
    B= bordure(0, n, 0, m)
    b=len(B)
    r.shuffle(B)
    
    for i in range(r.randint(0, b - 1)):
        if modifiable(M, B[i][0], B[i][1]):
            M[B[i][0]][B[i][1]] = False
    p = min(n, m)
    
    for k in range(1, p//2):
        B= bordure(k, n - k, k, m - k)
        r.shuffle(B)
        b=len(B)
        for l in range(r.randint(4*b//5, b - 1)):
            if modifiable(M, B[l][0], B[l][1]):
                M[B[l][0]][B[l][1]] = False
    
    return M

#______________Methode_Maze

def init_maze(n, m):

    """
    Initialise un labyrinthe pour la génération.
    
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        
    Returns:
        tuple: (Matrice du labyrinthe, Historique des positions)
    """
    
    M=[[0]*m]*n
    M = np.array(M)
    Hist = []
    compteur = 1
    
    for i in range(1, n, 2):
        for j in range(1, m, 2):
            M[i][j] = compteur
            Hist.append([i, j])
            compteur  += 1
            
    return M, Hist
    
def init_maze2(n, m, proba=0.1):

    """
    Initialise un labyrinthe avec une probabilité donnée.
    
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        proba (float): Probabilité d'activation
        
    Returns:
        list: Matrice du labyrinthe
    """
    
    M=[[0] * (2 * m)] * (2 * n)
    M=np.array(M)
    
    for i in range(1, 2*n, 2):
        for j in range(1, 2*m, 2):
            rand = r.random()
            if rand < proba:
                M[i][j] = 1
                
    M = M.tolist()
    print(M)
                
    for k in range(2*n, n,  - 1):
        supr_l(M, r.randint(0, k - 1))
        show(M)
        
    for k in range(2*m, m,  - 1):
        supr_col(M, r.randint(0, k - 1))
        show(M)
            
    return M
    
def Histo(M):

    """
    Crée un historique des positions non nulles.
    
    Args:
        M (np.array): Matrice à analyser
        
    Returns:
        list: Historique des positions
    """
    
    n, m = len(M), len(M[0])
    Hist = []
    
    for i in range(n):
        for j in range(m):
            if M[i][j] != 0:
                Hist.append([i, j])
                
    return Hist    
    
    
    
def supr_col(M, ind):
    """
    Supprime une colonne de la matrice.
    
    Args:
        M (list): Matrice
        ind (int): Index de la colonne à supprimer
        
    Returns:
        list: Matrice modifiée
    """
    
    n, m = len(M), len(M[0])
    
    for i in range(n):
        for j in range(ind, m - 1):
        
            M[i][j] ,  M[i][j + 1] = M[i][j + 1] ,  M[i][j]
            
    for i in range(n):
        print(M)
        M[i].pop()

   
    return M
    
def supr_l(M, ind):
    """
    Supprime une ligne de la matrice.
    
    Args:
        M (list): Matrice
        ind (int): Index de la ligne à supprimer
        
    Returns:
        list: Matrice modifiée
    """
    
    n= len(M)
    
    for i in range(ind, n - 1):
        
        M[i], M[i + 1] = M[i + 1], M[i]
    
    M.pop()
                
    return M
    
    
    
    
def link(M, i, j):
    """
    Trouve les liens possibles depuis une cellule.
    
    Args:
        M (np.array): Matrice du labyrinthe
        i, j (int): Position de la cellule
        
    Returns:
        list: Liste des liens possibles
    """
    
    n, m= len(M), len(M[0])
    L = []
    val = M[i][j]
    
    if M[i][j]  +  M[(i + 2)%n][j] != val and M[i][j]  +  M[(i + 2)%n][j] < 2*val:
        L.append([i + 2, j])
        
    if M[i][j]  +  M[(i - 2)%n][j] != val and M[i][j]  +  M[(i - 2)%n][j] < 2*val:
        L.append([i - 2, j])
    
    if M[i][j]  +  M[i][(j + 2)%m] != val and M[i][j]  +  M[i][(j + 2)%m] < 2*val:
        L.append([i, j + 2])
    
    if M[i][j]  +  M[i][(j - 2)%m] != val and M[i][j]  +  M[i][(j - 2)%m] < 2*val:
        L.append([i, j - 2])
        
    return L
    
def generate_maze(n, m):

    """
    Génère un puzzle avec la méthode du labyrinthe.
    
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        
    Returns:
        np.array: Matrice du puzzle
    """
    
    M, Hist= init_maze(n, m)

    h = len(Hist)
    r.shuffle(Hist)
    S=[]
    
    while h != 0:
        ind = r.randint(0, h - 1)
        Hist[ind][0], Hist[ - 1][0] = Hist[ - 1][0], Hist[ind][0]
        Hist[ind][1], Hist[ - 1][1] = Hist[ - 1][1], Hist[ind][1]
        i, j = Hist[ - 1][0], Hist[ - 1][1]
        L = link(M, i, j)
        
        if len(L) == 0:
            Hist.pop()
            h -=1
            
        else:
            r.shuffle(L)
            
            i_p ,  j_p = L[0][0], L[0][1]
            val = M[i][j]
            Hist.append([(i + i_p)//2, (j + j_p)//2])
            h +=1
            M[(i + i_p)//2][(j + j_p)//2] = val
            
            homog(M, i_p, j_p, val)

            
    return M
    
def homog(M, i, j, val):

    """
    Uniformise les valeurs dans une zone connectée.
    
    Args:
        M (np.array): Matrice à modifier
        i, j (int): Position de départ
        val: Valeur à propager
    """
    
    M[i][j] = val
    Croix = Croix_ind(M, i, j)
    for i_p, j_p in Croix:
        if M[i_p][j_p] != 0 and M[i_p][j_p] != val:
            homog(M, i_p, j_p, val)
            
#________________Methode_Opti

def generate_test(n,m):
    i_mid = n//2
    j_mid = m//2
    
    L =[[False]*m for _ in range(n)]
    M = np.array(L)
    
    for i in range(n):
        for j in range(m):
            if abs(i - i_mid) < n / 4*(2**0.5) and abs(j - j_mid) < m / 4*(2**0.5):
                M[i][j] = True
                
    L = List_modifiable(M)
    
    for _ in range(n*m):
        l = len(L)
        
        if l != 0:
            ind = r.randint(0, l - 1)
            #L[-1],L[ind] = L[ind], L[-1]
            i, j = L[ind][0], L[ind][1]
            
            if modifiable(M,i,j):
                M[i][j] = not(M[i][j])
                
        L = List_modifiable(M)
    
    
    return M
    
def List_modifiable(M):
    """renvoie une liste des indices des cases modifiable"""
    n,m = M.shape
    L=[]
    
    for i in range(n):
        for j in range(m):
            if modifiable(M,i,j):
                L.append([i,j])
    
    return L

"""
~~~~~~~~~~~~~~~~~~~
II - Etudes Dénombrement
~~~~~~~~~~~~~~~~~~~
"""

def List_3x3():
    """Génère toutes les matrices 3x3 possibles."""

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
    """Génère toutes les matrices n x m possibles."""
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
    
#____________Filtres
    
def filtre_equiv(L):
    """Filtre les matrices équivalentes par rotation/symétrie."""
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
    """Filtre les matrices valides pour le Slitherlink."""
    L_filtre =[]
    for M in L:
        if Verif(M):
            L_filtre.append(M)
    return L_filtre
    
def filtre_diag(L):
    """Filtre les matrices sans connexions diagonales."""
    L_filtre =[]
    for M in L:
        if in_pas_diag(M):
            L_filtre.append(M)
    return L_filtre
    
def filtre_Point(L,marge):
    """Filtre les matrices selon un critère de points."""
    L_filtre =[]
    for M in L:
        if Point(M) <= marge:
            L_filtre.append(M)
    return L_filtre
    
def filtre_nombre_modifiable(L,mini,maxi):
    L_filtre = []
    for M in L:
        nb_modif = len(List_modifiable(M))
        if nb_modif >= mini and nb_modif <= maxi:
            L_filtre.append(M)
            
    return L_filtre
    
#___________Test_de_bon_retour_pour_les_critères

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

def List_3x3_able():
    """Génère les matrices 3x3 valides pour le grignotage."""
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

        is_Croix = (s[0]==s[1]) or (s[0]==s[3]) or (s[0]==s[5]) or (s[0]==s[7])
        
        if var <= 2 and is_Croix :
            L.append(M)
    
    return L
    
def List_3x3_puzzle():
    """Génère les matrices 3x3 valides pour le puzzle."""

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
    """Compte le nombre de True dans une matrice."""
    n,m = len(M),len(M[0])
    nb = 0
    
    for i in range(n):
        for j in range(m):
            if M[i,j]:
                nb +=1
                
    return nb
    
def List_in(L):
    """Compte le nombre de True pour chaque matrice dans une liste."""
    L_in = []
    for M in L:
        L_in.append(nb_in(M))
        
    return L_in
    
def Sort_nb_in(L,L_in):
    """Trie une liste de matrices selon le nombre de cases interieurs"""
    n= len(L)
    maxi = max(L_in)
    
    L_sort = []
    
    for i in range(maxi+1):
        for j in range (n):
            if L_in[j] == i:
                L_sort.append(L[j])
        
    return np.array(L_sort)

def Sort_nb_modif(L,mini,maxi):
    """Trie une liste de matrices selon le nombre de cases interieurs"""
    n= len(L)
    maxi = 16
    
    L_sort = []
    
    for i in range(mini,maxi+1):
        for j in range (n):
            if len(List_modifiable(L[j])) == i:
                L_sort.append(L[j])
        
    return np.array(L_sort)

    
def Tri(L):
    """Trie une liste de matrices."""
    return Sort(L,List_in(L))
    

def Ligne(M,i):
    """Retourne une ligne d'une matrice."""
    return M[i]

def Colonne(M,j):
    """Retourne une colonne d'une matrice."""
    n= len(M)
    C=[]
    for i in range(n):
        C.append(M[i][j])
        
    return C
    
def Moyenne(L,M):
    """Calcule la moyenne pondérée d'une liste de booléens."""
    n= len(M)
    moy=0
    for Bool in L:
        if Bool:
            moy+=1/n
        else:
            moy-=1/n
    
    return moy
    
def Point(M):
    """Calcule un score de points pour une matrice."""
    n,m = len(M),len(M[0])
        
    pts = 0
    
    for i in range(n):
        pts += abs((Moyenne(Ligne(M,i),M))**2)
    
    for j in range(m):
        pts += abs((Moyenne(Colonne(M,j),M))**2)
        
    pts = pts/(n+m)
    
    return pts
    

    
"""
~~~~~~~~~~~~~~~~~~~
III - Verifier puzzle
~~~~~~~~~~~~~~~~~~~
"""

#___________Règles

def Rule1(M,NA):

    """
    Vérifie la règle 1 du Slitherlink.
    
    Args:
        M (np.array): Matrice du puzzle
        NA (np.array): Matrice des nombres attendus
        
    Returns:
        bool: True si la règle est respectée
    """
    
    n,m = len(M) , len(M[0])
    i=0
    res = True
    
    while i<n and res:
        j=0
        while j<m and res:
            C = Croix(M,i,j)
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
    C = Croix(M,i,j)
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

    """
    Calcule la matrice des nombres d'arrête.
    
    Args:
        M (np.array): Matrice du puzzle
        
    Returns:
        np.array: Matrice des nombres attendus
    """
    
    n,m = len(M) , len(M[0])
    L =[[0]*m]*n
    NA = np.array(L)
    
    for i in range (n):
        for j in range(m):
            C = Croix(M,i,j)
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
    
#_______________Règles équivalentes 2 couleurs

def Hist(M,i,j,L):

    """
    Crée un historique des cellules connectées.
    
    Args:
        M (np.array): Matrice du puzzle
        i, j (int): Position de départ
        L (list): Liste pour stocker l'historique
    """

    L.append([i,j])
    C = Croix_ind(M,i,j)
    
    for ind in C:
        if M[ind[0]][ind[1]] == M[i][j] and not ([ind[0],ind[1]] in L):
            Hist(M,ind[0],ind[1],L)

    
def Couche_ext(M):

    """
    Crée une couche externe pour la vérification.
    
    Args:
        M (np.array): Matrice du puzzle
        
    Returns:
        np.array: Matrice de la couche externe
    """

    n,m = len(M),len(M[0])
    Couche = [[False]*(m+2)]*(n+2)
    Couche = np.array(Couche)
    
    for i in range(n+2):
        for j in range(m+2):
            if i>0 and i<=n and j>0 and j<=m:
                Couche[i][j] = M[i-1][j-1]
    
    return Couche
    
def Verif(M):

    """
    Vérifie la validité du puzzle 2 colorable.
    ie : 1) interieur convexe
         2) exterieur + couche externe convexe
         3) forme une partition de l'ensemble
    
    Args:
        M (np.array): Matrice du puzzle
        
    Returns:
        bool: True si le puzzle est valide
    """

    n,m = len(M),len(M[0])

    i,j=0,0
    found = False
    
    while i<n and not found: #trouver une case à l'interieur
        j=0
        while j<m and not found:
            if M[i,j]:
                found = True
                i,j = i-1,j-1
            j+=1
        i+=1

    if found == False: #si pas trouvé matrice nulle
        return True
    else:
        
        # Histogramme des cases interieurs
        L_int=[]
        Hist(M,i,j,L_int)
        
        # Histogramme des cases exterieurs
        L_ext=[]
        Couche= Couche_ext(M)
        Hist(Couche,0,0,L_ext)
        
        return len(L_int) + len(L_ext) == (n+2)*(m+2)
   
        
        

"""
~~~~~~~~~~~~~~~~~~~
IV - Show puzzle
~~~~~~~~~~~~~~~~~~~
"""

def show(M):

    """
    Affiche une matrice simple.
    
    Args:
        M (np.array): Matrice à afficher
    """
    
    fig, ax = plt.subplots()
    img = ax.imshow(M)
    plt.show()
    
def show_gris(M):

    """
    Affiche une matrice simple en echelle de gris.
    
    Args:
        M (np.array): Matrice à afficher
    """

    fig, ax = plt.subplots()
    img = ax.imshow(M,cmap='gray')
    plt.show()

def show_2(M,NA):

    """
    Affiche deux matrices côte à côte.
    
    Args:
        M (np.array): Première matrice
        NA (np.array): Deuxième matrice
    """

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    img1 = ax1.imshow(M)
    img2 = ax2.imshow(NA)
    plt.show()
  
def show_n(L,n,m):

    """
    Affiche plusieurs matrices dans une grille.
    
    Args:
        L (list): Liste de matrices
        n, m (int): Dimensions de la grille d'affichage
    """

    l = len(L)
    fig,axs = plt.subplots(n,m)
    
    k=0
    for i in range(n):
        for j in range(m):
            if k<l:
                axs[i,j].imshow(L[k])
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
            
                
            k+=1
    plt.show()
    
def show_anim(L, interval=200):

    """
    Affiche une animation d'une séquence de matrices.
    
    Args:
        L (list): Liste de matrices à animer
        interval (int): Intervalle entre les frames en ms
    """
    
    fig, ax = plt.subplots()
    
    vmin = min(np.min(m) for m in L)
    vmax = max(np.max(m) for m in L)
    
    im = ax.imshow(L[0], vmin=vmin, vmax=vmax)
    
    def update(frame):
        im.set_array(L[frame])
        return [im]
    
    anim = animation.FuncAnimation(fig, update, frames=len(L), 
                                 interval=interval, blit=True, repeat=False)
    
    plt.show()
    return anim
