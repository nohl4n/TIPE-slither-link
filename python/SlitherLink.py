"""
====================
| Slitherlink GAME |
====================

Un programme complet pour générer,  résoudre et visualiser des puzzles Slitherlink.
Le Slitherlink est un puzzle logique où le but est de connecter des points pour former
une boucle unique fermée,  avec des indices numériques indiquant le nombre d'arêtes
à dessiner autour de chaque cellule.

Table des matières :
    
    0 - Fonction utile

    I - Generation_puzzle
        0) Critère de modifiabilité
        a) Grignotage recursif
        b) Grignotage carré
        c)  1.bruit chemin
            2.Amélioration
        d) Generation par mutation
        e) Generation type Labyrinthe
        f) méthode théoriquement optimal
        
    II - Etude de dénombrement
        a) Filtres
        b) Test pour les filtres
        
    III - Verifier_puzzle
        a) Règles
        b) Règle equivalente en 2 couleurs
    
    IV - Fonction pour visualiser des puzzles
        
    V - Solve_puzzle
        a) Brut_force
        b) méthode essaie modifie avec heuristique
        c) Backtracking efficasse
        d) solver z3
    
    VI - Show_puzzle
    
    VI - Test
        a) Comparaison Z3 avec Backtracking
        b) influence de la répartion des cases non indicé dans le temps pour résoudre
    
    VII - Générer Puzzle
    
    
    
Auteur: SAUCET Nohlan
PROJET TIPE 2025 - 2026
"""


# _____________BIBLIOTHEQUE

import matplotlib.pyplot as plt
import numpy as np
import random as r
import copy as c
import time
import sys
import z3


#_____________AUGMENTE_LA_RECURSION_DEPTH

sys.setrecursionlimit(100000)

"""
~~~~~~~~~~~~~~~~~~~
0 - Fonction utile
~~~~~~~~~~~~~~~~~~~
"""

def Croix(M,i,j):

    """
    Retourne la valeur des cellules voisines (en croix).
    """
    
    n,m = len(M) , len(M[0])
    if i==0 and j==0:
        C = [M[i][j+1],M[i+1][j],False,False]
    elif i==n-1 and j==0:
        C = [M[i-1][j],M[i][j+1],False,False]
    elif i==0 and j==m-1:
        C = [M[i][j-1],M[i+1][j],False,False]
    elif i==n-1 and j==m-1:
        C = [M[i][j-1],M[i-1][j],False,False]
    elif i==0:
        C = [M[i][j-1],M[i+1][j],M[i][j+1],False]
    elif j==0:
        C = [M[i-1][j],M[i][j+1],M[i+1][j],False]
    elif i==n-1:
        C = [M[i][j-1],M[i-1][j],M[i][j+1],False]
    elif j==m-1:
        C = [M[i-1][j],M[i][j-1],M[i+1][j],False]
    else:
        C = [M[i-1][j],M[i][j+1],M[i+1][j],M[i][j-1]]
    return C
    
def Croix_ind(M, i, j):

    """
    Retourne les indices des cellules voisines (en croix).
    """
    
    n, m = len(M), len(M[0])
    if i == 0 and j == 0:
        C =[[i, j+1], [i+1, j]]
    elif i == n-1 and j == 0:
        C =[[i-1, j], [i, j+1]]
    elif i == 0 and j == m-1:
        C =[[i, j-1], [i+1, j]]
    elif i == n - 1 and j == m - 1:
        C =[[i, j-1], [i-1, j]]
    elif i == 0:
        C = [[i, j-1], [i+1, j], [i, j+1]]    
    elif j == 0:
        C =[[i-1, j], [i, j+1], [i+1, j]]
    elif i == n-1:
        C =[[i, j-1], [i-1, j], [i, j+1]]
    elif j == m-1:
        C =[[i-1, j], [i, j-1], [i+1, j]]
    else:
        C = [[i-1, j], [i, j+1], [i+1, j], [i, j-1]]
    return C
    
def Contour(M,i,j):
    
    """
    Retourne les 8 cases qui entoure la case d'indice (i,j).
    """
    
    n,m = len(M) , len(M[0])
    if i==0 and j==0:
        C = [False,False,False,M[i][j+1],M[i+1][j+1],M[i+1][j],False,False]
    elif i==n-1 and j==0:
        C = [False,M[i-1][j],M[i-1][j+1],M[i][j+1],False,False,False,False]
    elif i==0 and j==m-1:
        C = [False,False,False,False,False,M[i][j-1],M[i+1][j-1],M[i+1][j]]
    elif i==n-1 and j==m-1:
        C = [M[i-1][j-1],M[i-1][j],False,False,False,False,False,M[i][j-1]]
    elif i==0:
        C = [False,False,False,M[i][j-1],M[i+1][j-1],M[i+1][j],M[i+1][j+1],M[i][j+1]]
    elif j==0:
        C = [False,M[i-1][j],M[i-1][j+1],M[i][j+1],M[i+1][j+1],M[i+1][j],False,False]
    elif i==n-1:
        C = [M[i-1][j-1],M[i-1][j],M[i-1][j+1],M[i][j+1],False,False,False,M[i][j-1]]
    elif j==m-1:
        C = [M[i-1][j-1],M[i-1][j],False,False,False,M[i+1][j],M[i+1][j-1],M[i][j-1]]
    else:
        C = [M[i-1][j-1],M[i-1][j],M[i-1][j+1],M[i][j+1],M[i+1][j+1], M[i+1][j], M[i+1][j-1],M[i][j-1]]
    return C
    
def Contour_ind(M,i,j):

    """
    Retourne l'indices des 8 cases qui entoure la case d'indice (i,j).
    """
    
    n,m = len(M) , len(M[0])
    if i==0 and j==0:
        C =[[i,j+1],[i+1,j+1],[i+1,j]]
    elif i==n-1 and j==0:
        C =[[i-1,j],[i-1,j+1],[i,j+1],False]
    elif i==0 and j==m-1:
        C =[[i,j-1],[i+1,j-1],[i+1,j]]
    elif i==n-1 and j==m-1:
        C =[[i-1,j-1],[i-1,j],[i,j-1]]
    elif i==0:
        C =[[i,j-1],[i+1,j-1],[i+1,j],[i+1,j+1],[i,j+1]]
    elif j==0:
        C =[[i-1,j],[i-1,j+1],[i,j+1],[i+1,j+1],[i+1,j],False]
    elif i==n-1:
        C =[[i-1,j-1],[i-1,j],[i-1,j+1],[i,j+1],[i,j-1]]
    elif j==m-1:
        C =[[i-1,j-1],[i-1,j],[i+1,j],[i+1,j-1],[i,j-1]]
    else:
        C =[[i-1,j-1],[i-1,j],[i-1,j+1],[i,j+1],[i+1,j+1], [i+1,j], [i+1,j-1],[i,j-1]]
    return C


"""
~~~~~~~~~~~~~~~~~~~
I - Generation_puzzle
~~~~~~~~~~~~~~~~~~~
"""

#_________________Critère de modifiabilité d'une case

def modifiable(M,i,j):
    """Verifie si une case peut être à la fois exterieur et interieur"""
    
    Contour_M = Contour(M,i,j)
    Croix_M = Croix(M,i,j)
    
    # la case est connecté à l'interieur et à l'exterieur
    bool1 = (False in Croix_M and True in Croix_M)
    
    # le contour n'as qu'une seule variation de couleur
    precedent = Contour_M[7]
    var = 0
    
    for k in range(8):
        if precedent != Contour_M[k]:
            var +=1
        precedent = Contour_M[k]
    bool2 = (var == 2)
    
    return bool1 and bool2
    
#1_________________METHODE_GRIGNOTAGE_CARRE

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
    for i in range(nd+1, nf):
        B.append([i, mf-1])
    for j in range(mf-2, md-1, -1):
        B.append([nf-1, j])
    for i in range(nf-1, nd-1, -1):
        B.append([i, md])
    return B

def grignotage_carré(M, n, m, proba = 0.8):

    """
    Génère un puzzle par grignotage carré.

    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        M (list of list, optional): Matrice existante
        
    Returns:
        list of list: Matrice du puzzle
    """
    
    #M =[[True]*m for _ in range(n)]
    
    H= []
    H.append(c.deepcopy(M))
    
    B= bordure(0, n, 0, m)
    b=len(B)
    r.shuffle(B)
    
    for i in range(r.randint(0, b - 1)):
        if modifiable(M, B[i][0], B[i][1]):
            M[B[i][0]][B[i][1]] = False
            
    H.append(c.deepcopy(M))
    p = min(n, m)
    
    for k in range(1, p//2):
        B= bordure(k, n - k, k, m - k)
        r.shuffle(B)
        b=len(B)
        for l in range(r.randint(int(proba*b), b)):
            if modifiable(M, B[l][0], B[l][1]):
                M[B[l][0]][B[l][1]] = False
                
        H.append(c.deepcopy(M))
    
    return M
    

#_______________Generation_par_Chemins

def sign(x):
    """Retourne le signe d'un nombre."""
    return -1 if x < 0 else 1
        

def chemin (i1, j1, i2, j2):
    """
    Génère un chemin entre deux points.
    
    Args:
        i1,  j1 (int): Point de départ
        i2,  j2 (int): Point d'arrivée
        
    Returns:
        list: Liste des points du chemin
    """
    
    W=[]
    i, j = i1, j1
    
    for _ in range (abs(i1 - i2)  +  abs(j1 - j2)):
        s = r.randint(0,  1)
        if s == 1:
            if i != i2:
                i += sign(i2-i)
            else:
                j += sign(j2-j)
        else:
            if j != j2:
                j += sign(j2-j)
            else:
                i += sign(i2-i)
        W.append([i, j])
    return W
    
def generate_chemins(n, m):
    """
    Génère des fissures sur une solution de taille n*m
    """
    
    #initialisation
    M =[[True]*m for _ in range(n)]
    B= bordure(0, n, 0, m)
    
    # choix de l'ordre des cases parcouruent sur le bord
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1, b//5)):
        #choix du point d'arrivé de la fissure
        i, j = r.randint(1, n - 1),  r.randint(1, m - 1)
        #création d'une fissure d'un point du bord au point choisi
        W = chemin (B[k][0], B[k][1], i, j)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if modifiable(M, w[0], w[1]):
                M[w[0]][w[1]]= False

    return M
    
#________Amélioration

def chemin_complexe (i1, j1, i2, j2, n, m):

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
    i, j = i1, j1
    
    while i != i2 and j != j2 :
        di = abs(i2-i)
        dj = abs(j2-j)
        a = 1
        b = 0.7
        p = r.random()
        
        if p <= (a+dj) / (di+dj+2*a):
            p = r.random()
            if p <= b:
                if ((sign(j2-j) == 1 and j < m-1) or 
                    (sign(j2-j) ==  -1 and j > 0)):
                    j += sign(j2-j)
                else:
                    j -= sign(j2-j)
            else:
                if ((sign(j2-j) == 1 and j > 0) or 
                    (sign(j2-j) == -1 and j < m-1)):
                    j -= sign(j2-j)
                else:
                    j += sign(j2-j)
            
        else:
            p = r.random()
            if p <= b:
                if ((sign(i2-i) == 1 and i < n-1) or
                    (sign(i2-i) ==  - 1 and i > 0)):
                    i += sign(i2-i)
                else:
                    i -= sign(i2-i)
            else:
                if ((sign(i2-i) == 1 and i > 0) or
                    (sign(i2-i) == - 1 and i < n-1)):
                    i -= sign(i2-i)
                else:
                    i += sign(i2-i)
        W.append([i, j])
    
    return W
    
def generate_chemins_complexes(n, m):
    """
    Génère des fissures sur une solution de taille n*m
    """
    
    #initialisation
    M =[[True]*m for _ in range(n)]
    B= bordure(0, n, 0, m)
    
    # choix de l'ordre des cases parcouruent sur le bord
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1, b//5)):
        #choix du point d'arrivé de la fissure
        i, j = r.randint(1, n - 1),  r.randint(1, m - 1)
        #création d'une fissure d'un point du bord au point choisi
        W = chemin_complexe (B[k][0], B[k][1], i, j, n, m)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if modifiable(M, w[0], w[1]):
                M[w[0]][w[1]]= False

    return M
    
#__________________Grignotage_récursif

def grignotage_rec(n, m, N = -1):

    """
    Génère un puzzle Slitherlink par grignotage récursif.
        
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
            
    Returns:
    list of list : Matrice booléenne représentant le puzzle
    """
    if N == -1:
        N = n*m
    
    
    M =[[True]*m for _ in range(n)]
    B= bordure(0, n, 0, m)
    b=len(B)
    r.shuffle(B)
    a = r.randint(1, b - 1)
    
    
    for k in range(a):
        if modifiable(M, B[k][0], B[k][1]):
            generate_rec(M, B[k][0], B[k][1], N)
    
    return M
    
def grignotage_rec_centre(n, m):

    """
    Autre méthode de grignotage récursif commençant au centre.
        
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        
    Returns:
        list of list : Matrice booléenne représentant le puzzle
    """

    M =[[True]*m for _ in range(n)]
    generate_rec(M, n//2, m//2, (n*m)//2)
    
    return M
    
def inverse(M):
    """
    Inverse les valeurs booléennes d'une matrice.
        
    Args:
        M : Matrice à inverser
        
    Returns:
        list of list : Matrice inversée
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
        M (list of list): Matrice du puzzle
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
    M = [[0]*m for _ in range(n)]
    Hist = []
    compteur = 1
    
    for i in range(1, n, 2):
        for j in range(1, m, 2):
            M[i][j] = compteur
            Hist.append([i, j])
            compteur  += 1
            
    return [M, Hist]
    
def Histo(M):

    """
    Crée un historique des positions non nulles.
    
    Args:
        M (list of list): Matrice à analyser
        
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
    
    
    
    
def link(M, i, j):
    """
    Trouve les liens possibles depuis une cellule.
    
    Args:
        M (list of list): Matrice du labyrinthe
        i, j (int): Position de la cellule
        
    Returns:
        list: Liste des liens possibles
    """
    
    n, m= len(M), len(M[0])
    L = []
    val = M[i][j]
    
    if M[i][j] + M[(i+2)%n][j] != val and M[i][j] + M[(i+2)%n][j] < 2*val:
        L.append([i+2, j])
        
    if M[i][j] + M[(i-2)%n][j] != val and M[i][j] + M[(i-2)%n][j] < 2*val:
        L.append([i-2, j])
    
    if M[i][j] + M[i][(j+2)%m] != val and M[i][j] + M[i][(j+2)%m] < 2*val:
        L.append([i, j+2])
    
    if M[i][j] + M[i][(j-2)%m] != val and M[i][j] + M[i][(j-2)%m] < 2*val:
        L.append([i, j-2])
        
    return L
    
def generate_maze(n, m):
    assert n%2 == 1 and m%2 == 1, 'n et m doivent être impaire'
    """
    Génère un puzzle avec la méthode du labyrinthe.
    
    Args:
        n (int): Nombre de lignes
        m (int): Nombre de colonnes
        
    Returns:
        list of list: Matrice du puzzle
    """
    
    M, Hist= init_maze(n, m)[0] , init_maze(n, m)[1]

    h = len(Hist)
    r.shuffle(Hist)
    S=[]
    
    while h != 0:
        ind = r.randint(0, h - 1)
        Hist[ind][0], Hist[-1][0] = Hist[-1][0], Hist[ind][0]
        Hist[ind][1], Hist[-1][1] = Hist[-1][1], Hist[ind][1]
        i, j = Hist[-1][0], Hist[-1][1]
        L = link(M, i, j)
        
        if len(L) == 0:
            Hist.pop()
            h -= 1
            
        else:
            r.shuffle(L)
            i_p, j_p = L[0][0], L[0][1]
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
        M (list of list): Matrice à modifier
        i, j (int): Position de départ
        val: Valeur à propager
    """
    
    M[i][j] = val
    Croix = Croix_ind(M, i, j)
    for i_p, j_p in Croix:
        if M[i_p][j_p] != 0 and M[i_p][j_p] != val:
            homog(M, i_p, j_p, val)

    
#______________Methode_Mutation

def modifie(M):
    """
    Modifie une case modifiable
    algorithme pas du tout optimal
    """
    n, m = len(M), len(M[0])
    
    L = []
    for i in range(n):
        for j in range(m):
            if modifiable(M, i, j):
                L.append([i, j])
    
    case = r.randint(0, len(L)-1)
    
    im, jm = L[case][0], L[case][1]
    
    M[im][jm] = not M[im][jm]
    
def mutation(N,n,m):

    M = grignotage_carré(generate_chemins_complexes(n, m), n, m)
    
    for _ in range(N):
        modifie(M)
    
    return M

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

        M = [[s[8],s[7],s[6]],[s[1],s[0],s[5]],[s[2],s[3],s[4]]]
        
        
        L.append(M)
        
    return L
    
def List_nxm(n,m):
    """Génère toutes les matrices n x m possibles."""
    L = []
    f = '0'
    f += str(n*m)
    f += 'b'
    
    for k in range(2**(n*m)):
    
        if k % 10000 == 0:
            print(k," ieme matrice crée")
        s = list(format(k,f))
    
        for p in range (n*m):
            if s[p] == '0':
                s[p] = False
            else:
                s[p] = True
                
        M = [[False]*m for _ in range(n)]
        somme = 0
    
        for i in range(n):
            for j in range(m):
                M[i][j] = s[somme]
                somme += 1
                
        L.append(M)
    
    return L
    
#____________Filtres
    
def filtre_equiv(L):
    """Filtre les matrices équivalentes par rotation/symétrie."""
    Hist = []
    L_filtre = []
    for M in L:
        Ma = np.array(M)
        if not M in Hist:
            Hist.append(M)
            Hist.append(np.rot90(Ma,k=1).tolist())
            Hist.append(np.rot90(Ma,k=2).tolist())
            Hist.append(np.rot90(Ma,k=3).tolist())
            Mta = np.transpose(Ma)
            Hist.append(Mta.tolist())
            Hist.append(np.rot90(Mta,k=1).tolist())
            Hist.append(np.rot90(Mta,k=2).tolist())
            Hist.append(np.rot90(Mta,k=3).tolist())
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
    """Filtre suivant le nombre de case modifiable"""
    L_filtre = []
    for M in L:
        nb_modif = len(List_modifiable(M))
        if nb_modif >= mini and nb_modif <= maxi:
            L_filtre.append(M)
            
    return L_filtre
    


#___________Test_de_bon_retour_pour_les_critères

def in_pas_diag(M):
    """Verifie si une matrice de True est false ne contient pas des True
    ou False en diagonale (ce qui rend impossible le fait d'être un puzzle SL"""
    
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

        M = [[s[8],s[7],s[6]],[s[1],s[0],s[5]],[s[2],s[3],s[4]]]
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

        M = [[s[8],s[7],s[6]],[s[1],s[0],s[5]],[s[2],s[3],s[4]]]
        
        
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
        
    return L_sort

def Sort_nb_modif(L,mini,maxi):
    """Trie une liste de matrices selon le nombre de cases interieurs"""
    n= len(L)
    maxi = 16
    
    L_sort = []
    
    for i in range(mini,maxi+1):
        for j in range (n):
            if len(List_modifiable(L[j])) == i:
                L_sort.append(L[j])
        
    return L_sort

    
def Tri(L):
    """Trie une liste de matrices suivant le nombre de True"""
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
    
def List_modifiable(M):
    """renvoie une liste des indices des cases modifiable"""
    n,m = M.shape
    L=[]
    
    for i in range(n):
        for j in range(m):
            if modifiable(M,i,j):
                L.append([i,j])
    
    return L
    
def List_bout(M):
    """renvoie une liste des indices des cases qui sont des bout"""
    n,m = M.shape
    L=[]
    
    for i in range(n):
        for j in range(m):
                
            acc = 0
            C = Croix(M,i,j)
            
            for b in C:
                if b:
                    acc += 1
                    
            if  (M[i][j] and acc ==3) or ( ( not M[i][j] ) and acc == 1 ) :
                L.append([i,j])
    
    return L
    

    
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
        M (list of list): Matrice du puzzle
        NA (list of list): Matrice des nombres attendus
        
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
                
def Nombre_Arrete(M):
    """
    Calcule la matrice des nombres d'arrête.
    
    Args:
        M (list of list): Matrice du puzzle
        
    Returns:
        list of list: Matrice des nombres attendus
    """
    
    n,m = len(M) , len(M[0])
    NA =[[0]*m for _ in range(n)]
    
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
        M (list of list): Matrice du puzzle
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
        M (list of list): Matrice du puzzle
        
    Returns:
        list of list: Matrice de la couche externe
    """

    n,m = len(M),len(M[0])
    Couche = [[False]*(m+2) for _ in range(n+2)]
    
    for i in range(n+2):
        for j in range(m+2):
            if i>0 and i<=n and j>0 and j<=m:
                Couche[i][j] = M[i-1][j-1]
    
    return Couche
    
def Verif(M):

    """
    Vérifie la validité d'une solution par modélisation 2 couleurs.
    ie : 1) interieur(I = True) convexe
         2) exterieur(E = False) + couche externe convexe
         3) forme une partition de l'ensemble
    
    Args:
        M (list of list): Matrice du puzzle
        
    Returns:
        bool: True si le puzzle est valide
    """

    n,m = len(M),len(M[0])

    i,j=0,0
    found = False
    
    while i<n and not found: #trouver une case à l'interieur
        j=0
        while j<m and not found:
            if M[i][j]:
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
        
        #vérifie si les deux ensembles forment une partition
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
        M (list of list): Matrice à afficher
    """
    
    fig, ax = plt.subplots()
    img = ax.imshow(M)
    plt.show()
    
def show_gris(M):

    """
    Affiche une matrice simple en echelle de gris.
    
    Args:
        M (list of list): Matrice à afficher
    """

    fig, ax = plt.subplots()
    img = ax.imshow(M,cmap='gray')
    plt.show()

def show_2(M,NA):

    """
    Affiche deux matrices côte à côte.
    
    Args:
        M (list of list): Première matrice
        NA (list of list): Deuxième matrice
    """

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    img1 = ax1.imshow(M)
    img2 = ax2.imshow(NA)
    plt.show()
  
def show_n(L,n,m):

    """
    Affiche plusieurs matrices dans une grille de taille n*m.
    
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
    n,m = len(M) , len(M[0])
    """
    Affiche une animation d'une séquence de matrices.
    
    Args:
        L (list): Liste de matrices à animer
        interval (int): Intervalle entre les frames en ms
    """
    
    fig, ax = plt.subplots()
    
    vmin = min(np.min(np.array(m)) for m in L)
    vmax = max(np.max(np.array(m)) for m in L)
    
    im = ax.imshow(L[0], vmin=vmin, vmax=vmax)
    
    def update(frame):
        im.set_array(L[frame])
        return [im]
    
    anim = animation.FuncAnimation(fig, update, frames=len(L), 
                                 interval=interval, blit=True, repeat=False)
    
    plt.show()
    return anim
    
"""
~~~~~~~~~~~~~~~~~~~
V - Solve puzzle
~~~~~~~~~~~~~~~~~~~
"""

#_______Brut_Force
def List_puzzle(n,m):

    L = List_nxm(n,m)
    print("L contient ",len(L),"élément")
    L1 = filtre_diag(L)
    print("filtre_diag : OK ",len(L1))
    #L2 = filtre_equiv(L1)
    #print("filtre_equiv : OK ",len(L2))
    L3 = filtre_puzzle(L1)
    print("filtre_puzzle : OK ",len(L3))
    
    return L3

def Brut_force(Na):
    """test tout"""
    n,m = len(Na),len(Na[0])
    
    L = List_puzzle(n,m)
    l = len (L)
    
    for i in range(l):
        M = Nombre_Arrete(L[i]).tolist()
        
        if i%10000 == 0:
            print(i)

        if M == Na :
            print("trouvé !")
            return L[i]

#________Tentative_par_essaie_modifie_avec_heuristique

def Solve_par_essaie_modifie(N,tentative = 100):

    n,m = len(N) , len(N[0])
    
    #genere une matrice puzzle
    
    M = grignotage_rec(n,m)
    L = []
    
    
    for _ in range(tentative):
        
        
        #pour voir la progression
        M_p = c.deepcopy(M)
        L.append(M_p)
        
        #regarde les modifiables et le score de la matrice
        L_modif = List_modifiable(M)
        l_modif = len(L_modif)
        score = Score(M,N,m,n)
        print(score,M)
        
        #On modifie aleatoirement les modifiables jusqu'a ce que le score soit <
        nb_essaie = 2**(l_modif)
        while Score(M,N,m,n) <= score and nb_essaie>0:
            nb_essaie -= 1
            L_random = List_random(l_modif)
            
            for k in range(l_modif):
            
                i,j = L_modif[k]
            
                if L_random == 1:
                    M[i,j] = True
                    
                else:
                    M[i,j] = False      
                
        #si c'est la matrice c'est bon
        if nb_essaie <=0:
            print("pas moyen de monter le score")
            M = grignotage_rec(n,m)
        
        if Score(M,N,m,n) == n*m:
            print("trouvé")
            return M
            
    print("pas trouvé")
   
    return L
                        

def List_random(n):
    """Génère aléatoirement une liste de booléen de taille n"""
    L=[]
    for i in range (n):
        ra = r.randint(0,1)
        
        if ra == 0:
            L.append(False)
        else:
            L.append(True)
            
    return L
    
def Score(M,Na,n,m):

    Na_m = Nombre_Arrete(M)
    s = 0
    
    for i in range(n):
        for j in range(m):
            if Na_m[i,j] == Na[i,j]:
                s+=1
                
    return s
    
#________Backtracking_Original
        
class SolutionFound(Exception):
    pass

def solve_SL(Na):
    n, m = len(Na), len(Na[0])
    M = [[False]*m for _ in range(n)]
    Na_M = [[0]*m for _ in range(n)]
    try:
        # on tente d'abord en posant la première case à 1 puis 2
        if Na[0][0] == -1 or Na[0][0] >= 2:
            Na_M,restore = modifie_Na_M(M,Na_M,0,0)
            rec_brut_SL(M, Na_M, Na, 0, 0, True,n,m)
            Na_M = restore_Na_M(Na_M,restore)

        if Na[0][0] == -1 or Na[0][0] <= 2:
            rec_brut_SL(M, Na_M, Na, 0, 0, False,n,m)
            
        print("Aucune solution trouvée.")
    except SolutionFound:
        #print(" Solution trouvée !")
        #for row in range(n):
            #print(M[row])
        #print('---')
        return M,Na
        
def rec_brut_SL(M, Na_M, Na, i, j, Val,n,m):
    # sauvegarde pour le backtracking
    prev = M[i][j]
    M[i][j] = Val

    # affichage pour debug (on affiche une copie pour éviter effet de référence)
    
    #for row in range(n):
        #print(M[row],Na_M[row],Na[row],i,j)
    #print("--")

    # test de solution
    if solution_trouvé(Na, Na_M, n, m):
        raise SolutionFound

    # avancer dans l'ordre des cases : (i,j) -> next_i, next_j
    # calcule next indices
    next_i, next_j = i, j
    if j > 0 and i < n-1:
        next_i, next_j = i+1, j-1

    elif i==n-1 and j == m-1:
        # dernière case ; on a déjà testé l'égalité, donc pas de suite
        # restore et return
        M[i][j] = prev
        return
        
    else:
        # première case de la prochaine diagonale (s+1), en haut à droite
        s = i + j
        next_i = max(0, s + 1 - (m - 1))
        next_j = s + 1 - next_i
        

    # obtenir états possibles pour la prochaine case
    E = états_possible(M, Na_M, Na, next_i, next_j,n,m)

    # explore selon ce qui est possible (tester présence plutôt qu'égalité stricte)
    if True in E:
        #modifie Na
        Na_M,restore = modifie_Na_M(M,Na_M,next_i,next_j)
        #applique l'algo sur la prochaine case
        rec_brut_SL(M, Na_M, Na, next_i, next_j, True,n,m)
        # restore l'ancienne valeur de Na_M
        Na_M = restore_Na_M(Na_M,restore)
    if False in E:
        rec_brut_SL(M, Na_M, Na, next_i, next_j, False,n,m)

    # backtrack : restaurer l'ancienne valeur avant de retourner
    M[i][j] = prev

#all trouve l'ensemble des solutions
        
def solve_SL_all(Na):
    n, m = len(Na), len(Na[0])
    solutions = []  # liste qui contiendra toutes les solutions
    # Initialisation
    M = [[False]*m for _ in range(n)]
    Na_M = [[0]*m for _ in range(n)]
    
    #si Na se trouve dans le cercle critique on tourne le puzzle
    #tourné = False
    #print(score_init(Na))
    #if score_init(Na) == 0 :
        #Na = np.rot90(Na,k=2).tolist()
        #tourné = True
    #print(score_init(Na))
        
    def rec_brut_SL_all(M, Na_M, Na, i, j, Val,n,m):
        # sauvegarde pour le backtracking
        prev = M[i][j]
        M[i][j] = Val
    
        # affichage pour debug (on affiche une copie pour éviter effet de référence)
        #for row in range(n):
            #print(M[row],Na_M[row],Na[row],i,j)
        #print("--")
    
        # avancer dans l'ordre des cases : (i,j) -> next_i, next_j
        # calcule next indices
        next_i, next_j = i, j
        if j > 0 and i < n-1:
            next_i, next_j = i+1, j-1
    
        elif i==n-1 and j == m-1:
            # dernière case ; on a déjà testé l'égalité, donc pas de suite
            # test de solution
            if solution_trouvé(Na, Na_M, n, m) : #and Verif(M):
                #if tourné :
                    #M = np.rot90(M,k=2).tolist()
                solutions.append([row[:] for row in M])
                print('est_solution')
            
            # restore et return
            M[i][j] = prev
            return
            
        else:
            # première case de la prochaine diagonale (s+1), en haut à droite
            s = i + j
            next_i = max(0, s + 1 - (m - 1))
            next_j = s + 1 - next_i
            
    
        # obtenir états possibles pour la prochaine case
        E = états_possible(M, Na_M, Na, next_i, next_j,n,m)
    
        # explore selon ce qui est possible (tester présence plutôt qu'égalité stricte)
        if True in E:
            #modifie Na
            Na_M,restore = modifie_Na_M(M,Na_M,next_i,next_j)
            
            rec_brut_SL_all(M, Na_M, Na, next_i, next_j, True,n,m)
            
            # restore l'ancienne valeur de Na_M
            Na_M = restore_Na_M(Na_M,restore)
        if False in E:
            rec_brut_SL_all(M, Na_M, Na, next_i, next_j, False,n,m)
    
        # backtrack : restaurer l'ancienne valeur avant de retourner
        M[i][j] = prev
    
    # Appels initiaux (première case)
    if Na[0][0] >= 2 or Na[0][0] == -1 :
        Na_M, restore = modifie_Na_M(M, Na_M, 0, 0)
        rec_brut_SL_all(M, Na_M, Na, 0, 0, True,n,m)
        Na_M = restore_Na_M(Na_M, restore)   # restauration après retour
    if Na[0][0] <= 2 or Na[0][0] == -1:
        rec_brut_SL_all(M, Na_M, Na, 0, 0, False,n,m)
    
    # Affichage des résultats
    if solutions:
        #print(f"{len(solutions)} solution(s) trouvée(s) :")
        #for idx, (M_sol) in enumerate(solutions):
            #print(f"Solution {idx+1}:")
            #for row in range(n):
                #print(M_sol[row])
            #print('---')
        #print('return')
        return solutions
    else:
        print("Aucune solution trouvée.")
        return []

def modifie_Na_M(M,Na_M,i,j):
    restore = []
    C = Croix_ind(M,i,j)
    
    val = 0
    c = len(C)
    
    for ic,jc in C:
        if M[ic][jc] == True :
            restore.append([ic,jc, Na_M[ic][jc]])
            Na_M[ic][jc] = Na_M[ic][jc] - 1
        else :
            restore.append([ic,jc, Na_M[ic][jc]])
            Na_M[ic][jc] = Na_M[ic][jc] + 1
            val += 1
    
    restore.append([i,j,Na_M[i][j]])
    Na_M[i][j] = val + (4 -c)
    return Na_M,restore
    
def restore_Na_M(Na_M,restore):
    for i,j,val in restore:
        Na_M[i][j] = val
        
    return Na_M

    
def états_possible(M,Na_M,Na,i,j,n,m):
    E = [True, False]
    #regarde si il y a une condition bas
    c_bas = condition_bas(M,Na_M,Na,i,j)
    if c_bas != 'pas de condition bas':
        E.remove(not c_bas)

    # regarde si il y a une condition a droite
    c_droite = condition_droite(M,Na_M,Na,i,j)
    if c_droite != 'pas de condition droite':
        if c_droite != c_bas:
            E.remove(not c_droite)
            
    #condition pas de motif de diagonal apparrait
    choix_qui_forme_diag = condition_non_diag(M,i,j)
    for choix in choix_qui_forme_diag:
        if choix in E:
            E.remove(choix)

    #vérifie qu'il n'y est pas de contradiction et agit en conséquence
    if not condition_mettre_out(M,Na_M,Na,i,j,n,m) and False in E :
        E.remove(False)

    if not condition_mettre_in(M,Na_M,Na,i,j,n,m) and True in E:
        E.remove(True)
    
    return E

def condition_non_diag(M,i,j):
    if i==0 or j==0 :
        return []
    E= []
    if M[i][j-1] == M[i-1][j] and M[i-1][j-1] == True and M[i][j-1] == False :
        E.append(True)
    if M[i][j-1] == M[i-1][j] and M[i-1][j-1] == False and M[i][j-1] == True :
        E.append(False)
    return E
        
def condition_bas(M,Na_M,Na,i,j):
    # l'état en dessous ne pouvant plus être modifier après
    if Na[i-1][j] == -1 or i == 0:
        return 'pas de condition bas'
    else :
        if Na_M[i-1][j] == Na[i-1][j]: #il ne faut plus modifier
            return False #il faut laissé out
        else: # il faut le modifier
            return True #il faut mettre in

def condition_droite(M,Na_M,Na,i,j):
    # état de droit ne pourras être modifier plus que 1 fois après
    if Na[i][j-1] == -1 or j == 0:
        return "pas de condition droite"
    else :
        if M[i][j-1] and Na_M[i][j-1] + 2 == Na[i][j-1] : 
            # si l'etat est in et doit être modifier 2 fois
            return True #il faut mettre a l'intérieur
        elif (not M[i][j-1]) and Na_M[i][j-1] - 2 == Na[i][j-1] :
            # si l'etat est out et doit être modifier 2 fois
            return True # il faut mettre a l'interieur
        elif Na_M[i][j-1] == Na[i][j-1] :
            # si l'etat final déjà atteint il ne faut plus le modifier
            return False #il faut mettre a l'exterieur
        else : # les deux états sont possibles
            return "pas de condition droite"

def condition_mettre_out(M,Na_M,Na,i,j,n,m):
    #il faut que la fonction renvoie true sinon il y a une contradiction
    # vérifier qu'il n'y a pas de contradiction si on laisse out
    if Na[i][j] == -1 :
        return True
        
    else :
        val_si_out = Na_M[i][j]
    
        if i+1 == n and j+1 == m :
            # On ne plus apporté de modification
            return val_si_out == Na[i][j]
        
        elif j+1 == m : #dernière ligne on ne pourras modifier qu'une fois
            return Na[i][j] - val_si_out >= 0 and Na[i][j] - val_si_out <= 1
        else : # on ne peut modifier que 2 fois
            return Na[i][j] - val_si_out >= 0 and Na[i][j] - val_si_out <= 2

def condition_mettre_in(M,Na_M,Na,i,j,n,m):
    # Il faut que la focntion renvoie true sinon il y a contradiction
    # vérifier qu'il n'y a pas de contradiction si on met in
    if Na[i][j] == -1:
        return True
        
    else :
        val_si_in = 4 - Na_M[i][j]
    
        if i+1 == n and j+1 == m :
            # On ne plus apporté de modification
            return val_si_in == Na[i][j]
        
        elif j+1 == m : #dernière ligne on ne pourras modifier qu'une fois
            return val_si_in - Na[i][j] >= 0 and val_si_in - Na[i][j] <= 1
        else : #on ne peut modifier que 2 fois
            return val_si_in - Na[i][j] >= 0 and val_si_in - Na[i][j] <= 2


def solution_trouvé(Na, Na_M, n, m):
    #vérifie si la solution trouvé est la bonne
    i = 0
    res = True
    while i<n and res:
        j = 0
        while j<m and res:
            if Na[i][j] != -1 and Na[i][j] != Na_M[i][j]:
                res = False
            j += 1
        i += 1
    return res

#_______Solver_Z3
def z3_solver(Na):
    """
    Na : liste de listes d'entiers (n x m), valeurs dans -1,0,1,2,3,4.
    Retourne la liste de toutes les grilles M (booléennes) qui vérifient :
      - Pour chaque (i,j), # de voisins (4 directions, hors grille=False) valant not M[i][j] == Na[i][j] (si Na[i][j] != -1)
      - Aucun des deux motifs 2x2 interdits.
    """
    if not Na or not Na[0]:
        return []
    n = len(Na)
    m = len(Na[0])

    # Variables booléennes pour chaque cellule
    M = [[z3.Bool(f"M_{i}_{j}") for j in range(m)] for i in range(n)]
    solver = z3.Solver()

    # Directions orthogonales
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Contraintes de voisinage
    for i in range(n):
        for j in range(m):
            voisins = []
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    voisins.append(M[ni][nj])
                else:
                    voisins.append(z3.BoolVal(False))  # hors grille = False

            # Compte combien de voisins valent (not M[i][j])
            somme = z3.Sum([z3.If(v != M[i][j], 1, 0) for v in voisins])
            if Na[i][j] != -1:
                solver.add(somme == Na[i][j])

    # Contraintes anti-motifs 2x2
    for i in range(n - 1):
        for j in range(m - 1):
            # Motif 1 : [[True, False], [False, True]]
            solver.add(z3.Not(z3.And(M[i][j] == True,
                               M[i][j+1] == False,
                               M[i+1][j] == False,
                               M[i+1][j+1] == True)))
            # Motif 2 : [[False, True], [True, False]]
            solver.add(z3.Not(z3.And(M[i][j] == False,
                               M[i][j+1] == True,
                               M[i+1][j] == True,
                               M[i+1][j+1] == False)))

    # Collecte de toutes les solutions
    solutions = []
    while solver.check() == z3.sat:
        model = solver.model()
        sol = [[z3.is_true(model[M[i][j]]) for j in range(m)] for i in range(n)]
        solutions.append(sol)

        # Blocage pour éviter de retrouver la même solution
        bloc = []
        for i in range(n):
            for j in range(m):
                val = z3.is_true(model[M[i][j]])
                if val:
                    bloc.append(z3.Not(M[i][j]))
                else:
                    bloc.append(M[i][j])
        solver.add(z3.Or(bloc))

    return solutions


"""
~~~~~~~~~~~~~~~~~~~
VI - Test
~~~~~~~~~~~~~~~~~~~
"""
def comparaison_Backtrack_Z3(debut,fin,pas = 5,p = 0,test_par_taille = 1):
    L_1 = []
    L_2 = []
    T = []
    for i in range(debut,fin,pas) :
        tot_1=0
        tot_2=0
        for _ in range (test_par_taille) :
            M = generer(i,i,p)
            
            start = time.time()
            S1 = solve_SL_all(M)
            end = time.time()
            tot_1 += end-start
            
            """
            start = time.time()
            S2 = z3_solver(M)
            end = time.time()
            tot_2 += end-start
            """
        
        L_1.append(tot_1/test_par_taille)
        T.append(i)
        #L_2.append(tot_2/test_par_taille)
    
    
    plt.plot(T,L_1,color = 'red')
    #plt.plot(T,L_2,color = 'blue')
    plt.ylabel('temps')
    plt.xlabel('largeur du puzzle carré')
    plt.show()
    
def score_init(Na):
    n, m = len(Na), len(Na[0])
    tot = 0
    moy_i, moy_j = 0,0
    m_i_2, m_j_2 = 0,0
    for i in range(n):
        for j in range(m):
            if Na[i][j] == -1:
                m_i_2 += i**2
                m_j_2 += j**2
                moy_i += i
                moy_j += j
                tot+=1

    #return ((moy_i/tot - m_i_2/(tot))**2 + (moy_j/tot - m_j_2/(tot))**2 )**0.5
    #return ((moy_i/tot)**2 + (moy_j/tot)**2)**0.5
    
    if tot != 0 and ((moy_i/tot)**2 + (moy_j/tot)**2)**0.5 < (((n/2)**2 + (m/2)**2)**0.5)*0.8 :
        return 0
    else :
        return 1

def liens_init_temps(n,N,p):
    S = []
    T = []
    for _ in range (N) :
        Na = generer(n,n,p)
        S.append(score_init(Na))
        
        start = time.time()
        s = solve_SL_all(Na)
        end = time.time()
        T.append (end-start)
        
    compteur = 0
    for e in S:
        if e == 0 :
            compteur += 1
    print(compteur)
        
    plt.plot(S,T,'o',color = 'blue')
    plt.ylabel('temps')
    plt.xlabel('score initiale')
    plt.show()
    
"""
~~~~~~~~~~~~~~~~~~~
VI - Générer Puzzle
~~~~~~~~~~~~~~~~~~~
"""

def generer(n,m,p= 0):
    Na = Nombre_Arrete(grignotage_rec(n,m))
    H = []
    
    while len(H) < int((n*m)*p) :
        i,j = r.randint(0,n-1), r.randint(0,m-1)
        if not [i,j] in H:
            Na[i][j] = -1
            H.append([i,j])
        
    return Na
