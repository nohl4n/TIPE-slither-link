"""
====================
| Slitherlink GAME |
====================

Un programme complet pour générer, résoudre et visualiser des puzzles Slitherlink.
Le Slitherlink est un puzzle logique où le but est de connecter des points pour former
une boucle unique fermée, avec des indices numériques indiquant le nombre d'arêtes
à dessiner autour de chaque cellule.

Table des matières :
    
    0 - Fonction utile

    I - Génération du puzzle
        0) Critère de modifiabilité
        1) Grignotage carré
        2) Grignotage carré + crevasse
            1. Chemin simple
            2. Chemin complexe
        3) Grignotage récursif
        4) Génération type Labyrinthe
        5) Génération par mutation
        
    II - Étude de dénombrement
        a) Filtres
        b) Tests pour les filtres
        
    III - Vérifier puzzle
        a) Règles
        b) Règle équivalente en 2 couleurs
    
    IV - Fonctions pour visualiser des puzzles
        
    V - Solveur
        a) Brute force
        b) Méthode essai-modifié avec heuristique
        c) Backtracking original
        d) Solveur z3
    
    VI - Tests
        a) Comparaison Z3 avec Backtracking
        b) Influence de la répartition des cases non indicées sur le temps de résolution
    
    VII - Générer Puzzle
    
Auteur : SAUCET Nohlan
PROJET TIPE 2025 - 2026
"""


# _____________BIBLIOTHÈQUE

import matplotlib.pyplot as plt
import numpy as np
import random as r
import copy as c
import time
import sys
import z3


# _____________AUGMENTE LA PROFONDEUR DE RÉCURSION

sys.setrecursionlimit(100000)

"""
~~~~~~~~~~~~~~~~~~~
0 - Fonction utile
~~~~~~~~~~~~~~~~~~~
"""

def Croix(M,i,j):
    """Retourne la valeur des cellules voisines (en croix)."""
    
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
    """Retourne les indices des cellules voisines (en croix)."""
    
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
    """Retourne les 8 cases qui entourent la case d'indice (i,j)."""
    
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
    """Retourne les indices des 8 cases qui entourent la case d'indice (i,j)."""
    
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
I - Génération du puzzle
~~~~~~~~~~~~~~~~~~~
"""

#_________________Critère de modifiabilité d'une case

def modifiable(M,i,j):
    """Vérifie si une case peut être à la fois extérieure et intérieure."""
    
    Contour_M = Contour(M,i,j)
    Croix_M = Croix(M,i,j)
    
    # la case est connectée à l'intérieur et à l'extérieur
    bool1 = (False in Croix_M and True in Croix_M)
    
    # le contour n'a qu'une seule variation de couleur
    precedent = Contour_M[7]
    var = 0
    
    for k in range(8):
        if precedent != Contour_M[k]:
            var +=1
        precedent = Contour_M[k]
    bool2 = (var == 2)
    
    return bool1 and bool2
    
#1_________________Grignotage carré

def bordure(nd, nf, md, mf):
    """Donne la liste des indices des cellules de bordure."""
    
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
    """Génère un puzzle par grignotage carré."""
    
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
    

#_______________Génération par Chemins

def sign(x):
    """Retourne le signe d'un nombre."""
    return -1 if x < 0 else 1
        

def chemin (i1, j1, i2, j2):
    """Génère un chemin entre deux points."""
    
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
    
def generation_par_chemins(n, m):
    """Génère des fissures sur une solution de taille n*m."""
    
    #initialisation
    M =[[True]*m for _ in range(n)]
    B= bordure(0, n, 0, m)
    
    # choix de l'ordre des cases parcourues sur le bord
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1, b//5)):
        #choix du point d'arrivée de la fissure
        i, j = r.randint(1, n - 1),  r.randint(1, m - 1)
        #création d'une fissure d'un point du bord au point choisi
        W = chemin (B[k][0], B[k][1], i, j)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if modifiable(M, w[0], w[1]):
                M[w[0]][w[1]]= False

    return M
    
#________Génération par chemins complexes

def chemin_complexe (i1, j1, i2, j2, n, m):
    """Génère un chemin plus complexe entre deux points."""
    
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
    
def generation_par_chemins_complexes(n, m):
    """Génère des fissures sur une solution de taille n*m."""
    
    #initialisation
    M =[[True]*m for _ in range(n)]
    B= bordure(0, n, 0, m)
    
    # choix de l'ordre des cases parcourues sur le bord
    r.shuffle(B)
    b= len(B)
    
    for k in range(r.randint(1, b//5)):
        #choix du point d'arrivée de la fissure
        i, j = r.randint(1, n - 1),  r.randint(1, m - 1)
        #création d'une fissure d'un point du bord au point choisi
        W = chemin_complexe (B[k][0], B[k][1], i, j, n, m)
        M[B[k][0]][B[k][1]]= False
        for w in W:
            if modifiable(M, w[0], w[1]):
                M[w[0]][w[1]]= False

    return M
    
#__________________Grignotage récursif

def grignotage_rec(n, m, N = -1):
    """Génère une solution Slitherlink par grignotage récursif."""
    if N == -1:
        N = n*m
    
    
    M =[[True]*m for _ in range(n)]
    B= bordure(0, n, 0, m)
    b=len(B)
    r.shuffle(B)
    a = r.randint(1, b - 1)
    
    
    for k in range(a):
        if modifiable(M, B[k][0], B[k][1]):
            grignotage_aux(M, B[k][0], B[k][1], N)
    
    return M

def grignotage_aux (M, i, j, n):
    """Fonction auxiliaire pour le grignotage récursif."""

    if n>0:
    
        M[i][j] = not(M[i][j])
    
        C = Croix_ind(M, i, j)
        r.shuffle(C)
        c = len (C)

    
        for k in range(r.randint(1, c - 1)):
            if modifiable(M, C[k][0], C[k][1]):
                grignotage_aux(M, C[k][0], C[k][1], n - 1)

#______________Génération labyrinthe

def init_labyrinthe(n, m):
    assert n%2 == 1 and m%2 == 1, 'n et m doivent être impairs'
    """Initialise un labyrinthe pour la génération."""
    
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
    """Crée un historique des positions non nulles."""
    
    n, m = len(M), len(M[0])
    Hist = []
    
    for i in range(n):
        for j in range(m):
            if M[i][j] != 0:
                Hist.append([i, j])
                
    return Hist
    
    
    
    
def link(M, i, j):
    """Trouve les liens possibles depuis une cellule."""
    
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
    
def generation_par_labyrinthe(n, m):
    assert n%2 == 1 and m%2 == 1, 'n et m doivent être impairs'
    """Génère un puzzle avec la méthode du labyrinthe."""
    
    M, Hist= init_labyrinthe(n, m)[0] , init_labyrinthe(n, m)[1]

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
    """Uniformise les valeurs dans une zone connectée."""
    
    M[i][j] = val
    Croix = Croix_ind(M, i, j)
    for i_p, j_p in Croix:
        if M[i_p][j_p] != 0 and M[i_p][j_p] != val:
            homog(M, i_p, j_p, val)

    
#______________Méthode Mutation

def modifie(M):
    """Modifie une case modifiable (algorithme pas du tout optimal)."""
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
    """On part d'une solution générée par crevasse + grignotage carré puis on fait N mutations."""

    M = grignotage_carré(generation_par_chemins_complexes(n, m), n, m)
    
    for _ in range(N):
        modifie(M)
    
    return M

"""
~~~~~~~~~~~~~~~~~~~
II - Étude de Dénombrement
~~~~~~~~~~~~~~~~~~~
"""

def List_nxm(n,m):
    """Génère toutes les matrices n x m possibles."""
    
    L = []
    f = '0'
    f += str(n*m)
    f += 'b'
    
    for k in range(2**(n*m)):
    
        if k % 10000 == 0:
            print(k," ième matrice créée")
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

def filtre_diag(L):
    """Filtre les matrices sans connexions diagonales."""
    
    L_filtre =[]
    for M in L:
        if Pas_de_motif_diagonale(M):
            L_filtre.append(M)
    return L_filtre
    
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
    
def filtre_nombre_modifiable(L,mini,maxi):
    """Filtre suivant le nombre de cases modifiables."""
    
    L_filtre = []
    for M in L:
        nb_modif = len(List_modifiable(M))
        if nb_modif >= mini and nb_modif <= maxi:
            L_filtre.append(M)
            
    return L_filtre
    
def filtre_nombre_bout(L,mini,maxi):
    """Filtre suivant le nombre de cases modifiables."""
    
    L_filtre = []
    for M in L:
        nb_modif = len(List_bout(M))
        if nb_modif >= mini and nb_modif <= maxi:
            L_filtre.append(M)
    


#___________Test de bon retour pour les critères
## Dans cette partie on essaie de trouver des critères d'esthétisme d'un puzzle avec le nombre de bouts ou le nombre de modifiables

def Pas_de_motif_diagonale(M):
    """Vérifie si une matrice de True et False ne contient pas des True
    ou False en diagonale (ce qui rend impossible le fait d'être un puzzle Slitherlink). Voir dans le papier : M n'est pas muni d'une CS-coloration."""
    
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

def List_modifiable(M):
    """Renvoie une liste des indices des cases modifiables."""
    
    n,m = len(M),len(M[0])
    L=[]
    
    for i in range(n):
        for j in range(m):
            if modifiable(M,i,j):
                L.append((i,j))
    
    return L

def Tri_par_nb_modifiable(L,mini,maxi):
    """Trie une liste de matrices selon le nombre de cases intérieures."""
    
    l = len(L)
    L_sort = []
    
    for i in range(mini,maxi+1):
        for j in range (l):
            if len(List_modifiable(L[j])) == i:
                L_sort.append(L[j])
        
    return L_sort
    
def List_bout(M):
    """Renvoie une liste des indices des cases qui sont des bouts."""
    
    n,m = len(M),len(M[0])
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
    
def Tri_par_nb_bout(L,mini,maxi):
    """Trie une liste de matrices selon le nombre de cases qui sont des bouts."""
    
    l = len(L)
    
    L_sort = []
    
    for i in range(mini,maxi+1):
        for j in range (l):
            if len(List_bout(L[j])) == i:
                L_sort.append(L[j])
        
    return L_sort
    

    
"""
~~~~~~~~~~~~~~~~~~~
III - Vérifier puzzle
~~~~~~~~~~~~~~~~~~~
"""

#___________Solution -> Puzzle
                
def Nombre_Arrete(M):
    """Calcule la matrice des nombres d'arêtes, c'est-à-dire le puzzle associé à la solution (CS-coloration)."""
    
    n,m = len(M) , len(M[0])
    NA =[[0]*m for _ in range(n)]
    
    for i in range (n):
        for j in range(m):
            C = Croix(M,i,j)
            c=len(C)
            n_=0
            if M[i][j]:
                n_= 4-c
                for k in range(c):
                    if not C[k]:
                        n_+=1
                NA[i][j]=n_
                
            if not M[i][j]:
                for k in range(c):
                    if C[k]:
                        n_+=1
                NA[i][j]=n_
    
    return NA
    
#_______________Vérification CS-coloration

def Hist(M,i,j,L):
    """Crée un historique des cellules connectées."""

    L.append([i,j])
    C = Croix_ind(M,i,j)
    
    for ind in C:
        if M[ind[0]][ind[1]] == M[i][j] and not ([ind[0],ind[1]] in L):
            Hist(M,ind[0],ind[1],L)

    
def Couche_ext(M):
    """Crée une couche externe pour la vérification (pour la connexité de la couleur extérieure)."""

    n,m = len(M),len(M[0])
    Couche = [[False]*(m+2) for _ in range(n+2)]
    
    for i in range(n+2):
        for j in range(m+2):
            if i>0 and i<=n and j>0 and j<=m:
                Couche[i][j] = M[i-1][j-1]
    
    return Couche
    
def Verif(M):
    """Vérifie la validité d'une solution par modélisation 2 couleurs.
    c'est-à-dire : 1) intérieur (I = True) convexe
                   2) extérieur (E = False) + couche externe convexe
                   3) forme une partition de l'ensemble
    (on vérifie ainsi que M soit muni d'une CS-coloration)."""

    n,m = len(M),len(M[0])

    i,j=0,0
    found = False
    
    while i<n and not found: #trouver une case à l'intérieur
        j=0
        while j<m and not found:
            if M[i][j]:
                found = True
                i,j = i-1,j-1
            j+=1
        i+=1

    if found == False: #si pas trouvé, matrice nulle
        return True
    else:
        
        # Histogramme des cases intérieures
        L_int=[]
        Hist(M,i,j,L_int)
        
        # Histogramme des cases extérieures
        L_ext=[]
        Couche= Couche_ext(M)
        Hist(Couche,0,0,L_ext)
        
        #vérifie si les deux ensembles forment une partition
        return len(L_int) + len(L_ext) == (n+2)*(m+2)
   

"""
~~~~~~~~~~~~~~~~~~~
IV - Afficher puzzle
~~~~~~~~~~~~~~~~~~~
"""

def show(M):
    """Affiche une matrice simple."""
    
    fig, ax = plt.subplots()
    img = ax.imshow(M)
    plt.show()
    
def show_gris(M):
    """Affiche une matrice simple en échelle de gris."""

    fig, ax = plt.subplots()
    img = ax.imshow(M,cmap='gray')
    plt.show()

def show_2(M,NA):
    """Affiche deux matrices côte à côte."""

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    img1 = ax1.imshow(M)
    img2 = ax2.imshow(NA)
    plt.show()
  
def show_n(L,n,m):
    """Affiche plusieurs matrices dans une grille de taille n*m."""

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
    """Affiche une animation d'une séquence de matrices."""
    M = L[0]
    n,m = len(M) , len(M[0])
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
V - Solveur
~~~~~~~~~~~~~~~~~~~
"""

#_______Brute Force
def List_puzzle(n,m, equiv = False):
    """Fait la liste de tous les puzzles."""

    L = List_nxm(n,m)
    print("L contient ",len(L),"éléments")
    L1 = filtre_diag(L)
    print("filtre_diag : OK ",len(L1))
    if equiv:
        L2 = filtre_equiv(L1)
        print("filtre_equiv : OK ",len(L2))
    else:
        L2 = L1
    L3 = filtre_puzzle(L2)
    print("filtre_puzzle : OK ",len(L3))
    
    return L3

def brut_force_solveur(Na):
    """Teste tout."""
    
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

#________Tentative par essai-modifié avec heuristique

def heuristique_solveur(N,tentative = 100):
    """On génère une matrice aléatoirement et avec une heuristique (score) on se rapproche de la solution cherchée (ne marche pas sur plus de 10*10)."""
    
    n,m = len(N) , len(N[0])
    print(N)
    
    #génère une matrice puzzle
    
    #M = grignotage_rec(n,m) #[[True]*m for _ in range(n)]
    M = [[True]*m for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            if r.random() < 0.5:
                M[i][j] = True
            else:
                M[i][j] = False
    
    Historique = []
    
    
    for _ in range(tentative):
        
        #on vérifie que Na(M) = N
        Na_m = Nombre_Arrete(M)
        print("Na_m = ",Na_m)
        
        if solution_trouvé(N, Na_m, n, m) :
            print('trouvé')
            return M, Na_m
        
        #enregistrer les matrices déjà vues
        M_p = c.deepcopy(M)
        Historique.append(M_p)
        
        #regarde les modifiables
        #L_modif = List_modifiable(M)
        
        #Liste des indices
        L_modif = liste_ind(n,m)
        
        #score de la matrice initiale
        score_init = Score(M,N,m,n)
        print("score init = ",score_init,"/ M = ",M)
        
        #On modifie aléatoirement les modifiables jusqu'à ce que le score soit le plus petit
        
        mini = score_init
        i_min, j_min = -1,-1
        
        for i,j in L_modif :
            #modification
            M[i][j] = not M[i][j]
            M_modif = c.deepcopy(M)
            #restauration
            M[i][j] = not M[i][j]
            
            score_modif = Score(M_modif,N,m,n)
            
            if score_modif <= mini and (not M_modif in Historique) :
                mini = score_modif
                i_min, j_min = i,j
            
                
        #si on n'a pas réussi à trouver plus petit, on est bloqué
        if i_min == -1:
            print("pas moyen de descendre le score")
            return heuristique_solveur(N,tentative - 1)
        
        #sinon on prend le score qui minimise
        else :
            M[i_min][j_min] = not M[i_min][j_min]
            
    print("pas trouvé après ", tentative, "descentes de score")

    
def Score(M,Na,n,m):
    """Renvoie la somme des différences entre Na_M et Na."""
    
    Na_m = Nombre_Arrete(M)
    s = 0
    for i in range(n):
        for j in range(m):
            s+= abs( Na_m[i][j] - Na[i][j] )
                
    return s
    
def liste_ind(n,m) :
    """Renvoie la liste des indices d'une matrice de taille n*m."""
    
    L = []
    for i in range(n):
        for j in range(m):
            L.append((i,j))
    return L
    
#________Backtracking Original
        
class SolutionFound(Exception):
    pass

def backtrack_solveur(Na):
    n, m = len(Na), len(Na[0])
    M = [[False]*m for _ in range(n)]
    Na_M = [[0]*m for _ in range(n)]
    try:
        # on tente d'abord en posant la première case à 1 puis 2
        if Na[0][0] == -1 or Na[0][0] >= 2:
            Na_M,restore = modifie_Na_M(M,Na_M,0,0)
            rec_backtrack(M, Na_M, Na, 0, 0, True,n,m)
            Na_M = restore_Na_M(Na_M,restore)

        if Na[0][0] == -1 or Na[0][0] <= 2:
            rec_backtrack(M, Na_M, Na, 0, 0, False,n,m)
            
        print("Aucune solution trouvée.")
    except SolutionFound:
        #print(" Solution trouvée !")
        #for row in range(n):
            #print(M[row])
        #print('---')
        return M,Na
        
def rec_backtrack(M, Na_M, Na, i, j, Val,n,m):
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
            # test de solution
        if solution_trouvé(Na, Na_M, n, m):
            raise SolutionFound

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
        rec_backtrack(M, Na_M, Na, next_i, next_j, True,n,m)
        # restore l'ancienne valeur de Na_M
        Na_M = restore_Na_M(Na_M,restore)
    if False in E:
        rec_backtrack(M, Na_M, Na, next_i, next_j, False,n,m)

    # backtrack : restaurer l'ancienne valeur avant de retourner
    M[i][j] = prev

#all trouve l'ensemble des solutions
        
def backtrack_solveur_all(Na):
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
        
    def rec_backtrack_all(M, Na_M, Na, i, j, Val,n,m):
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
            if solution_trouvé(Na, Na_M, n, m) and Verif(M):
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
            
            rec_backtrack_all(M, Na_M, Na, next_i, next_j, True,n,m)
            
            # restore l'ancienne valeur de Na_M
            Na_M = restore_Na_M(Na_M,restore)
        if False in E:
            rec_backtrack_all(M, Na_M, Na, next_i, next_j, False,n,m)
    
        # backtrack : restaurer l'ancienne valeur avant de retourner
        M[i][j] = prev
    
    # Appels initiaux (première case)
    if Na[0][0] >= 2 or Na[0][0] == -1 :
        Na_M, restore = modifie_Na_M(M, Na_M, 0, 0)
        rec_backtrack_all(M, Na_M, Na, 0, 0, True,n,m)
        Na_M = restore_Na_M(Na_M, restore)   # restauration après retour
    if Na[0][0] <= 2 or Na[0][0] == -1:
        rec_backtrack_all(M, Na_M, Na, 0, 0, False,n,m)
    
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
    #regarde s'il y a une condition bas
    c_bas = condition_bas(M,Na_M,Na,i,j)
    if c_bas != 'pas de condition bas':
        E.remove(not c_bas)

    # regarde s'il y a une condition à droite
    c_droite = condition_droite(M,Na_M,Na,i,j)
    if c_droite != 'pas de condition droite':
        if c_droite != c_bas:
            E.remove(not c_droite)
            
    #condition pas de motif diagonal apparaît
    choix_qui_forme_diag = condition_non_diag(M,i,j)
    for choix in choix_qui_forme_diag:
        if choix in E:
            E.remove(choix)

    #vérifie qu'il n'y a pas de contradiction et agit en conséquence
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
    # l'état en dessous ne pouvant plus être modifié après
    if Na[i-1][j] == -1 or i == 0:
        return 'pas de condition bas'
    else :
        if Na_M[i-1][j] == Na[i-1][j]: #il ne faut plus modifier
            return False #il faut laisser out
        else: # il faut le modifier
            return True #il faut mettre in

def condition_droite(M,Na_M,Na,i,j):
    # état de droite ne pourra être modifié plus qu'une fois après
    if Na[i][j-1] == -1 or j == 0:
        return "pas de condition droite"
    else :
        if M[i][j-1] and Na_M[i][j-1] + 2 == Na[i][j-1] : 
            # si l'état est in et doit être modifié 2 fois
            return True #il faut mettre à l'intérieur
        elif (not M[i][j-1]) and Na_M[i][j-1] - 2 == Na[i][j-1] :
            # si l'état est out et doit être modifié 2 fois
            return True # il faut mettre à l'intérieur
        elif Na_M[i][j-1] == Na[i][j-1] :
            # si l'état final déjà atteint il ne faut plus le modifier
            return False #il faut mettre à l'extérieur
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
            # On ne peut plus apporter de modification
            return val_si_out == Na[i][j]
        
        elif j+1 == m : #dernière ligne on ne pourra modifier qu'une fois
            return Na[i][j] - val_si_out >= 0 and Na[i][j] - val_si_out <= 1
        else : # on ne peut modifier que 2 fois
            return Na[i][j] - val_si_out >= 0 and Na[i][j] - val_si_out <= 2

def condition_mettre_in(M,Na_M,Na,i,j,n,m):
    # Il faut que la fonction renvoie true sinon il y a contradiction
    # vérifier qu'il n'y a pas de contradiction si on met in
    if Na[i][j] == -1:
        return True
        
    else :
        val_si_in = 4 - Na_M[i][j]
    
        if i+1 == n and j+1 == m :
            # On ne peut plus apporter de modification
            return val_si_in == Na[i][j]
        
        elif j+1 == m : #dernière ligne on ne pourra modifier qu'une fois
            return val_si_in - Na[i][j] >= 0 and val_si_in - Na[i][j] <= 1
        else : #on ne peut modifier que 2 fois
            return val_si_in - Na[i][j] >= 0 and val_si_in - Na[i][j] <= 2


def solution_trouvé(Na, Na_M, n, m):
    #vérifie si la solution trouvée est la bonne
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

#_______Solveur Z3
def z3_solveur(Na):
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
    solveur = z3.Solver()

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
                solveur.add(somme == Na[i][j])

    # Contraintes anti-motifs 2x2
    for i in range(n - 1):
        for j in range(m - 1):
            # Motif 1 : [[True, False], [False, True]]
            solveur.add(z3.Not(z3.And(M[i][j] == True,
                               M[i][j+1] == False,
                               M[i+1][j] == False,
                               M[i+1][j+1] == True)))
            # Motif 2 : [[False, True], [True, False]]
            solveur.add(z3.Not(z3.And(M[i][j] == False,
                               M[i][j+1] == True,
                               M[i+1][j] == True,
                               M[i+1][j+1] == False)))

    # Collecte de toutes les solutions
    solutions = []
    while solveur.check() == z3.sat:
        model = solveur.model()
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
        solveur.add(z3.Or(bloc))

    return solutions


"""
~~~~~~~~~~~~~~~~~~~
VI - Tests
~~~~~~~~~~~~~~~~~~~
"""

def comparaison_solveur(debut,fin,pas = 5,p = 0,test_par_taille = 1):
    """Fonction permettant de comparer le temps de résolution des solveurs."""
    L_1 = []
    L_2 = []
    L_3 = []
    T = []
    for i in range(debut,fin,pas) :
        tot_1=0
        tot_2=0
        tot_3=0
        for _ in range (test_par_taille) :
            M = generer(i,i,p)
            
            start = time.time()
            S1 = backtrack_solveur_all(M)
            end = time.time()
            tot_1 += end-start
            
            
            start = time.time()
            S2 = z3_solveur(M)
            end = time.time()
            tot_2 += end-start
            
            start = time.time()
            S3 = heuristique_solveur(M)
            end = time.time()
            tot_3 += end-start
            
        
        L_1.append(tot_1/test_par_taille)
        L_2.append(tot_2/test_par_taille)
        L_3.append(tot_3/test_par_taille)
        T.append(i)
        
    
    
    plt.plot(T,L_1,color = 'red')
    plt.plot(T,L_2,color = 'blue')
    plt.plot(T,L_3,color = 'green')
    plt.ylabel('temps')
    plt.xlabel('largeur du puzzle carré')
    plt.show()
    

def liens_init_temps(n,N,p):
    """Teste l'influence de la répartition des cases non indicées sur le temps de résolution pour le backtracking."""

    S = []
    T = []
    for _ in range (N) :
        Na = generer(n,n,p)
        S.append(score_init(Na))
        
        start = time.time()
        s = backtrack_solveur_all(Na)
        end = time.time()
        T.append (end-start)
        
    compteur = 0
    for e in S:
        if e == 0 :
            compteur += 1
    print(compteur)
        
    plt.plot(S,T,'o',color = 'blue')
    plt.ylabel('temps')
    plt.xlabel('score initial')
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
        
def proportion_ind(nb,N = 100):
    """calcul la proportion de 0,1,2,3 et 4 dans un puzzle généré aléatoirement"""
    
    Compteur = [[0]*5 for _ in range(nb)]
    
    for n in range(3,nb+3):
    
        for _ in range(N):
            Na = Nombre_Arrete(mutation(n*n,n,n))
            for i in range(n):
                for j in range(n):
                    Compteur[n-3][Na[i][j]] += 1/(N*n*n)
    
    for n in range(3,nb+3):
        plt.plot([0,1,2,3,4], Compteur[n-3])
    
    plt.show()
    
    Moy = [0]*5
    
    for k in range(5):
        for n in range(3,nb+3):
            Moy[k] += Compteur[n-3][k]/nb
    
    
    return Moy   # environ [0.11020918761673644, 0.3322940755030184, 0.38178673579356437, 0.17331906951263848, 0.002390931574037635] pour mutation

        
#generation_par_chemins_complexes(n,n)
#grignotage_rec(n,n)
#mutation(n*n,n,n)

"""
~~~~~~~~~~~~~~~~~~~
VII - Générer Puzzle
~~~~~~~~~~~~~~~~~~~
"""

def generer(n,m,p= 0):
    """Génère un puzzle avec une proportion p de cases non indicées."""
    
    Na = Nombre_Arrete(grignotage_rec(n,m))
    H = []
    
    while len(H) < int((n*m)*p) :
        i,j = r.randint(0,n-1), r.randint(0,m-1)
        if not [i,j] in H:
            Na[i][j] = -1
            H.append([i,j])
        
    return Na
    
def generer_unique_sol(n,m,tentative):
    """Génération d'une solution puis on enlève tentative indices si cela ne casse pas l'unicité de la solution."""
    
    Na = Nombre_Arrete(grignotage_rec(n,m))
    H = []
    compteur = 0
    
    for _ in range(tentative) :
        i,j = r.randint(0,n-1), r.randint(0,m-1)
        
        if not [i,j] in H :
            H.append([i,j])
            na_i_j = Na[i][j]
            Na[i][j] = -1
        
            if not len(backtrack_solveur_all(Na)) == 1:
                Na[i][j] = na_i_j
            else : 
                compteur += 1
    
    return Na, compteur
    

    
def generer_random(n,m):
    """fait aléatoirement un puzzle jusqu'a ce qu'il y est au moins une solution"""
    # il y a rarement unicité de la solution
    Na = [[0]*m for _ in range(n)]
    H = []
    compteur = 0
    
    for i in range(n):
        for j in range(m):
            p = r.random()
            if p < 0.11020918761673644:
                Na[i][j] = 0
            elif p < 0.442503263:
                Na[i][j] = 1
            elif p < 0.824289999:
                Na[i][j] = 2
            else:
                Na[i][j] = 3
                
    for _ in range(int(n*m)) :
        i,j = r.randint(0,n-1), r.randint(0,m-1)
        
        if not [i,j] in H :
            H.append([i,j])
            na_i_j = Na[i][j]
            Na[i][j] = -1
            
            nb_sol = len(backtrack_solveur_all(Na))
        
            if nb_sol > 5:
                Na[i][j] = na_i_j
            
            elif nb_sol != 0:
                return Na,compteur

            else : 
                compteur += 1
    
    print('echec')
    return None
