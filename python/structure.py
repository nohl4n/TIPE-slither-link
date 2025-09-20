"""
Slitherlink Generator and Visualizer
===================================

Un programme complet pour générer, résoudre et visualiser des puzzles Slitherlink.
Le Slitherlink est un puzzle logique où le but est de connecter des points pour former
une boucle unique fermée, avec des indices numériques indiquant le nombre d'arêtes
à dessiner autour de chaque cellule.

Fonctionnalités:
- Génération de puzzles via différentes méthodes (grignotage, bruit, labyrinthe)
- Visualisation des puzzles avec matplotlib
- Vérification des règles du Slitherlink
- Résolution basique via force brute

Auteur: SAUCET Nohlan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import random
import copy
import sys
import math
from copy import deepcopy

# Augmenter la limite de récursion pour les grandes grilles
sys.setrecursionlimit(10000)

class SlitherlinkGenerator:
    """Classe principale pour générer des puzzles Slitherlink."""
    
    def __init__(self):
        self.animation_frames = []
    
    def grignotage_rec(self, n, m):
        """
        Génère un puzzle Slitherlink par grignotage récursif.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            
        Returns:
            np.array: Matrice booléenne représentant le puzzle
        """
        L = [[True] * m] * n
        M = np.array(L)
        B = self.bordure(0, n, 0, m)
        b = len(B)
        random.shuffle(B)
        a = random.randint(1, b - 1)
        
        for k in range(a):
            if self.In_able(M, B[k][0], B[k][1]):
                self.generate_rec(M, B[k][0], B[k][1], n * m)
        
        return M
    
    def grignotage_rec2(self, n, m):
        """
        Autre méthode de grignotage récursif commençant au centre.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            
        Returns:
            np.array: Matrice booléenne représentant le puzzle
        """
        L = [[True] * m] * n
        M = np.array(L)
        self.generate_rec(M, n // 2, m // 2, n * m)
        return M
    
    def inverse(self, M):
        """
        Inverse les valeurs booléennes d'une matrice.
        
        Args:
            M (np.array): Matrice à inverser
            
        Returns:
            np.array: Matrice inversée
        """
        n, m = M.shape
        for i in range(n):
            for j in range(m):
                M[i][j] = not M[i][j]
        return M
    
    def generate_rec(self, M, i, j, n):
        """
        Fonction récursive pour le grignotage.
        
        Args:
            M (np.array): Matrice du puzzle
            i (int): Index de ligne
            j (int): Index de colonne
            n (int): Nombre d'itérations restantes
        """
        if n > 0:
            M[i][j] = False
            C = self.Cross_ind(M, i, j)
            random.shuffle(C)
            c = len(C)

            for k in range(random.randint(1, c - 1)):
                if self.In_able(M, C[k][0], C[k][1]):
                    self.generate_rec(M, C[k][0], C[k][1], n - 1)
    
    def Cross_ind(self, M, i, j):
        """
        Retourne les indices des cellules voisines (en croix).
        
        Args:
            M (np.array): Matrice du puzzle
            i (int): Index de ligne
            j (int): Index de colonne
            
        Returns:
            list: Liste des indices des voisins
        """
        n, m = M.shape

        if i == 0 and j == 0:
            C = [[i, j + 1], [i + 1, j]]
        elif i == n - 1 and j == 0:
            C = [[i - 1, j], [i, j + 1]]
        elif i == 0 and j == m - 1:
            C = [[i, j - 1], [i + 1, j]]
        elif i == n - 1 and j == m - 1:
            C = [[i, j - 1], [i - 1, j]]
        elif i == 0:
            C = [[i, j - 1], [i + 1, j], [i, j + 1]]
        elif j == 0:
            C = [[i - 1, j], [i, j + 1], [i + 1, j]]
        elif i == n - 1:
            C = [[i, j - 1], [i - 1, j], [i, j + 1]]
        elif j == m - 1:
            C = [[i - 1, j], [i, j - 1], [i + 1, j]]
        else:
            C = [[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]]
        
        return C
    
    def noise(self, n, m, r):
        """
        Génère du bruit aléatoire dans une matrice.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            r (float): Intensité du bruit
            
        Returns:
            tuple: (Matrice bruitée, Historique des positions modifiées)
        """
        L = [[False] * m] * n
        M = np.array(L)
        H = []
        
        for _ in range(int((n + m) * r)):
            i, j = self.random_case(n, m)
            H.append([i, j])
            M[i][j] = True
        
        return M, H
    
    def sign(self, x):
        """Retourne le signe d'un nombre."""
        return -1 if x < 0 else 1
    
    def way(self, i1, j1, i2, j2):
        """
        Génère un chemin entre deux points.
        
        Args:
            i1, j1 (int): Point de départ
            i2, j2 (int): Point d'arrivée
            
        Returns:
            list: Liste des points du chemin
        """
        W = []
        i, j = i1, j1
        
        for _ in range(abs(i1 - i2) + abs(j1 - j2)):
            s = random.randint(0, 1)
            if s == 1:
                if i != i2:
                    i += self.sign(i2 - i)
                else:
                    j += self.sign(j2 - j)
            else:
                if j != j2:
                    j += self.sign(j2 - j)
                else:
                    i += self.sign(i2 - i)
            
            W.append([i, j])
        
        return W
    
    def way2(self, i1, j1, i2, j2, n, m):
        """
        Génère un chemin plus complexe entre deux points.
        
        Args:
            i1, j1 (int): Point de départ
            i2, j2 (int): Point d'arrivée
            n, m (int): Dimensions de la grille
            
        Returns:
            list: Liste des points du chemin
        """
        W = []
        i, j = i1, j1
        
        while i != i2 and j != j2:
            di = abs(i2 - i)
            dj = abs(j2 - j)
            a = 1
            b = 0.7
            p = random.random()
            
            if p <= (a + dj) / (di + dj + 2 * a):
                p = random.random()
                if p <= b:
                    if ((self.sign(j2 - j) == 1 and j < m - 1) or 
                        (self.sign(j2 - j) == -1 and j > 0)):
                        j += self.sign(j2 - j)
                    else:
                        j -= self.sign(j2 - j)
                else:
                    if ((self.sign(j2 - j) == 1 and j > 0) or 
                        (self.sign(j2 - j) == -1 and j < m - 1)):
                        j -= self.sign(j2 - j)
                    else:
                        j += self.sign(j2 - j)
            else:
                p = random.random()
                if p <= b:
                    if ((self.sign(i2 - i) == 1 and i < n - 1) or 
                        (self.sign(i2 - i) == -1 and i > 0)):
                        i += self.sign(i2 - i)
                    else:
                        i -= self.sign(i2 - i)
                else:
                    if ((self.sign(i2 - i) == 1 and i > 0) or 
                        (self.sign(i2 - i) == -1 and i < n - 1)):
                        i -= self.sign(i2 - i)
                    else:
                        i += self.sign(i2 - i)
            
            W.append([i, j])
        
        return W
    
    def generate_2(self, n, m, r):
        """
        Génère un puzzle avec la méthode bruit + chemin.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            r (float): Intensité du bruit
            
        Returns:
            np.array: Matrice du puzzle
        """
        M, H = self.noise(n, m, r)
        h = len(H)
        
        for k in range(h - 1):
            i1, j1 = H[k][0], H[k][1]
            i2, j2 = H[k + 1][0], H[k + 1][1]
            W = self.way(i1, j1, i2, j2)
            
            for c in W:
                M[c[0]][c[1]] = True
        
        return M
    
    def bordure(self, nd, nf, md, mf):
        """
        Génère les indices des cellules de bordure.
        
        Args:
            nd, nf (int): Début et fin des lignes
            md, mf (int): Début et fin des colonnes
            
        Returns:
            list: Liste des indices de bordure
        """
        B = []
        
        for j in range(md, mf):
            B.append([nd, j])
        
        for i in range(nd + 1, nf):
            B.append([i, mf - 1])
        
        for j in range(mf - 2, md - 1, -1):
            B.append([nf - 1, j])
        
        for i in range(nf - 1, nd - 1, -1):
            B.append([i, md])
        
        return B
    
    def grignotage(self, n, m, M=None):
        """
        Génère un puzzle par grignotage carré.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            M (np.array, optional): Matrice existante
            
        Returns:
            np.array: Matrice du puzzle
        """
        if M is None:
            L = [[True] * m] * n
            M = np.array(L)
        
        B = self.bordure(0, n, 0, m)
        b = len(B)
        random.shuffle(B)
        
        for i in range(random.randint(0, b - 1)):
            if self.In_able(M, B[i][0], B[i][1]):
                M[B[i][0]][B[i][1]] = False
        
        p = min(n, m)
        
        for k in range(1, p // 2):
            B = self.bordure(k, n - k, k, m - k)
            random.shuffle(B)
            b = len(B)
            
            for l in range(random.randint(4 * b // 5, b - 1)):
                if self.In_able(M, B[l][0], B[l][1]):
                    M[B[l][0]][B[l][1]] = False
        
        return M
    
    def Crack(self, n, m):
        """
        Génère un puzzle avec la méthode Crack.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            
        Returns:
            np.array: Matrice du puzzle
        """
        L = [[True] * m] * n
        M = np.array(L)
        B = self.bordure(0, n, 0, m)
        random.shuffle(B)
        b = len(B)
        
        for k in range(random.randint(1, b // 5)):
            i, j = random.randint(1, n - 1), random.randint(1, m - 1)
            W = self.way(B[k][0], B[k][1], i, j)
            M[B[k][0]][B[k][1]] = False
            
            for w in W:
                if self.In_able(M, w[0], w[1]):
                    M[w[0]][w[1]] = False
        
        return M
    
    def Crack2(self, n, m):
        """
        Génère un puzzle avec la méthode Crack2.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            
        Returns:
            np.array: Matrice du puzzle
        """
        L = [[True] * m] * n
        M = np.array(L)
        B = self.bordure(0, n, 0, m)
        random.shuffle(B)
        b = len(B)
        
        for k in range(random.randint(1, b // 5)):
            i, j = random.randint(1, n - 1), random.randint(1, m - 1)
            W = self.way2(B[k][0], B[k][1], i, j, n, m)
            M[B[k][0]][B[k][1]] = False
            
            for w in W:
                if self.In_able(M, w[0], w[1]):
                    M[w[0]][w[1]] = False
        
        return M
    
    def init_maze(self, n, m):
        """
        Initialise un labyrinthe pour la génération.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            
        Returns:
            tuple: (Matrice du labyrinthe, Historique des positions)
        """
        M = [[0] * m] * n
        M = np.array(M)
        Hist = []
        compteur = 1
        
        for i in range(1, n, 2):
            for j in range(1, m, 2):
                M[i][j] = compteur
                Hist.append([i, j])
                compteur += 1
        
        return M, Hist
    
    def init_maze2(self, n, m, proba=0.1):
        """
        Initialise un labyrinthe avec une probabilité donnée.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            proba (float): Probabilité d'activation
            
        Returns:
            list: Matrice du labyrinthe
        """
        M = [[0] * (2 * m)] * (2 * n)
        M = np.array(M)
        
        for i in range(1, 2 * n, 2):
            for j in range(1, 2 * m, 2):
                rand = random.random()
                if rand < proba:
                    M[i][j] = 1
        
        M = M.tolist()
        
        for k in range(2 * n, n, -1):
            self.supr_l(M, random.randint(0, k - 1))
        
        for k in range(2 * m, m, -1):
            self.supr_col(M, random.randint(0, k - 1))
        
        return M
    
    def Histo(self, M):
        """
        Crée un historique des positions non nulles.
        
        Args:
            M (np.array): Matrice à analyser
            
        Returns:
            list: Historique des positions
        """
        n, m = M.shape
        Hist = []
        
        for i in range(n):
            for j in range(m):
                if M[i][j] != 0:
                    Hist.append([i, j])
        
        return Hist
    
    def supr_col(self, M, ind):
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
                M[i][j], M[i][j + 1] = M[i][j + 1], M[i][j]
        
        for i in range(n):
            M[i].pop()
        
        return M
    
    def supr_l(self, M, ind):
        """
        Supprime une ligne de la matrice.
        
        Args:
            M (list): Matrice
            ind (int): Index de la ligne à supprimer
            
        Returns:
            list: Matrice modifiée
        """
        n = len(M)
        
        for i in range(ind, n - 1):
            M[i], M[i + 1] = M[i + 1], M[i]
        
        M.pop()
        return M
    
    def link(self, M, i, j):
        """
        Trouve les liens possibles depuis une cellule.
        
        Args:
            M (np.array): Matrice du labyrinthe
            i, j (int): Position de la cellule
            
        Returns:
            list: Liste des liens possibles
        """
        n, m = M.shape
        L = []
        val = M[i][j]
        
        if M[i][j] + M[(i + 2) % n][j] != val and M[i][j] + M[(i + 2) % n][j] < 2 * val:
            L.append([i + 2, j])
        
        if M[i][j] + M[(i - 2) % n][j] != val and M[i][j] + M[(i - 2) % n][j] < 2 * val:
            L.append([i - 2, j])
        
        if M[i][j] + M[i][(j + 2) % m] != val and M[i][j] + M[i][(j + 2) % m] < 2 * val:
            L.append([i, j + 2])
        
        if M[i][j] + M[i][(j - 2) % m] != val and M[i][j] + M[i][(j - 2) % m] < 2 * val:
            L.append([i, j - 2])
        
        return L
    
    def generate_maze(self, n, m):
        """
        Génère un puzzle avec la méthode du labyrinthe.
        
        Args:
            n (int): Nombre de lignes
            m (int): Nombre de colonnes
            
        Returns:
            np.array: Matrice du puzzle
        """
        M, Hist = self.init_maze(n, m)
        h = len(Hist)
        random.shuffle(Hist)
        S = []
        
        while h != 0:
            ind = random.randint(0, h - 1)
            Hist[ind][0], Hist[-1][0] = Hist[-1][0], Hist[ind][0]
            Hist[ind][1], Hist[-1][1] = Hist[-1][1], Hist[ind][1]
            i, j = Hist[-1][0], Hist[-1][1]
            L = self.link(M, i, j)
            
            if len(L) == 0:
                Hist.pop()
                h -= 1
            else:
                random.shuffle(L)
                i_p, j_p = L[0][0], L[0][1]
                val = M[i][j]
                Hist.append([(i + i_p) // 2, (j + j_p) // 2])
                h += 1
                M[(i + i_p) // 2][(j + j_p) // 2] = val
                self.homog(M, i_p, j_p, val)
        
        return M
    
    def homog(self, M, i, j, val):
        """
        Uniformise les valeurs dans une zone connectée.
        
        Args:
            M (np.array): Matrice à modifier
            i, j (int): Position de départ
            val: Valeur à propager
        """
        M[i][j] = val
        Cross = self.Cross_ind(M, i, j)
        
        for i_p, j_p in Cross:
            if M[i_p][j_p] != 0 and M[i_p][j_p] != val:
                self.homog(M, i_p, j_p, val)
    
    def In_able(self, M, i, j):
        """
        Vérifie si une cellule peut être retirée (grignotée).
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de la cellule
            
        Returns:
            bool: True si la cellule peut être retirée
        """
        C = self.Cycle(M, i, j)
        n = len(C)
        precedent = C[0]
        var = 0
        
        for k in range(1, n + 1):
            if (precedent and (not C[k % n])) or ((not precedent) and C[k % n]):
                var += 1
            precedent = C[k % n]
        
        if (n == 5 or n == 3) and var == 2:
            var += 3
        
        C = self.Cross(M, i, j)
        c = len(C)
        is_Cross = (not C[0]) or (not C[1 % c]) or (not C[2 % c]) or (not C[3 % c])
        
        return var <= 2 and (is_Cross or n < 6) and M[i][j]
    
    def Cross(self, M, i, j):
        """
        Retourne les valeurs des cellules voisines (en croix).
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de la cellule
            
        Returns:
            np.array: Valeurs des voisins
        """
        n, m = M.shape

        if i == 0 and j == 0:
            C = np.array([M[i][j + 1], M[i + 1][j]])
        elif i == n - 1 and j == 0:
            C = np.array([M[i - 1][j], M[i][j + 1]])
        elif i == 0 and j == m - 1:
            C = np.array([M[i][j - 1], M[i + 1][j]])
        elif i == n - 1 and j == m - 1:
            C = np.array([M[i][j - 1], M[i - 1][j]])
        elif i == 0:
            C = np.array([M[i][j - 1], M[i + 1][j], M[i][j + 1]])
        elif j == 0:
            C = np.array([M[i - 1][j], M[i][j + 1], M[i + 1][j]])
        elif i == n - 1:
            C = np.array([M[i][j - 1], M[i - 1][j], M[i][j + 1]])
        elif j == m - 1:
            C = np.array([M[i - 1][j], M[i][j - 1], M[i + 1][j]])
        else:
            C = np.array([M[i - 1][j], M[i][j + 1], M[i + 1][j], M[i][j - 1]])
        
        return C
    
    def Cycle(self, M, i, j):
        """
        Retourne les valeurs des cellules voisines (en cycle).
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de la cellule
            
        Returns:
            np.array: Valeurs des voisins
        """
        n, m = M.shape

        if i == 0 and j == 0:
            C = np.array([M[i][j + 1], M[i + 1][j + 1], M[i + 1][j]])
        elif i == n - 1 and j == 0:
            C = np.array([M[i - 1][j], M[i - 1][j + 1], M[i][j + 1]])
        elif i == 0 and j == m - 1:
            C = np.array([M[i][j - 1], M[i + 1][j - 1], M[i + 1][j]])
        elif i == n - 1 and j == m - 1:
            C = np.array([M[i][j - 1], M[i - 1][j - 1], M[i - 1][j]])
        elif i == 0:
            C = np.array([M[i][j - 1], M[i + 1][j - 1], M[i + 1][j], 
                         M[i + 1][j + 1], M[i][j + 1]])
        elif j == 0:
            C = np.array([M[i - 1][j], M[i - 1][j + 1], M[i][j + 1], 
                         M[i + 1][j + 1], M[i + 1][j]])
        elif i == n - 1:
            C = np.array([M[i][j - 1], M[i - 1][j - 1], M[i - 1][j], 
                         M[i - 1][j + 1], M[i][j + 1]])
        elif j == m - 1:
            C = np.array([M[i - 1][j], M[i - 1][j - 1], M[i][j - 1], 
                         M[i + 1][j - 1], M[i + 1][j]])
        else:
            C = np.array([M[i - 1][j - 1], M[i - 1][j], M[i - 1][j + 1], 
                         M[i][j + 1], M[i + 1][j + 1], M[i + 1][j], 
                         M[i + 1][j - 1], M[i][j - 1]])
        
        return C
    
    def random_case(self, n, m):
        """
        Génère une position aléatoire dans la grille.
        
        Args:
            n, m (int): Dimensions de la grille
            
        Returns:
            tuple: (i, j) position aléatoire
        """
        return random.randint(0, n - 1), random.randint(0, m - 1)


class SlitherlinkRules:
    """Classe pour vérifier les règles du Slitherlink."""
    
    def Rule1(self, M, NA):
        """
        Vérifie la règle 1 du Slitherlink.
        
        Args:
            M (np.array): Matrice du puzzle
            NA (np.array): Matrice des nombres attendus
            
        Returns:
            bool: True si la règle est respectée
        """
        n, m = M.shape
        i = 0
        res = True
        
        while i < n and res:
            j = 0
            while j < m and res:
                C = self.Cross(M, i, j)
                c = len(C)
                p = 0
                
                if M[i][j]:
                    p = 4 - c
                    for k in range(c):
                        if not C[k]:
                            p += 1
                    
                    if NA[i][j] != p:
                        res = False
                
                if not M[i][j]:
                    for k in range(c):
                        if C[k]:
                            p += 1
                    
                    if NA[i][j] != p:
                        res = False
                
                j += 1
            i += 1
        
        return res
    
    def Rule1_ind(self, M, i, j, NA):
        """
        Vérifie la règle 1 pour une cellule spécifique.
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de la cellule
            NA (np.array): Matrice des nombres attendus
            
        Returns:
            bool: True si la règle est respectée
        """
        res = True
        C = self.Cross(M, i, j)
        c = len(C)
        p = 0
        
        if M[i][j]:
            p = 4 - c
            for k in range(c):
                if not C[k]:
                    p += 1
            
            if NA[i][j] != p:
                res = False
        
        if not M[i][j]:
            for k in range(c):
                if C[k]:
                    p += 1
            
            if NA[i][j] != p:
                res = False
        
        return res
    
    def NA(self, M):
        """
        Calcule la matrice des nombres attendus.
        
        Args:
            M (np.array): Matrice du puzzle
            
        Returns:
            np.array: Matrice des nombres attendus
        """
        n, m = M.shape
        L = [[0] * m] * n
        NA = np.array(L)
        
        for i in range(n):
            for j in range(m):
                C = self.Cross(M, i, j)
                c = len(C)
                n_val = 0
                
                if M[i][j]:
                    n_val = 4 - c
                    for k in range(c):
                        if not C[k]:
                            n_val += 1
                    NA[i][j] = n_val
                
                if not M[i][j]:
                    for k in range(c):
                        if C[k]:
                            n_val += 1
                    NA[i][j] = n_val
        
        return NA
    
    def Hist(self, M, i, j, L):
        """
        Crée un historique des cellules connectées.
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de départ
            L (list): Liste pour stocker l'historique
        """
        L.append([i, j])
        C = self.Cross_ind(M, i, j)
        
        for ind in C:
            if (M[ind[0]][ind[1]] == M[i][j] and 
                not ([ind[0], ind[1]] in L)):
                self.Hist(M, ind[0], ind[1], L)
    
    def Couche_ext(self, M):
        """
        Crée une couche externe pour la vérification.
        
        Args:
            M (np.array): Matrice du puzzle
            
        Returns:
            np.array: Matrice de la couche externe
        """
        n, m = M.shape
        Couche = [[False] * (m + 2)] * (n + 2)
        Couche = np.array(Couche)
        
        for i in range(n + 2):
            for j in range(m + 2):
                if i > 0 and i <= n and j > 0 and j <= m:
                    Couche[i][j] = M[i - 1][j - 1]
        
        return Couche
    
    def Verif(self, M):
        """
        Vérifie la validité du puzzle.
        
        Args:
            M (np.array): Matrice du puzzle
            
        Returns:
            bool: True si le puzzle est valide
        """
        n, m = M.shape
        i, j = 0, 0
        found = False
        
        while i < n and not found:
            j = 0
            while j < m and not found:
                if M[i, j]:
                    found = True
                    i, j = i - 1, j - 1
                j += 1
            i += 1

        if not found:
            return True
        else:
            L_int = []
            self.Hist(M, i, j, L_int)
            
            L_ext = []
            Couche = self.Couche_ext(M)
            self.Hist(Couche, 0, 0, L_ext)
            
            return len(L_int) + len(L_ext) == (n + 2) * (m + 2)
    
    def in_pas_diag(self, M):
        """
        Vérifie l'absence de connexions diagonales.
        
        Args:
            M (np.array): Matrice du puzzle
            
        Returns:
            bool: True s'il n'y a pas de connexions diagonales
        """
        res = True
        n, m = M.shape
        
        i, j = 0, 0
        
        while i < n - 1 and res:
            j = 0
            while j < m - 1 and res:
                Carre = [[M[i][j], M[i][j + 1]], 
                         [M[i + 1][j], M[i + 1][j + 1]]]
                
                if (Carre == [[True, False], [False, True]] or 
                    Carre == [[False, True], [True, False]]):
                    res = False
                
                j += 1
            i += 1
        
        return res
    
    def Cross(self, M, i, j):
        """
        Retourne les valeurs des cellules voisines (en croix).
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de la cellule
            
        Returns:
            np.array: Valeurs des voisins
        """
        n, m = M.shape

        if i == 0 and j == 0:
            C = np.array([M[i][j + 1], M[i + 1][j]])
        elif i == n - 1 and j == 0:
            C = np.array([M[i - 1][j], M[i][j + 1]])
        elif i == 0 and j == m - 1:
            C = np.array([M[i][j - 1], M[i + 1][j]])
        elif i == n - 1 and j == m - 1:
            C = np.array([M[i][j - 1], M[i - 1][j]])
        elif i == 0:
            C = np.array([M[i][j - 1], M[i + 1][j], M[i][j + 1]])
        elif j == 0:
            C = np.array([M[i - 1][j], M[i][j + 1], M[i + 1][j]])
        elif i == n - 1:
            C = np.array([M[i][j - 1], M[i - 1][j], M[i][j + 1]])
        elif j == m - 1:
            C = np.array([M[i - 1][j], M[i][j - 1], M[i + 1][j]])
        else:
            C = np.array([M[i - 1][j], M[i][j + 1], M[i + 1][j], M[i][j - 1]])
        
        return C
    
    def Cross_ind(self, M, i, j):
        """
        Retourne les indices des cellules voisines (en croix).
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de la cellule
            
        Returns:
            list: Liste des indices des voisins
        """
        n, m = M.shape

        if i == 0 and j == 0:
            C = [[i, j + 1], [i + 1, j]]
        elif i == n - 1 and j == 0:
            C = [[i - 1, j], [i, j + 1]]
        elif i == 0 and j == m - 1:
            C = [[i, j - 1], [i + 1, j]]
        elif i == n - 1 and j == m - 1:
            C = [[i, j - 1], [i - 1, j]]
        elif i == 0:
            C = [[i, j - 1], [i + 1, j], [i, j + 1]]
        elif j == 0:
            C = [[i - 1, j], [i, j + 1], [i + 1, j]]
        elif i == n - 1:
            C = [[i, j - 1], [i - 1, j], [i, j + 1]]
        elif j == m - 1:
            C = [[i - 1, j], [i, j - 1], [i + 1, j]]
        else:
            C = [[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]]
        
        return C


class SlitherlinkVisualizer:
    """Classe pour visualiser les puzzles Slitherlink."""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def visualize_slitherlink(self, M, title="Slitherlink Puzzle"):
        """
        Visualise un puzzle Slitherlink.
        
        Args:
            M (np.array): Matrice booléenne (True = intérieur, False = extérieur)
            title (str): Titre du graphique
        """
        n, m = M.shape
        
        # Créer la figure avec un ratio approprié
        self.fig, self.ax = plt.subplots(figsize=(m * 0.8, n * 0.8))
        
        # Configuration de base
        self.ax.set_xlim(-1, m)
        self.ax.set_ylim(-1, n)
        self.ax.invert_yaxis()  # Pour avoir (0,0) en haut à gauche
        
        # Dessiner la grille de points
        for i in range(n + 1):
            for j in range(m + 1):
                self.ax.plot(j - 0.5, i - 0.5, 'ko', markersize=4)
        
        # Calculer et afficher les chiffres pour chaque cellule
        for i in range(n):
            for j in range(m):
                # Calculer le nombre d'arêtes requises pour cette cellule
                nb_edges = self.calculate_edges_needed(M, i, j)
                
                # Afficher le chiffre au centre de la cellule
                if nb_edges is not None:
                    self.ax.text(j, i, str(nb_edges), 
                               ha='center', va='center', 
                               fontsize=14)
        
        # Dessiner les lignes de la solution (optionnel - pour debug)
        self.draw_solution_outline(M)
        
        # Configuration finale
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title, fontsize=16)
        
        # Supprimer les bordures
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_edges_needed(self, M, i, j):
        """
        Calcule le nombre d'arêtes nécessaires autour d'une cellule.
        
        Args:
            M (np.array): Matrice du puzzle
            i, j (int): Position de la cellule
            
        Returns:
            int or None: Nombre d'arêtes nécessaires, ou None si pas d'indice
        """
        n, m = M.shape
        
        # Obtenir les voisins (haut, droite, bas, gauche)
        neighbors = []
        
        # Haut
        if i > 0:
            neighbors.append(M[i - 1, j])
        else:
            neighbors.append(False)  # Extérieur du puzzle
        
        # Droite  
        if j < m - 1:
            neighbors.append(M[i, j + 1])
        else:
            neighbors.append(False)
        
        # Bas
        if i < n - 1:
            neighbors.append(M[i + 1, j])
        else:
            neighbors.append(False)
        
        # Gauche
        if j > 0:
            neighbors.append(M[i, j - 1])
        else:
            neighbors.append(False)
        
        # Compter les transitions entre intérieur et extérieur
        current_cell = M[i, j]
        edge_count = 0
        
        for neighbor in neighbors:
            if current_cell != neighbor:  # Transition = arête nécessaire
                edge_count += 1
        
        return edge_count if edge_count > 0 else None
    
    def draw_solution_outline(self, M, show_solution=False):
        """
        Dessine le contour de la solution.
        
        Args:
            M (np.array): Matrice du puzzle
            show_solution (bool): Si True, affiche la solution
        """
        if not show_solution:
            return
            
        n, m = M.shape
        
        # Dessiner les arêtes horizontales
        for i in range(n + 1):
            for j in range(m):
                # Vérifier s'il y a une transition verticale
                top_cell = M[i - 1, j] if i > 0 else False
                bottom_cell = M[i, j] if i < n else False
                
                if top_cell != bottom_cell:
                    self.ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], 'r-', linewidth=3)
        
        # Dessiner les arêtes verticales
        for i in range(n):
            for j in range(m + 1):
                # Vérifier s'il y a une transition horizontale
                left_cell = M[i, j - 1] if j > 0 else False
                right_cell = M[i, j] if j < m else False
                
                if left_cell != right_cell:
                    self.ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], 'r-', linewidth=3)
    
    def visualize_slitherlink_with_solution(self, M, title="Slitherlink avec Solution"):
        """
        Visualise le puzzle avec la solution.
        
        Args:
            M (np.array): Matrice du puzzle
            title (str): Titre du graphique
        """
        n, m = M.shape
        
        self.fig, self.ax = plt.subplots(figsize=(m * 0.8, n * 0.8))
        
        self.ax.set_xlim(-1, m)
        self.ax.set_ylim(-1, n)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        
        # Grille de points
        for i in range(n + 1):
            for j in range(m + 1):
                self.ax.plot(j - 0.5, i - 0.5, 'ko', markersize=4)
        
        # Chiffres
        for i in range(n):
            for j in range(m):
                nb_edges = self.calculate_edges_needed(M, i, j)
                if nb_edges is not None:
                    self.ax.text(j, i, str(nb_edges), 
                               ha='center', va='center', 
                               fontsize=14)
        
        # Solution en rouge
        self.draw_solution_outline(M, show_solution=True)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title, fontsize=16)
        
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def show_anim(self, L, interval=200):
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
    
    def show(self, M):
        """
        Affiche une matrice simple.
        
        Args:
            M (np.array): Matrice à afficher
        """
        fig, ax = plt.subplots()
        img = ax.imshow(M)
        plt.show()
    
    def show2(self, M, NA):
        """
        Affiche deux matrices côte à côte.
        
        Args:
            M (np.array): Première matrice
            NA (np.array): Deuxième matrice
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        img1 = ax1.imshow(M)
        img2 = ax2.imshow(NA)
        plt.show()
    
    def show_n(self, L, n, m):
        """
        Affiche plusieurs matrices dans une grille.
        
        Args:
            L (list): Liste de matrices
            n, m (int): Dimensions de la grille d'affichage
        """
        l = len(L)
        fig, axs = plt.subplots(n, m)
        
        k = 0
        for i in range(n):
            for j in range(m):
                if k < l:
                    axs[i, j].imshow(L[k])
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                k += 1
        
        plt.show()


class SlitherlinkSolver:
    """Classe pour résoudre les puzzles Slitherlink."""
    
    def __init__(self):
        self.generator = SlitherlinkGenerator()
        self.rules = SlitherlinkRules()
    
    def List_3x3(self):
        """Génère toutes les matrices 3x3 possibles."""
        L = []
        
        for i in range(512):
            s = list(format(i, '09b'))
            
            for p in range(9):
                if s[p] == '0':
                    s[p] = False
                else:
                    s[p] = True

            M = np.array([[s[8], s[7], s[6]], 
                         [s[1], s[0], s[5]], 
                         [s[2], s[3], s[4]]])
            
            L.append(M)
        
        return L
    
    def List_nxm(self, n, m):
        """Génère toutes les matrices n x m possibles."""
        L = []
        f = '0' + str(n * m) + 'b'
        
        for k in range(2 ** (n * m)):
            s = list(format(k, f))
            
            for p in range(n * m):
                if s[p] == '0':
                    s[p] = False
                else:
                    s[p] = True
                    
            M = [[False] * m] * n
            M = np.array(M)
            somme = 0
            
            for i in range(n):
                for j in range(m):
                    M[i, j] = s[somme]
                    somme += 1
                    
            L.append(M)
        
        return L
    
    def filtre_equiv(self, L):
        """Filtre les matrices équivalentes par rotation/symétrie."""
        Hist = []
        L_filtre = []
        
        for M in L:
            Ml = M.tolist()
            
            if not Ml in Hist:
                Hist.append(Ml)
                Hist.append(np.rot90(M, k=1).tolist())
                Hist.append(np.rot90(M, k=2).tolist())
                Hist.append(np.rot90(M, k=3).tolist())
                
                Mt = np.transpose(M)
                Hist.append(Mt.tolist())
                Hist.append(np.rot90(Mt, k=1).tolist())
                Hist.append(np.rot90(Mt, k=2).tolist())
                Hist.append(np.rot90(Mt, k=3).tolist())
                
                L_filtre.append(M)
        
        return L_filtre
    
    def filtre_puzzle(self, L):
        """Filtre les matrices valides pour le Slitherlink."""
        L_filtre = []
        
        for M in L:
            if self.rules.Verif(M):
                L_filtre.append(M)
        
        return L_filtre
    
    def filtre_diag(self, L):
        """Filtre les matrices sans connexions diagonales."""
        L_filtre = []
        
        for M in L:
            if self.rules.in_pas_diag(M):
                L_filtre.append(M)
        
        return L_filtre
    
    def filtre_Point(self, L, marge):
        """Filtre les matrices selon un critère de points."""
        L_filtre = []
        
        for M in L:
            if self.Point(M) <= marge:
                L_filtre.append(M)
        
        return L_filtre
    
    def List_3x3_able(self):
        """Génère les matrices 3x3 valides pour le grignotage."""
        L = []
        
        for i in range(512):
            s = list(format(i, '09b'))
            
            for p in range(9):
                if s[p] == '0':
                    s[p] = False
                else:
                    s[p] = True

            M = np.array([[s[8], s[7], s[6]], 
                         [s[1], s[0], s[5]], 
                         [s[2], s[3], s[4]]])
            
            precedent = s[8]
            var = 0
            k = 1
            
            while k < 9 and var <= 2:
                if precedent != s[k]:
                    var += 1
                
                precedent = s[k]
                k += 1

            is_Cross = (s[0] == s[1]) or (s[0] == s[3]) or (s[0] == s[5]) or (s[0] == s[7])
            
            if var <= 2 and is_Cross:
                L.append(M)
        
        return L
    
    def List_3x3_puzzle(self):
        """Génère les matrices 3x3 valides pour le puzzle."""
        L = []
        
        for i in range(512):
            s = list(format(i, '09b'))
            
            for p in range(9):
                if s[p] == '0':
                    s[p] = False
                else:
                    s[p] = True

            M = np.array([[s[8], s[7], s[6]], 
                         [s[1], s[0], s[5]], 
                         [s[2], s[3], s[4]]])
            
            if self.rules.Verif(M):
                L.append(M)
        
        return L
    
    def nb_in(self, M):
        """Compte le nombre de True dans une matrice."""
        n, m = M.shape
        nb = 0
        
        for i in range(n):
            for j in range(m):
                if M[i, j]:
                    nb += 1
        
        return nb
    
    def List_in(self, L):
        """Compte le nombre de True pour chaque matrice dans une liste."""
        L_in = []
        
        for M in L:
            L_in.append(self.nb_in(M))
        
        return L_in
    
    def Sort(self, L, L_in):
        """Trie une liste de matrices selon le nombre de True."""
        n = len(L)
        maxi = max(L_in)
        
        L_sort = []
        
        for i in range(maxi + 1):
            for j in range(n):
                if L_in[j] == i:
                    L_sort.append(L[j])
        
        return np.array(L_sort)
    
    def Tri(self, L):
        """Trie une liste de matrices."""
        return self.Sort(L, self.List_in(L))
    
    def Ligne(self, M, i):
        """Retourne une ligne d'une matrice."""
        return M[i]
    
    def Colonne(self, M, j):
        """Retourne une colonne d'une matrice."""
        n = len(M)
        C = []
        
        for i in range(n):
            C.append(M[i][j])
        
        return C
    
    def Moyenne(self, L, M):
        """Calcule la moyenne pondérée d'une liste de booléens."""
        n = len(M)
        moy = 0
        
        for Bool in L:
            if Bool:
                moy += 1 / n
            else:
                moy -= 1 / n
        
        return moy
    
    def Point(self, M):
        """Calcule un score de points pour une matrice."""
        n, m = M.shape
        pts = 0
        
        for i in range(n):
            pts += abs((self.Moyenne(self.Ligne(M, i), M)) ** 2)
        
        for j in range(m):
            pts += abs((self.Moyenne(self.Colonne(M, j), M)) ** 2)
        
        pts = pts / (n + m)
        return pts


# Fonction principale de test
def main():
    """Fonction principale pour tester le programme."""
    print("Testing Slitherlink Generator and Visualizer")
    
    # Créer les instances des classes
    generator = SlitherlinkGenerator()
    visualizer = SlitherlinkVisualizer()
    solver = SlitherlinkSolver()
    
    # Générer un petit puzzle simple
    print("Generating simple 3x3 puzzle...")
    M_simple = np.array([
        [True, True, False],
        [True, False, False],
        [False, False, False]
    ])
    
    # Visualiser le puzzle
    visualizer.visualize_slitherlink(M_simple, "Test Simple")
    visualizer.visualize_slitherlink_with_solution(M_simple, "Test Simple avec Solution")
    
    # Générer un puzzle plus complexe
    print("Generating complex 11x11 puzzle...")
    try:
        M_complex = generator.generate_maze(11, 11)
        visualizer.visualize_slitherlink(M_complex, "Générateur Récursif")
        visualizer.visualize_slitherlink_with_solution(M_complex, "Générateur Récursif avec Solution")
    except Exception as e:
        print(f"Error generating complex puzzle: {e}")
    
    # Tester le solveur avec des matrices 3x3
    print("Testing solver with 3x3 matrices...")
    matrices_3x3 = solver.List_3x3_able()
    print(f"Generated {len(matrices_3x3)} valid 3x3 matrices")
    
    # Afficher quelques matrices
    if len(matrices_3x3) > 0:
        visualizer.show_n(matrices_3x3[:9], 3, 3)


if __name__ == "__main__":
    main()
