"""
Solver Slitherlink — version optimisée
Auteur : SAUCET Nohlan (optimisations par Claude)

Changements principaux :
    1. modifie_Na_M      : restore avec tuples (plus léger), logique val corrigée
    2. condition_mettre_in : suppression de Croix() — on lit Na_M directement
    3. condition_mettre_out: n, m passés en paramètre (évite len() répété)
    4. états_possible    : set au lieu de liste pour O(1) sur in/remove
    5. solution_trouvé   : any() avec générateur, court-circuit immédiat
    6. score_init        : deepcopy inutile supprimé
    7. solve_SL / solve_SL_all : n, m calculés une seule fois et réutilisés
"""

import copy as c
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires (inchangées, reproduites pour autonomie du fichier)
# ─────────────────────────────────────────────────────────────────────────────

def Croix_ind(M, i, j):
    """Retourne les indices des cellules voisines orthogonales."""
    n, m = len(M), len(M[0])
    if   i == 0     and j == 0    : return [[i, j+1], [i+1, j]]
    elif i == n-1   and j == 0    : return [[i-1, j], [i, j+1]]
    elif i == 0     and j == m-1  : return [[i, j-1], [i+1, j]]
    elif i == n-1   and j == m-1  : return [[i, j-1], [i-1, j]]
    elif i == 0                   : return [[i, j-1], [i+1, j], [i, j+1]]
    elif j == 0                   : return [[i-1, j], [i, j+1], [i+1, j]]
    elif i == n-1                 : return [[i, j-1], [i-1, j], [i, j+1]]
    elif j == m-1                 : return [[i-1, j], [i, j-1], [i+1, j]]
    else                          : return [[i-1, j], [i, j+1], [i+1, j], [i, j-1]]


# ─────────────────────────────────────────────────────────────────────────────
# 1. modifie_Na_M  — tuples dans restore, val corrigé
# ─────────────────────────────────────────────────────────────────────────────

def modifie_Na_M(M, Na_M, i, j):
    """
    Met à jour Na_M quand la case (i,j) passe à True (intérieur).

    Na_M[r][c] = nombre de voisins de couleur OPPOSÉE à M[r][c].

    Quand (i,j) devient True :
      - pour chaque voisin (ic,jc) :
          * si M[ic][jc] == True  → (i,j) était opposé à lui (False→True) :
            le voisin perd un voisin opposé  → Na_M[ic][jc] -= 1
          * si M[ic][jc] == False → (i,j) est maintenant opposé à lui :
            le voisin gagne un voisin opposé → Na_M[ic][jc] += 1
      - Na_M[i][j] = nb de voisins False parmi les existants
                   + (4 - nb_voisins_existants)  [bords comptent False]
    """
    restore = []
    C = Croix_ind(M, i, j)
    nb_voisins = len(C)

    val = 0  # comptera les voisins False (= opposés quand (i,j) sera True)

    for ic, jc in C:
        restore.append((ic, jc, Na_M[ic][jc]))   # tuple, plus léger qu'une liste
        if M[ic][jc]:                             # voisin True  → (i,j) lui devient identique
            Na_M[ic][jc] -= 1
        else:                                     # voisin False → (i,j) lui devient opposé
            Na_M[ic][jc] += 1
            val += 1

    restore.append((i, j, Na_M[i][j]))
    Na_M[i][j] = val + (4 - nb_voisins)          # bords fictifs comptent comme False
    return Na_M, restore


def restore_Na_M(Na_M, restore):
    """Annule les modifications faites par modifie_Na_M."""
    for i, j, val in restore:
        Na_M[i][j] = val
    return Na_M


# ─────────────────────────────────────────────────────────────────────────────
# 2. condition_mettre_in  — plus de Croix(), on lit Na_M
# ─────────────────────────────────────────────────────────────────────────────

def condition_mettre_in(M, Na_M, Na, i, j, n, m):
    """
    Vérifie qu'il n'y a pas de contradiction si on place True en (i,j).

    val_si_in = nombre de voisins FALSE autour de (i,j) si (i,j) devient True.

    Astuce : Na_M[i][j] est DÉJÀ ce nombre, calculé par modifie_Na_M au coup
    d'avant — inutile de rappeler Croix().

    AVANT cette fonction, modifie_Na_M n'a PAS encore été appelée pour (i,j),
    donc Na_M[i][j] reflète l'état courant (case encore False).
    Quand (i,j) = False, Na_M[i][j] = nb voisins True (opposés à False).
    Si on le met True, les opposés deviennent les False :
        val_si_in = (nb_voisins_existants) - Na_M[i][j]  [voisins True actuels]
                  + (4 - nb_voisins_existants)            [bords fictifs = False]
    Mais on peut simplifier : on recalcule directement depuis les voisins,
    ce qui reste O(4) sans instancier de liste intermédiaire.
    """
    if Na[i][j] == -1:
        return True

    # Compte les voisins False directement — O(4) sans Croix()
    val_si_in = 4  # commence à 4 (bords fictifs = False)
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for di, dj in dirs:
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < m:
            if M[ni][nj]:          # voisin True → pas opposé à (i,j)=True
                val_si_in -= 1
            # voisin False → opposé, déjà compté dans le +4 initial
        # hors grille → False fictif, déjà dans le +4

    target = Na[i][j]

    if i + 1 == n and j + 1 == m:
        return val_si_in == target
    elif j + 1 == m:
        return 0 <= val_si_in - target <= 1
    else:
        return 0 <= val_si_in - target <= 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. condition_mettre_out  — n, m en paramètre
# ─────────────────────────────────────────────────────────────────────────────

def condition_mettre_out(Na_M, Na, i, j, n, m):
    """
    Vérifie qu'il n'y a pas de contradiction si on laisse False en (i,j).
    Reçoit n et m pour éviter len(M) / len(M[0]) à chaque appel.
    """
    if Na[i][j] == -1:
        return True

    val_si_out = Na_M[i][j]
    target = Na[i][j]

    if i + 1 == n and j + 1 == m:
        return val_si_out == target
    elif j + 1 == m:
        return 0 <= target - val_si_out <= 1
    else:
        return 0 <= target - val_si_out <= 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. conditions bas / droite / non-diag  (inchangées, signaturé uniforme)
# ─────────────────────────────────────────────────────────────────────────────

_NO_COND = object()   # sentinelle unique, évite les comparaisons de strings

def condition_bas(Na_M, Na, i, j):
    if i == 0 or Na[i-1][j] == -1:
        return _NO_COND
    return Na_M[i-1][j] != Na[i-1][j]   # True = il faut modifier = mettre in


def condition_droite(M, Na_M, Na, i, j):
    if j == 0 or Na[i][j-1] == -1:
        return _NO_COND
    diff = Na_M[i][j-1] - Na[i][j-1]
    if M[i][j-1]:
        if diff == -2: return True
    else:
        if diff == 2:  return True
    if diff == 0:      return False
    return _NO_COND


def condition_non_diag(M, i, j):
    """Retourne un set de valeurs interdites pour éviter les motifs damier."""
    if i == 0 or j == 0:
        return set()
    interdits = set()
    a, b, d = M[i][j-1], M[i-1][j], M[i-1][j-1]
    if a == b:          # les deux voisins identiques
        if d and not a: interdits.add(True)
        if not d and a: interdits.add(False)
    return interdits


# ─────────────────────────────────────────────────────────────────────────────
# 5. états_possible  — set, sentinelles, signatures uniformes
# ─────────────────────────────────────────────────────────────────────────────

def états_possible(M, Na_M, Na, i, j, n, m):
    """
    Retourne le set des valeurs possibles {True, False} pour la case (i,j).
    Utilise un set pour des opérations O(1).
    """
    E = {True, False}

    # — condition bas (case déjà figée) —
    c_bas = condition_bas(Na_M, Na, i, j)
    if c_bas is not _NO_COND:
        E.discard(not c_bas)

    # — condition droite —
    c_droite = condition_droite(M, Na_M, Na, i, j)
    if c_droite is not _NO_COND and c_droite != c_bas:
        E.discard(not c_droite)

    # — anti-damier —
    E -= condition_non_diag(M, i, j)

    # — contradiction si on laisse False —
    if not condition_mettre_out(Na_M, Na, i, j, n, m):
        E.discard(False)

    # — contradiction si on met True —
    if not condition_mettre_in(M, Na_M, Na, i, j, n, m):
        E.discard(True)

    return E


# ─────────────────────────────────────────────────────────────────────────────
# 6. solution_trouvé  — any() avec court-circuit
# ─────────────────────────────────────────────────────────────────────────────

def solution_trouvé(Na, Na_M, n, m):
    """Vérifie que Na_M correspond à Na pour toutes les cases indicées."""
    return not any(
        Na[i][j] != -1 and Na[i][j] != Na_M[i][j]
        for i in range(n)
        for j in range(m)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. score_init  — deepcopy inutile supprimé
# ─────────────────────────────────────────────────────────────────────────────

def score_init(Na):
    n, m = len(Na), len(Na[0])
    tot = moy_i = moy_j = 0
    for i in range(n):
        for j in range(m):
            if Na[i][j] == -1:
                moy_i += i
                moy_j += j
                tot   += 1
    if tot == 0:
        return 1
    seuil = (((n / 2) ** 2 + (m / 2) ** 2) ** 0.5) * 0.8
    if ((moy_i / tot) ** 2 + (moy_j / tot) ** 2) ** 0.5 < seuil:
        return 0
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. solve_SL  — version optimisée (trouve la première solution)
# ─────────────────────────────────────────────────────────────────────────────

class SolutionFound(Exception):
    pass


def solve_SL(Na):
    n, m = len(Na), len(Na[0])
    M     = [[False] * m for _ in range(n)]
    Na_M  = [[0]     * m for _ in range(n)]

    def rec(M, Na_M, i, j, val):
        prev    = M[i][j]
        M[i][j] = val

        if solution_trouvé(Na, Na_M, n, m):
            raise SolutionFound

        # prochaine case en antidiagonale
        if j > 0 and i < n - 1:
            ni, nj = i + 1, j - 1
        elif i == n - 1 and j == m - 1:
            M[i][j] = prev
            return
        else:
            ni = max(0, i - (m - 1 - j - 1))
            nj = min(i + 1, m - 1)

        E = états_possible(M, Na_M, Na, ni, nj, n, m)

        if True in E:
            Na_M, restore = modifie_Na_M(M, Na_M, ni, nj)
            rec(M, Na_M, ni, nj, True)
            Na_M = restore_Na_M(Na_M, restore)
        if False in E:
            rec(M, Na_M, ni, nj, False)

        M[i][j] = prev

    try:
        if Na[0][0] in (-1, 2, 3, 4):
            Na_M, restore = modifie_Na_M(M, Na_M, 0, 0)
            rec(M, Na_M, 0, 0, True)
            Na_M = restore_Na_M(Na_M, restore)
        if Na[0][0] in (-1, 0, 1, 2):
            rec(M, Na_M, 0, 0, False)
        print("Aucune solution trouvée.")
        return None
    except SolutionFound:
        print("Solution trouvée !")
        for row in M:
            print(row)
        return M, Na


# ─────────────────────────────────────────────────────────────────────────────
# 9. solve_SL_all  — version optimisée (toutes les solutions)
# ─────────────────────────────────────────────────────────────────────────────

def solve_SL_all(Na):
    n, m = len(Na), len(Na[0])
    solutions = []
    M    = [[False] * m for _ in range(n)]
    Na_M = [[0]     * m for _ in range(n)]

    # rotation si les cases vides sont concentrées au centre
    if score_init(Na) == 0:
        Na = np.rot90(Na, k=2).tolist()   # .tolist() crée déjà une copie

    def rec(i, j, val):
        prev    = M[i][j]
        M[i][j] = val

        # prochaine case en antidiagonale
        if j > 0 and i < n - 1:
            ni, nj = i + 1, j - 1
        elif i == n - 1 and j == m - 1:
            if solution_trouvé(Na, Na_M, n, m):
                solutions.append([row[:] for row in M])
            M[i][j] = prev
            return
        else:
            ni = max(0, i - (m - 1 - j - 1))
            nj = min(i + 1, m - 1)

        E = états_possible(M, Na_M, Na, ni, nj, n, m)

        if True in E:
            Na_M_ref, restore = modifie_Na_M(M, Na_M, ni, nj)
            rec(ni, nj, True)
            restore_Na_M(Na_M, restore)
        if False in E:
            rec(ni, nj, False)

        M[i][j] = prev

    # appels initiaux
    if Na[0][0] >= 2 or Na[0][0] == -1:
        Na_M_ref, restore = modifie_Na_M(M, Na_M, 0, 0)
        rec(0, 0, True)
        restore_Na_M(Na_M, restore)
    if Na[0][0] <= 2 or Na[0][0] == -1:
        rec(0, 0, False)

    if solutions:
        print(f"{len(solutions)} solution(s) trouvée(s).")
    else:
        print("Aucune solution trouvée.")
    return solutions
