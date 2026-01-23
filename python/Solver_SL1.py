class SolutionFound(Exception):
    pass

def solve3(S):
    n, m = len(S), len(S[0])
    M = [[0]*m for _ in range(n)]
    try:
        # on tente d'abord en posant la première case à 1 puis 2
        rec_brut3(M, S, 0, 0, 1)
        rec_brut3(M, S, 0, 0, 2)
        print("Aucune solution trouvée.")
    except SolutionFound:
        print(" Solution trouvée !")

def rec_brut3(M, S, i, j, Val):
    # sauvegarde pour le backtracking
    prev = M[i][j]
    M[i][j] = Val

    # affichage pour debug (on affiche une copie pour éviter effet de référence)
    for row in M:
        print(row)
    print("--")

    # test de solution
    if M == S:
        raise SolutionFound

    n = len(M)
    m = len(M[0])

    # avancer dans l'ordre des cases : (i,j) -> next_i, next_j
    if j < m - 1:
        # on explore la case suivante sur la même ligne (i, j+1)
        rec_brut3(M, S, i, j+1, 1)
        rec_brut3(M, S, i, j+1, 2)
    elif i < n - 1:
        # fin de colonne : on passe à la ligne suivante, première colonne
        rec_brut3(M, S, i+1, 0, 1)
        rec_brut3(M, S, i+1, 0, 2)
    # sinon nous sommes à la dernière case et on a déjà testé la solution

    # backtrack : restaurer l'ancienne valeur avant de retourner
    M[i][j] = prev
    

def solve_SL(Na):
    n, m = len(Na), len(Na[0])
    M = [[False]*m for _ in range(n)]
    Na_M = [[0]*m for _ in range(n)]
    try:
        # on tente d'abord en posant la première case à 1 puis 2
        if Na[0][0] >= 2:
            Na_M,restore = modifie_Na_M(M,Na_M,0,0)
            
            rec_brut_SL(M, Na_M, Na, 0, 0, True)
            
            Na_M = restore_Na_M(Na_M,restore)
        if Na[0][0] <= 2:
            rec_brut_SL(M, Na_M, Na, 0, 0, False)
        print("Aucune solution trouvée.")
    except SolutionFound:
        print(" Solution trouvée !")

def rec_brut_SL(M, Na_M, Na, i, j, Val):
    # sauvegarde pour le backtracking
    prev = M[i][j]
    M[i][j] = Val
    
    n = len(M)
    m = len(M[0])

    # affichage pour debug (on affiche une copie pour éviter effet de référence)
    
    for row in range(n):
        print(M[row],Na_M[row],Na[row],i,j)
    print("--")

    # test de solution
    if Na_M == Na:
        raise SolutionFound

    # avancer dans l'ordre des cases : (i,j) -> next_i, next_j
    # calcule next indices
    next_i, next_j = i, j
    if j < m - 1:
        next_i, next_j = i, j + 1
    elif i < n - 1:
        next_i, next_j = i + 1, 0
    else:
        # dernière case ; on a déjà testé l'égalité, donc pas de suite
        # restore et return
        M[i][j] = prev
        return

    # obtenir états possibles pour la prochaine case
    E = états_possible(M, Na_M, Na, next_i, next_j)

    # explore selon ce qui est possible (tester présence plutôt qu'égalité stricte)
    if True in E:
        #modifie Na
        Na_M,restore = modifie_Na_M(M,Na_M,next_i,next_j)
        
        rec_brut_SL(M, Na_M, Na, next_i, next_j, True)
        
        # restore l'ancienne valeur de Na_M
        Na_M = restore_Na_M(Na_M,restore)
    if False in E:
        rec_brut_SL(M, Na_M, Na, next_i, next_j, False)

    # backtrack : restaurer l'ancienne valeur avant de retourner
    M[i][j] = prev


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
            
            

    
def états_possible(M,Na_M,Na,i,j):
    E = []
    if i == 0 :
        if j == 0 or condition_droite(M,Na_M,Na,i,j) == "pas de condition":
            if condition_mettre_in(M,Na_M,Na,i,j):
                E.append(True)
            if condition_mettre_out(M,Na_M,Na,i,j):
                E.append(False)
        elif condition_droite(M,Na_M,Na,i,j) == True:
            if condition_mettre_in(M,Na_M,Na,i,j):
                E.append(True)
        else : 
            if condition_mettre_out(M,Na_M,Na,i,j):
                E.append(False)
                
    else :
        if condition_bas(M,Na_M,Na,i,j) == True :
            if j == 0 or condition_droite(M,Na_M,Na,i,j) == "pas de condition":
                if condition_mettre_in(M,Na_M,Na,i,j):
                    E.append(True)
            elif condition_droite(M,Na_M,Na,i,j) == True:
                if condition_mettre_in(M,Na_M,Na,i,j):
                    E.append(True)
        
        else : 
            if j == 0 or condition_droite(M,Na_M,Na,i,j) == "pas de condition":
                if condition_mettre_out(M,Na_M,Na,i,j):
                    E.append(False)
            elif condition_droite(M,Na_M,Na,i,j) == False:
                if condition_mettre_out(M,Na_M,Na,i,j):
                    E.append(False)
                    
    return E
        
        
def condition_bas(M,Na_M,Na,i,j):
    # l'état en dessous ne pouvant plus être modifier après
    if Na_M[i-1][j] == Na[i-1][j]:
        return False
    else: # il faut le modifier
        return True

def condition_droite(M,Na_M,Na,i,j):
    # état de droit ne pourras être modifier plus que 1 fois après
    if M[i][j-1] and Na_M[i][j-1] + 2 == Na[i][j-1] : 
        # si l'etat est in et doit être modifier 2 fois
        return True
    elif (not M[i][j-1]) and Na_M[i][j-1] - 2 == Na[i][j-1] :
        # si l'etat est out et doit être modifier 2 fois
        return True
    elif Na_M[i][j-1] == Na[i][j-1] :
        # si l'etat final déjà atteint il ne faut plus le modifier
        return False
    else : # les deux états sont possibles
        return "pas de condition"

def condition_mettre_out(M,Na_M,Na,i,j):
    # vérifier qu'il n'y a pas de contradiction si on laisse out
    n = len(M)
    m = len(M[0])

    val_si_out = Na_M[i][j]
    
    if i+1 == n and j+1 == m :
        # On ne plus apporté de modification
        return val_si_out == Na[i][j]
        
    elif j+1 == m :
        return Na[i][j] - val_si_out >= 0 and Na[i][j] - val_si_out <= 1
    
    else :
        return Na[i][j] - val_si_out >= 0 and Na[i][j] - val_si_out <= 2

def condition_mettre_in(M,Na_M,Na,i,j):
    # vérifier qu'il n'y a pas de contradiction si on met in
    n = len(M)
    m = len(M[0])

    C = Croix(M,i,j)
    nb= 0
    for k in range(4):
        if not C[k]:
            nb+=1
    
    val_si_in = nb
    
    if i+1 == n and j+1 == m :
        # On ne plus apporté de modification
        return val_si_in == Na[i][j]
        
    elif j+1 == m :
        return val_si_in - Na[i][j] >= 0 and val_si_in - Na[i][j] <= 1
    
    else :
        return val_si_in - Na[i][j] >= 0 and val_si_in - Na[i][j] <= 2

