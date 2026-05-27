import z3


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

        
def graphique(debut,fin,pas = 5,p = 0,test_par_taille = 1):
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
            
            start = time.time()
            S2 = z3_solver(M)
            end = time.time()
            tot_2 += end-start
        
        L_1.append(tot_1/test_par_taille)
        T.append(i)
        L_2.append(tot_2/test_par_taille)
    
    
    plt.plot(T,L_1,color = 'red')
    plt.plot(T,L_2,color = 'blue')
    plt.ylabel('temps')
    plt.xlabel('largeur du puzzle carré')
    plt.show()
