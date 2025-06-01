import matplotlib.pyplot as plt

'''n, m = 5, 5

G = [
    [None, None, 2, 1, None],
    [2, 2, 1, 1, 3],
    [None, 2, 2, 0, 3],
    [None, None, 2, None, None],
    [2, None, None, None, 3]
]

# Initialiser les matrices d'arêtes
L = [[0]*(m+1) for _ in range(n+1+1)]
C = [[0]*(m+1+1) for _ in range(n+1)]

# Exemple : on active quelques arêtes
L[0][2] = 1
L[0][3] = 1
C[0][4] = 1
C[1][4] = 1
L[2][3] = 1
C[3][0] = 1
L[5][0] = 1

# Pour mettre une croix : -1
L[2][2] = -1
C[2][3] = -1'''

def draw_grid(M):

    M= G,L,C
    
    n,m = len(G[0]),len(G)
    fig, ax = plt.subplots(figsize=(n,m))

    # Tracer les points
    for i in range(n+1):
        for j in range(m+1):
            ax.plot(j, n-i, 'ko', markersize=8)

    # Ajouter les indices
    for i in range(n):
        for j in range(m):
            if G[i][j] is not None:
                ax.text(j+0.5, n-i-0.5, str(G[i][j]),
                        fontsize=16, ha='center', va='center')

    # Tracer les arêtes horizontales
    for i in range(n+1):
        for j in range(m):
            if L[i][j] == 1:
                ax.plot([j, j+1], [n-i, n-i], 'k', linewidth=4)
            elif L[i][j] == -1:
                ax.text(j+0.5, n-i, '×', color='red', fontsize=18, ha='center', va='center')

    # Tracer les arêtes verticales
    for i in range(n):
        for j in range(m+1):
            if C[i][j] == 1:
                ax.plot([j, j], [n-i, n-(i+1)], 'k', linewidth=4)
            elif C[i][j] == -1:
                ax.text(j, n-i-0.5, '×', color='red', fontsize=18, ha='center', va='center')

    ax.set_xlim(-0.5, m+1-0.5)
    ax.set_ylim(-0.5, n+1-0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

