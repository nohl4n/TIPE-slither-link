import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from Generate_SL import generate_maze

def visualize_slitherlink(M, title="Slitherlink Puzzle"):
    """
    Visualise un puzzle Slitherlink à partir d'une matrice booléenne.
    
    Args:
        M: Matrice numpy booléenne (True = intérieur, False = extérieur)
        title: Titre du graphique
    """
    n, m = M.shape
    
    # Créer la figure avec un ratio approprié
    fig, ax = plt.subplots(figsize=(m*0.8, n*0.8))
    
    # Configuration de base
    ax.set_xlim(-1, m)
    ax.set_ylim(-1, n)
    ax.invert_yaxis()  # Pour avoir (0,0) en haut à gauche
    
    # Dessiner la grille de points
    for i in range(n+1):
        for j in range(m+1):
            ax.plot(j-0.5, i-0.5, 'ko', markersize=4)
    
    # Calculer et afficher les chiffres pour chaque cellule
    for i in range(n):
        for j in range(m):
            # Calculer le nombre d'arêtes requises pour cette cellule
            nb_edges = calculate_edges_needed(M, i, j)
            
            # Afficher le chiffre au centre de la cellule
            if nb_edges is not None:
                ax.text(j, i, str(nb_edges), 
                       ha='center', va='center', 
                       fontsize=14)
    
    # Dessiner les lignes de la solution (optionnel - pour debug)
    draw_solution_outline(ax, M)
    
    # Configuration finale
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16)
    
    # Supprimer les bordures
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()

def calculate_edges_needed(M, i, j):
    """
    Calcule le nombre d'arêtes nécessaires autour d'une cellule.
    Retourne None si la cellule ne doit pas avoir de chiffre.
    """
    n, m = M.shape
    
    # Obtenir les voisins (haut, droite, bas, gauche)
    neighbors = []
    
    # Haut
    if i > 0:
        neighbors.append(M[i-1, j])
    else:
        neighbors.append(False)  # Extérieur du puzzle
    
    # Droite  
    if j < m-1:
        neighbors.append(M[i, j+1])
    else:
        neighbors.append(False)
    
    # Bas
    if i < n-1:
        neighbors.append(M[i+1, j])
    else:
        neighbors.append(False)
    
    # Gauche
    if j > 0:
        neighbors.append(M[i, j-1])
    else:
        neighbors.append(False)
    
    # Compter les transitions entre intérieur et extérieur
    current_cell = M[i, j]
    edge_count = 0
    
    for neighbor in neighbors:
        if current_cell != neighbor:  # Transition = arête nécessaire
            edge_count += 1
    
    return edge_count

def draw_solution_outline(ax, M, show_solution=False):
    """
    Dessine le contour de la solution (optionnel, pour visualiser la boucle).
    """
    if not show_solution:
        return
        
    n, m = M.shape
    
    # Dessiner les arêtes horizontales
    for i in range(n+1):
        for j in range(m):
            # Vérifier s'il y a une transition verticale
            top_cell = M[i-1, j] if i > 0 else False
            bottom_cell = M[i, j] if i < n else False
            
            if top_cell != bottom_cell:
                ax.plot([j-0.5, j+0.5], [i-0.5, i-0.5], 'r-', linewidth=3)
    
    # Dessiner les arêtes verticales
    for i in range(n):
        for j in range(m+1):
            # Vérifier s'il y a une transition horizontale
            left_cell = M[i, j-1] if j > 0 else False
            right_cell = M[i, j] if j < m else False
            
            if left_cell != right_cell:
                ax.plot([j-0.5, j-0.5], [i-0.5, i+0.5], 'r-', linewidth=3)

def visualize_slitherlink_with_solution(M, title="Slitherlink avec Solution"):
    """
    Version qui montre aussi la solution en rouge.
    """
    n, m = M.shape
    
    fig, ax = plt.subplots(figsize=(m*0.8, n*0.8))
    
    ax.set_xlim(-1, m)
    ax.set_ylim(-1, n)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Grille de points
    for i in range(n+1):
        for j in range(m+1):
            ax.plot(j-0.5, i-0.5, 'ko', markersize=4)
    
    # Chiffres
    for i in range(n):
        for j in range(m):
            nb_edges = calculate_edges_needed(M, i, j)
            if nb_edges is not None:
                ax.text(j, i, str(nb_edges), 
                       ha='center', va='center', 
                       fontsize=14)
    
    # Solution en rouge
    draw_solution_outline(ax, M, show_solution=True)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Fonction de test avec vos méthodes
def test_visualizer():
    """
    Teste le visualiseur avec vos fonctions de génération.
    """
    # Test avec une petite matrice simple
    print("Test avec matrice simple...")
    M_simple = np.array([
        [True, True, False],
        [True, False, False],
        [False, False, False]
    ])
    visualize_slitherlink(M_simple, "Test Simple")
    visualize_slitherlink_with_solution(M_simple, "Test Simple avec Solution")
    
    # Test avec vos fonctions (si disponibles)
    try:
        print("Test avec grignotage_rec...")
        M_rec = generate_maze(11, 11)
        visualize_slitherlink(M_rec, "Générateur Récursif")
        visualize_slitherlink_with_solution(M_rec, "Générateur Récursif avec Solution")
    except NameError:
        print("Fonctions de génération non disponibles dans ce contexte")

if __name__ == "__main__":
    test_visualizer()
