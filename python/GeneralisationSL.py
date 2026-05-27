import networkx as nx
import numpy as np
import random as r
import matplotlib.pyplot as plt
from matplotlib import animation


# Liste d'adjacence sous forme de dictionnaire
adj_list = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C']
}



def random_planar(N,p = 0.7,c=2,m = 4, nb_sommet_max = 6):

    adj = {}
    bordure = []
    cycle_bordure = []
    
    # initialisation des bordures
    if c == 2:
        adj[0] = {'voisins' : [1,2], 'couleur' : 0, 'bordure' : True}
        adj[1] = {'voisins' : [0,2], 'couleur' : 1, 'bordure' : True}
        adj[2] = {'voisins' : [0,1],
                'couleur' : r.choice([0,1]),
                'bordure' : True}
        
        bordure = [0,1,2]
        c+=1

    else :
        for k in range(c):
            adj[k] = {'voisins' : [(k+1)%c,(k-1)%c], 'couleur' : k, 'bordure' : True}
            bordure.append(k)
            
    cycle_bordure.append(bordure)
    
    #fonction de mutation du graphe
    def inserer(i,j,k,cycle) :
        est_dans_bordure = adj[i]['bordure'] and adj[j]['bordure']
        adj[i]['voisins'].remove(j)
        adj[j]['voisins'].remove(i)
        adj[k] = {'voisins' : [i,j],
                'couleur' : r.choice([adj[i]['couleur'],adj[j]['couleur']]),
                'bordure' : est_dans_bordure }
        adj[i]['voisins'].append(k)
        adj[j]['voisins'].append(k)
        
        # modifier le cycle en gardant le bonne ordre
        
        ind = -1
        trouver = 0
        while ind<len(cycle) and trouver != 2:
            ind += 1
            if cycle[ind] == i or cycle[ind] == j :
                trouver += 1
                
        if ind == len(cycle) - 1 and (cycle[ind-1] !=i or cycle[ind-1] !=i) :
            new_cycle = [k] + cycle
        else :
            new_cycle = cycle[:ind] + [k] + cycle[ind:]
        
        cycle_bordure.remove(cycle)
        cycle_bordure.append(new_cycle)
    
    def lier(cycle,k):
    
        couleur_adj = [] #liste des couleurs possible pour ce noeud
        for s in cycle:
            couleur_adj.append(adj[s]['couleur'])
            adj[s]['voisins'].append(k)
        
        adj[k] = {'voisins' : cycle,
                'couleur' : r.choice(couleur_adj),
                'bordure' : False}
        
        # modification des cycles bordures
        cycle_bordure.remove(cycle)
        
        n_c = len(cycle)
        for i in range(n_c):
            #si les deux noeuds sont dans la bordure alors c'es un cycle de bordure
            if adj[cycle[i]]['bordure'] and adj[cycle[(i+1)%n_c]]['bordure']:
                cycle_bordure.append([cycle[i],cycle[(i+1)%n_c],k])
                
    def supprimer(i,j) :
        if j in adj[i]['voisins'] :
            adj[i]['voisins'].remove(j)
        if i in adj[j]['voisins'] :
            adj[j]['voisins'].remove(i)
        
        

    #création du graphe
    for k in range(c,N):
        #je prend un cycle dans la bordure
        cycle = r.choice(cycle_bordure)
        n_c = len(cycle)
        #si le cycle est trop petit on l'agrandit sur une partie bordure
        if n_c < m :
            #on prend les arrêtes bordure
            arretes = []
            for i in range(n_c):
                if adj[cycle[i]]['bordure'] and adj[cycle[(i+1)%n_c]]['bordure']:
                    arretes.append([cycle[i],cycle[(i+1)%n_c]])
            #on choisi une arrête
            arrete = r.choice(arretes)
            #on insere
            inserer(arrete[0],arrete[1],k,cycle)
        
        #si il est asser grand alors on les connectes par un noeud
        if n_c >= m :
            lier(cycle,k)
    
    # supprime le surplus d'arrête
    for k in range(N) :
        nb_sommet_connecte = len(adj[k]['voisins'])
        if nb_sommet_connecte > nb_sommet_max :
            enlever = r.choices( adj[k]['voisins'] , k = nb_sommet_connecte - nb_sommet_max )
            for sommet in enlever:
                supprimer( k, sommet)
        
        
    
    return adj

def cubic(n,m) :
    adj = {}
    for i in range(n) :
        for j in range(m):
            if i==0 and j==0:
                adj[(i,j)] = {'voisins' : [(i,j+1),(i+1,j)],
                                'couleur' : 1}
            elif i==n-1 and j==0:
                adj[(i,j)] = {'voisins' : [(i-1,j),(i,j+1)],
                                'couleur' : 1}
            elif i==0 and j==m-1:
                adj[(i,j)] = {'voisins' : [(i+1,j),(i,j-1)],
                                'couleur' : 1}                      
            elif i==n-1 and j==m-1:
                adj[(i,j)] = {'voisins' : [(i-1,j),(i,j-1)],
                                'couleur' : 1}            
            elif i==0:
                adj[(i,j)] = {'voisins' : [(i,j+1),(i+1,j),(i,j-1)],
                                'couleur' : 1}         
            elif j==0:
                adj[(i,j)] = {'voisins' : [(i-1,j),(i,j+1),(i+1,j)],
                                'couleur' : 1}       
            elif i==n-1:
                adj[(i,j)] = {'voisins' : [(i-1,j),(i,j+1),(i,j-1)],
                                'couleur' : 1}
            elif j==m-1:
                adj[(i,j)] = {'voisins' : [(i-1,j),(i+1,j),(i,j-1)],
                                'couleur' : 1}
            else:
                adj[(i,j)] =    {'voisins' : [(i-1,j),(i,j+1),(i+1,j),(i,j-1)],
                                'couleur' : 1}
        
    return adj
    
# Transformation liste d'adjacence de graphe colorer en graphe nx

def transformation(graphe_colore):
    # Extraire liste d'adjacence et couleurs
    adj_dict = {noeud: data['voisins'] for noeud, data in graphe_colore.items()}
    couleurs = {noeud: data['couleur'] for noeud, data in graphe_colore.items()}

    # Créer le graphe
    G = nx.Graph(adj_dict)
    
    return G


# Visualisation de graphe

def show_colorer(graphe_colore) :
    # Extraire liste d'adjacence et couleurs
    adj_dict = {noeud: data['voisins'] for noeud, data in graphe_colore.items()}
    couleurs = {noeud: data['couleur'] for noeud, data in graphe_colore.items()}

    # Créer le graphe
    G = nx.Graph(adj_dict)

    # Visualiser avec couleurs
    #pos = nx.spring_layout(G, seed=42)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, 
        node_color=[couleurs[node] for node in G.nodes()],
        with_labels=False,
        node_size=200)
    plt.show()

def show_colorer_bordure(graphe_colore, layout='kamada_kawai'):
    # Extraire données
    adj_dict = {noeud: data['voisins'] for noeud, data in graphe_colore.items()}
    couleurs = {noeud: data['couleur'] for noeud, data in graphe_colore.items()}
    bordures = {noeud: data.get('bordure', False) for noeud, data in graphe_colore.items()}
    
    # Créer graphe
    G = nx.Graph(adj_dict)
    
    # Layout
    if layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Préparer bordures
    edgecolors = ['red' if bordures[node] else 'black' for node in G.nodes()]
    linewidths = [3 if bordures[node] else 1 for node in G.nodes()]
    
    # Dessiner
    plt.figure(figsize=(10, 8))
    
    # Arêtes
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    
    # Noeuds avec couleurs et bordures
    nx.draw_networkx_nodes(G, pos,
                          node_color=[couleurs[node] for node in G.nodes()],
                          node_size=300,
                          edgecolors=edgecolors,
                          linewidths=linewidths,
                          alpha=0.8)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def show_3D(graphe_colore) :
    # Extraire liste d'adjacence et couleurs
    adj_dict = {noeud: data['voisins'] for noeud, data in graphe_colore.items()}
    couleurs = {noeud: data['couleur'] for noeud, data in graphe_colore.items()}
    bordures = {noeud: data.get('bordure', False) for noeud, data in graphe_colore.items()}

    # Créer le graphe
    G = nx.Graph(adj_dict)

    # Visualiser avec couleurs
    #pos = nx.spring_layout(G, seed=42)
    pos = nx.spring_layout(G, dim=3)
    nodes = np.array([pos[v] for v in G])
    edges = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.clear()
    ax.scatter(*nodes.T, alpha=0.2, s=100, color='blue')
    for vizedge in edges:
        ax.plot(*vizedge.T, color="gray")
    ax.grid(False)
    ax.set_axis_off()
    plt.show()

    
def show_3D_colorer_bordure(graphe_colore):
    # Extraire liste d'adjacence et couleurs
    adj_dict = {noeud: data['voisins'] for noeud, data in graphe_colore.items()}
    couleurs = {noeud: data['couleur'] for noeud, data in graphe_colore.items()}
    bordures = {noeud: data.get('bordure', False) for noeud, data in graphe_colore.items()}

    # Créer le graphe
    G = nx.Graph(adj_dict)

    # Obtenir les positions 3D
    pos = nx.spring_layout(G, dim=3)
    nodes = np.array([pos[v] for v in G])

    # Extraire les coordonnées
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]

    # Créer la liste des couleurs des nœuds (dans l'ordre de G.nodes())
    node_color_values = [couleurs[node] for node in G.nodes()]

    # Créer la liste des couleurs de bordure et des largeurs
    edge_colors = []
    linewidths = []
    for node in G.nodes():
        if bordures[node]:
            edge_colors.append('red')
            linewidths.append(3)
        else:
            edge_colors.append('none')
            linewidths.append(0)

    # Créer la figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Tracer les nœuds
    scatter = ax.scatter(x, y, z, 
                         c=node_color_values, 
                         edgecolors=edge_colors, 
                         linewidths=linewidths, 
                         s=100, 
                         alpha=0.9)

    # Tracer les arêtes
    edges = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    for vizedge in edges:
        ax.plot(*vizedge.T, color="gray", alpha=0.5)

    ax.grid(False)
    ax.set_axis_off()
    plt.show()
