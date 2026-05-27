import random
import networkx as nx
import matplotlib.pyplot as plt


# =========================
# Génération graphe planaire
# =========================

def generate_planar_graph(n):

    G = nx.Graph()

    # triangle initial
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 0)
    ])

    for i in range(3, n):

        added = False
        attempts = 0

        while not added and attempts < 20:

            nodes = list(G.nodes())

            # choisir 2 ou 3 voisins
            k = random.choice([2, 3])
            neighbors = random.sample(nodes, k)

            # vérifier degré max 6
            if any(G.degree(v) >= 6 for v in neighbors):
                attempts += 1
                continue

            G.add_node(i)

            for v in neighbors:
                G.add_edge(i, v)

            # vérifier planarité
            is_planar, _ = nx.check_planarity(G)

            if is_planar:
                added = True
            else:
                G.remove_node(i)

            attempts += 1

    return G


# =========================
# Extraction des faces
# =========================

def get_faces(embedding):

    visited = set()
    faces = []

    for u in embedding:

        for v in embedding[u]:

            if (u, v) in visited:
                continue

            face = list(embedding.traverse_face(u, v))

            for i in range(len(face)):
                a = face[i]
                b = face[(i+1) % len(face)]
                visited.add((a, b))

            faces.append(face)

    return faces


# =========================
# Construction dual
# =========================

def build_dual(G):

    is_planar, embedding = nx.check_planarity(G)

    if not is_planar:
        raise ValueError("Graph non planaire")

    faces = get_faces(embedding)

    dual = nx.Graph()

    for i in range(len(faces)):
        dual.add_node(i)

    edge_map = {}

    for i, face in enumerate(faces):

        for j in range(len(face)):

            u = face[j]
            v = face[(j+1) % len(face)]

            edge = tuple(sorted((u, v)))

            if edge not in edge_map:
                edge_map[edge] = [i]
            else:
                edge_map[edge].append(i)

    for edge in edge_map:

        if len(edge_map[edge]) == 2:
            f1, f2 = edge_map[edge]
            dual.add_edge(f1, f2)

    return dual, faces


# =========================
# Visualisation
# =========================

def visualize(G, dual, faces):

    pos = nx.spring_layout(G)

    dual_pos = {}

    for i, face in enumerate(faces):
        x = sum(pos[v][0] for v in face) / len(face)
        y = sum(pos[v][1] for v in face) / len(face)
        dual_pos[i] = (x, y)

    plt.figure(figsize=(8,8))

    # graphe principal
    nx.draw(
        G,
        pos,
        node_color="lightblue",
        with_labels=True,
        node_size=600
    )

    # graphe dual
    nx.draw(
        dual,
        dual_pos,
        node_color="red",
        edge_color="red",
        node_size=300
    )

    plt.title("Graphe planaire et son dual")
    plt.show()

def visualize_both(G, dual, faces):

    plt.figure(figsize=(8,8))

    # position graphe planaire
    pos = nx.planar_layout(G)

    # position dual (centre des faces)
    dual_pos = {}

    for i, face in enumerate(faces):
        x = sum(pos[v][0] for v in face) / len(face)
        y = sum(pos[v][1] for v in face) / len(face)
        dual_pos[i] = (x, y)

    # dessiner graphe principal
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=700,
        font_size=11,
        font_weight="bold"
    )

    # labels du dual
    dual_labels = {i: f"F{i}" for i in dual.nodes()}

    # dessiner dual
    nx.draw(
        dual,
        dual_pos,
        labels=dual_labels,
        node_color="lightcoral",
        node_size=500,
        edge_color="red",
        font_size=10
    )

    plt.title("Graphe planaire (bleu) et graphe dual (rouge)")
    plt.show()

# =========================
# Programme principal
# =========================

if __name__ == "__main__":

    G = generate_planar_graph(15)

    dual, faces = build_dual(G)

    visualize(G, dual, faces)