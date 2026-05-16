TIPE – Slither Link

Ce dépôt regroupe les outils développés dans le cadre de mon TIPE (2025‑2026) sur le Slither Link (ou Slitherlink).
Le projet contient :

    un visualiseur interactif (Pygame) permettant de créer, modifier et jouer à des grilles de Slither Link ;

    une bibliothèque Python complète pour générer, vérifier, résoudre et analyser des puzzles Slither Link.

📁 Contenu du dépôt
text

TIPE-slither-link/
├── pygame_visualiser.py      # Interface interactive (éditeur + mode jeu)
├── SlitherLink.py            # Module complet : génération, vérification, résolution, affichage
├── images/                   # (dossier) captures d’écran, illustrations
└── sources/                  # (dossier) articles PDF utilisés pour la recherche

🔧 Dépendances

Les deux scripts nécessitent les bibliothèques suivantes (Python 3.8+) :
bash

pip install pygame matplotlib numpy z3-solver

    pygame – pour l’interface interactive

    matplotlib / numpy – pour l’affichage et les calculs matriciels

    z3-solver – pour la résolution par contraintes (optionnel, mais recommandé)

🎮 Utilisation du visualiseur Pygame

Lance l’éditeur interactif :
bash

python pygame_visualiser.py

Fonctionnalités
Mode	Touche / action	Effet
Édition (par défaut)	← ↑ ↓ →	déplacer la case sélectionnée
	0…9	entrer un chiffre (0‑4) dans la case
	5 ou ?	mettre un point d’interrogation (-1)
	C	changer la couleur de fond de la case
	E	passer en mode jeu
	G	afficher/masquer le quadrillage (points vs lignes)
	champ Lignes / Colonnes + bouton Appliquer	redimensionner la grille
Jeu	clic gauche	tracer une arête noire
	clic droit	tracer une croix rouge (arête interdite)
	clic milieu	effacer l’arête
	E	revenir en mode édition

    ⚠️ La validation des règles et la recherche de solution ne sont pas encore intégrées au visualiseur – elles sont disponibles dans le module SlitherLink.py.

📚 Bibliothèque SlitherLink.py

Ce module contient toutes les fonctions scientifiques du projet.
Principales fonctionnalités

    Génération de puzzles
    grignotage_rec(), grignotage_carré(), generate_chemins(), generate_maze(), etc.

    Vérification de validité
    Verif(M) – test si une matrice booléenne forme une boucle unique (modèle 2 couleurs).
    Nombre_Arrete(M) – calcule la matrice des indices à partir d’une solution.

    Résolution

        solve_SL_all(Na) : recherche toutes les solutions par backtracking intelligent.

        z3_solver(Na) : résolution par solveur SAT/SMT (Z3).

    Analyse & dénombrement
    Filtres (diagonales, symétries, nombre de cases modifiables…), étude des petites grilles 3×3, etc.

    Affichage
    show(), show_n(), show_anim() pour visualiser des matrices ou des séquences.

Exemple d’utilisation
python

from SlitherLink import generer, solve_SL_all, show

# Générer un puzzle aléatoire 6x6 avec 40% de cases masquées (-1)
Na = generer(6, 6, p=0.4)

# Résoudre
solutions = solve_SL_all(Na)
print(f"{len(solutions)} solution(s) trouvée(s)")

# Afficher la première solution
if solutions:
    show(solutions[0])

🖼️ Captures d’écran

(Tu peux ajouter ici quelques images issues du dossier images/)
👤 Auteur

Nohlan SAUCET – Projet TIPE 2025‑2026
Lycée / CPGE (à compléter)
📖 Références

Les articles scientifiques consultés sont disponibles dans le dossier sources/.
Quelques liens utiles :

    Règles du Slither Link (Nikoli)

    Article sur la complexité du Slither Link (Wikipedia)

🧪 État du projet

    Visualiseur interactif (Pygame)

    Génération de grilles (multi‑méthodes)

    Backtracking complet (trouve toutes les solutions)

    Solveur Z3

    Interface de résolution intégrée au visualiseur

    Benchmark des méthodes de génération

Ce dépôt est en cours d’évolution dans le cadre d’un projet personnel et scolaire.
