import random as r
import copy as c


def set_to_list (s : set )-> list :

    l = []
    for (a,b) in s :
        if not a in l:
            l.append(a)
        if not b in l:
            l.append(b)

    return l


def random_planar(N,m = 4, nb_sommet_max = 6):
        
    # initialisation du graphe
    
    G={} 
    G[0] = [1,2]
    G[1] = [0,2]
    G[2] = [0,1]

    # initialisation des bordures et cycle

    bordure = {(0,1),(1,2),(0,2)}
    cycle_bordure = [bordure]
    cycle_all = [bordure]
    bordure = [(0,1),(1,2),(0,2)]


    # initialisation dual_graphe

    G_dual = { tuple(bordure) : () , () : tuple(bordure) }

    #fonction de mutation du graphe

    def inserer(i,j,k,cycle : set) :

        if j<i :
            i,j = j,i
        
        G[i].remove(j)
        G[j].remove(i)
        G[k] = [i,j]
        G[i].append(k)
        G[j].append(k)
        
        # modifier le cycle
        
        cycle.remove((i,j))
        cycle.add((i,k))
        cycle.add((j,k))
        bordure.remove((i,j))
        bordure.append((i,k))
        bordure.append((j,k))
        print('bordure inserer',bordure)
        print(cycle)

    
    def lier(cycle,k):

        print('bordure lier 1',bordure )
        print(cycle)
        
        #tous les sommets du cycle
        l = []
        for (a,b) in cycle :
            if not a in l:
                l.append(a)
            if not b in l:
                l.append(b)

        #on choisit les sommet que l'on veut lier a k pour former les sous cycle
        sommet_choisi_p = r.choices(l, k = r.randint(2,max (len(l), nb_sommet_max)))

        sommet_choisi = []
        for a in sommet_choisi_p :
            if not a in sommet_choisi:
                sommet_choisi.append(a)
        
        G[k] = sommet_choisi

        for s in sommet_choisi:
            G[s].append(k)
        
        # modification des cycles bordures
        cycle_bordure.remove(cycle)
        cycle_all.remove(cycle)

        cycle_c = c.deepcopy(cycle)
        
        # tant que le set n'est pas vide on prend un élément on l'étand 
        # a gauche et a droite tant que g et d ne sont pas des sommet choisi
        while cycle_c != set():
            
            (a,b) = cycle_c.pop()
            sous_cycle = {(a,b)}
            g,d = a,b
            # étendre à gauche
            while not g in sommet_choisi:

                for (s1,s2) in cycle_c:
                    if g == s1:
                        sous_cycle.add((s1,s2))
                        cycle_c.remove((s1,s2))
                        g = s2
                        break
                    elif g == s2:
                        sous_cycle.add((s1,s2))
                        cycle_c.remove((s1,s2))
                        g = s1
                        break

            # étendre à droite
            while not d in sommet_choisi:

                for (s1,s2) in cycle_c:
                    if d == s1:
                        sous_cycle.add((s1,s2))
                        cycle_c.remove((s1,s2))
                        d = s2
                        break
                    elif d == s2:
                        sous_cycle.add((s1,s2))
                        cycle_c.remove((s1,s2))
                        d = s1
                        break
            
            # a présent pour fermé le cycle on rajoute (d,k) et (g,k)
            sous_cycle.add((g,k))
            sous_cycle.add((d,k))
                        
            # on ajoute le sous cycle trouvé au cycle bordure et cycle_all
            if sous_cycle - set(bordure) != set() :
                cycle_bordure.append(sous_cycle)
            cycle_all.append(sous_cycle)

            print('bordure lier 2',bordure )
            print(cycle)

    #création du graphe
    for k in range(3,N):
        #je prend un cycle dans la bordure
        cycle = r.choice(cycle_bordure)
        n_c = len(cycle)
        print('bordure lier 3',bordure )
        print(cycle)
        #si le cycle est trop petit on l'agrandit sur une partie bordure
        if n_c < m :
            #on prend une arrête bordure 
            (i,j) = r.choice( list (cycle & set (bordure)))
            #on insere
            inserer(i,j,k,cycle)
        
        #si il est asser grand alors on les connectes par un noeud
        elif n_c >= m :
            lier(cycle,k)       
    
    return G
