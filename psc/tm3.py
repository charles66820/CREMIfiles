#!/usr/bin/python3.6

import random

# Anatole et Philémon

def SimulePileOuFace():
    return random.randint(0,1)

# 1.1
def SimuleJeu():
    nbTirs = 0
    while (True):
        if nbTirs % 2 == 0:
            if  SimulePileOuFace():
                return 1 # Anatole
        else:
            if not SimulePileOuFace():
                return 0 # Philémon
        nbTirs += 1

# 1.2
a = 0
p = 0
nbPartys = 2000
for i in range(nbPartys):
    if SimuleJeu():
        a += 1
    else:
        p += 1

print("nbWin Anatole :", a, "; nbWin Philémon :", p, "; Anatole proba :", a*1/nbPartys, "; Philémon proba :", p*1/nbPartys)

# Om vois que Anatole a 66.66% (donc 0.66) de chance de gagner et Philémon a 33.33% (donc 0.33) de chance de gagner


# La ruine du joueur
# 2.1
def SimuleCasino(p,k,n): # k : euros en poche ; n : fortune max
    while True:
        if random.choices([0, 1], [1-p, p])[0]:
            k += 1
        else:
            k -= 1

        if k == n:
            return True
        elif k == 0:
            return False

lose = 0
win = 0
nbSoirees = 100
for i in range(nbSoirees):
    if SimuleCasino(1/2, 5, 10):
        win += 1
    else:
        lose += 1

print("nbSoirees win :", win, "; nbSoirees lose :", lose)

# je vois pas quel formule utiliser formule

# Jeux de ballon
# 3.1
def SimuleCoup(x):
    return random.choices([0, 1], [1-x, x])[0]

def SimuleSetTableSimplifie(x):
    score_joueur = 0
    score_adversaire = 0
    while score_joueur < 11 and score_adversaire < 11:
        coup = SimuleCoup(x)
        if coup == 1:
            score_joueur = score_joueur +1
        else:
            score_adversaire = score_adversaire +1
    if score_joueur > score_adversaire:
        return 1
    else:
        return 0

def SimuleSetTable(x):
    scoreJoueur = 0
    scoreAdversaire = 0
    while True:
        if scoreJoueur > 11 and scoreJoueur - 2 >= scoreAdversaire:
            return 1
        elif scoreAdversaire > 11 and scoreAdversaire - 2 >= scoreJoueur:
            return 0
        if SimuleCoup(x):
            scoreJoueur += 1
        else:
            scoreAdversaire += 1

j = 0
a = 0
nbSets = 2000
for i in range(nbSets):
    if SimuleSetTable(0.51):
        j += 1
    else:
        a += 1

print("Table : nb joueur win sets :", j, "; nb adversaire win sets :", a, "; joueur win sets % :", j*100/nbSets, "; adversaire win sets % :", a*100/nbSets)

# Pour 0.7 la probabilite estimer remporter un set est de 98% (49/50 donc 0.98)
# Pour 0.6 la probabilite estimer remporter un set est de 85% (17/20 donc 0.85)
# Pour 0.55 la probabilite estimer remporter un set est de 69% (69/100 donc 0.69)
# Pour 0.51 la probabilite estimer remporter un set est de 54% (27/50 donc 0.54)
#
# Pour 0.5 la probabilite est de 1/2 donc 0.5
#
# La regle des deux points d’ecart sur la probabilite de remporter un set a pour effet
# d'eviter que le joueur qui commance est plus de chance de remporter le set

def SimuleMatchTable(x):
    setsWinJoueur = 0
    setsWinAdversaire = 0
    nbSetsWin = 3
    while True:
        if SimuleSetTable(x):
            setsWinJoueur += 1
        else:
            setsWinAdversaire += 1
        if setsWinJoueur == nbSetsWin:
            return 1
        elif setsWinAdversaire == nbSetsWin:
            return 0

j = 0
a = 0
nbMatchTable = 2000
for i in range(nbMatchTable):
    if SimuleMatchTable(0.5):
        j += 1
    else:
        a += 1

print("Table : nb joueur win matchs :", j, "; nb adversaire win matchs :", a, "; joueur win matchs % :", j*100/nbMatchTable, "; adversaire win matchs % :", a*100/nbMatchTable)

# Quant on fait des millier de matchs avec x qui auguemment (de 0.5 a 1) on remarque que la probabiliter de gagner auguemmente et a partir de x = 0.6 la probabiliter passe a 1

def SimuleJeuTennis(x):
    scoreJoueur = 0
    scoreAdversaire = 0
    while True:
        if scoreJoueur > 4 and scoreJoueur - 2 >= scoreAdversaire:
            return 1
        elif scoreAdversaire > 4 and scoreAdversaire - 2 >= scoreJoueur:
            return 0
        if SimuleCoup(x):
            scoreJoueur += 1
        else:
            scoreAdversaire += 1

def SimuleSetTennis(x):
    jeuWinJoueur = 0
    jeuWinAdversaire = 0
    while True:
        if jeuWinJoueur > 6 and jeuWinJoueur - 2 >= jeuWinAdversaire:
            return 1
        elif jeuWinAdversaire > 6 and jeuWinAdversaire - 2 >= jeuWinJoueur:
            return 0
        if SimuleJeuTennis(x):
            jeuWinJoueur += 1
        else:
            jeuWinAdversaire += 1

def SimuleMatchTennis(x):
    setsWinJoueur = 0
    setsWinAdversaire = 0
    nbSetsWin = 3
    while True:
        if SimuleSetTennis(x):
            setsWinJoueur += 1
        else:
            setsWinAdversaire += 1
        if setsWinJoueur == nbSetsWin:
            return 1
        elif setsWinAdversaire == nbSetsWin:
            return 0

j = 0
a = 0
nbMatchTennis = 2000
for i in range(nbMatchTennis):
    if SimuleMatchTennis(0.5):
        j += 1
    else:
        a += 1

print("Tennis : nb joueur win matchs :", j, "; nb adversaire win matchs :", a, "; joueur win matchs % :", j*100/nbMatchTennis, "; adversaire win matchs % :", a*100/nbMatchTennis)

# La probabilite, determiner experimentalement, pour qu'un joueur est 9 chances sur 10 de remporter un match en 3 sets gagnants est de 0.532

# Bis
def SimuleSetTableBis(p, q):
    scoreJoueur = 0
    scoreAdversaire = 0
    while True:
        if scoreJoueur > 11 and scoreJoueur - 2 >= scoreAdversaire:
            return 1
        elif scoreAdversaire > 11 and scoreAdversaire - 2 >= scoreJoueur:
            return 0

        if scoreJoueur >= 10 and scoreAdversaire >= 10:
            coup = SimuleCoup(p if scoreJoueur + scoreAdversaire % 2 == 0 else q)
        else:
            test = (scoreJoueur + scoreAdversaire) % 4
            coup = SimuleCoup(p if test < 2 else q)

        if coup:
            scoreJoueur += 1
        else:
            scoreAdversaire += 1

def SimuleMatchTableBis(p, q):
    setsWinJoueur = 0
    setsWinAdversaire = 0
    nbSetsWin = 3
    while True:
        if SimuleSetTableBis(p, q):
            setsWinJoueur += 1
        else:
            setsWinAdversaire += 1
        if setsWinJoueur == nbSetsWin:
            return 1
        elif setsWinAdversaire == nbSetsWin:
            return 0

j = 0
a = 0
nbMatchTableBis = 2000
for i in range(nbMatchTableBis):
    if SimuleMatchTableBis(0.7, 0.5):
        j += 1
    else:
        a += 1

print("TableBis : nb joueur win matchs :", j, "; nb adversaire win matchs :", a, "; joueur win matchs % :", j*100/nbMatchTableBis, "; adversaire win matchs % :", a*100/nbMatchTableBis)


def SimuleSetTennisBis(p, q):
    jeuWinJoueur = 0
    jeuWinAdversaire = 0
    while True:
        if jeuWinJoueur > 6 and jeuWinJoueur - 2 >= jeuWinAdversaire:
            return 1
        elif jeuWinAdversaire > 6 and jeuWinAdversaire - 2 >= jeuWinJoueur:
            return 0

        if SimuleJeuTennis(p if jeuWinJoueur + jeuWinAdversaire % 2 == 0 else q):
            jeuWinJoueur += 1
        else:
            jeuWinAdversaire += 1

def SimuleMatchTennisBis(p, q):
    setsWinJoueur = 0
    setsWinAdversaire = 0
    nbSetsWin = 3
    while True:
        if SimuleSetTennisBis(p, q):
            setsWinJoueur += 1
        else:
            setsWinAdversaire += 1
        if setsWinJoueur == nbSetsWin:
            return 1
        elif setsWinAdversaire == nbSetsWin:
            return 0

j = 0
a = 0
nbMatchTennisBis = 2000
for i in range(nbMatchTennisBis):
    if SimuleMatchTennisBis(0.7, 0.5):
        j += 1
    else:
        a += 1

print("TennisBis : nb joueur win matchs :", j, "; nb adversaire win matchs :", a, "; joueur win matchs % :", j*100/nbMatchTennisBis, "; adversaire win matchs % :", a*100/nbMatchTennisBis)

# volley
def SimuleSetVolley(p, q):
    scoreJoueur = 0
    scoreAdversaire = 0
    sere = True
    while True:
        if scoreJoueur > 15 and scoreJoueur - 2 >= scoreAdversaire:
            return 1
        elif scoreAdversaire > 15 and scoreAdversaire - 2 >= scoreJoueur:
            return 0

        coup = SimuleCoup(p if sere else q)

        if coup:
            if sere:
                scoreJoueur += 1
            else:
                scoreAdversaire += 1
        else:
            sere = not sere

def SimuleMatchVolley(p, q):
    setsWinJoueur = 0
    setsWinAdversaire = 0
    nbSetsWin = 3
    while True:
        if SimuleSetVolley(p, q):
            setsWinJoueur += 1
        else:
            setsWinAdversaire += 1
        if setsWinJoueur == nbSetsWin:
            return 1
        elif setsWinAdversaire == nbSetsWin:
            return 0

j = 0
a = 0
nbMatchVolley = 2000
for i in range(nbMatchVolley):
    if SimuleMatchVolley(0.5, 0.5):
        j += 1
    else:
        a += 1

print("Volley : nb joueur win matchs :", j, "; nb adversaire win matchs :", a, "; joueur win matchs % :", j*100/nbMatchVolley, "; adversaire win matchs % :", a*100/nbMatchVolley)


# 7. fait

# Monty Hall
def initJeux():
    return random.randint(1,3)

def choixJoueur():
    return random.randint(1,3)

def animateur(bon, choix):
    ports = [1, 2, 3]
    ports.remove(bon)
    if bon != choix:
        ports.remove(choix)
    return ports[random.randint(0, len(ports) - 1)]

def strategieJoueur_tetu(choixInitial, choixAnimateur):
    # Joueur qui ne change pas son choix
    return choixInitial

def strategieJoueur_change(choixInitial, choixAnimateur):
    # Joueur "qui change"
    ports = [1, 2, 3]
    ports.remove(choixInitial)
    if choixInitial != choixAnimateur:
        ports.remove(choixAnimateur)
    return ports[0]

def strategieJoueur_hasard(choixInitial, choixAnimateur):
    # Joueur qui choisit aléatoirement de changer ou de ne pas changer
    ports = [1, 2, 3]
    ports.remove(choixAnimateur)
    return ports[random.randint(0, len(ports) - 1)]

def simuleJeu(iJ, cJ, a, sJ):
    # simule le jeu; les parametres sont des fonctions
    choixBon = iJ()
    choixInitial = cJ()
    choixAnimateur = a(choixBon, choixInitial)

    return 1 if sJ(choixInitial, choixAnimateur) == choixBon else 0


gagner = 0
nbSimulation = 2000
for i in range(nbSimulation):
    gagner += simuleJeu(initJeux, choixJoueur, animateur, strategieJoueur_tetu)

print("Monty Hall (tetu) : nb de jeu win matchs :", gagner, "; jeu gangner en % :", gagner*100/nbSimulation)


gagner = 0
nbSimulation = 2000
for i in range(nbSimulation):
    gagner += simuleJeu(initJeux, choixJoueur, animateur, strategieJoueur_change)

print("Monty Hall (change) : nb de jeu win matchs :", gagner, "; jeu gangner en % :", gagner*100/nbSimulation)


gagner = 0
nbSimulation = 2000
for i in range(nbSimulation):
    gagner += simuleJeu(initJeux, choixJoueur, animateur, strategieJoueur_hasard)

print("Monty Hall (matchs) : nb de jeu win matchs :", gagner, "; jeu gangner en % :", gagner*100/nbSimulation)
