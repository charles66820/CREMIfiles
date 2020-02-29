#!/usr/bin/python3

# Probabilités, Statistiques, Combinatoire (Licence 2)
# TM2: génération exhaustive, générateurs Python

# Page du cours: www.labri.fr/perso/duchon/Enseignements/Probas/

# Certaines fonctions sont fournies comme exemples; celles dont le
# corps est seulement "return None" sont celles qu'il faut écrire soi-même.
# Il est possible que pour certains exercices, il soit judicieux d'écrire
# des fonctions en plus!

# pour vérifier qu'une liste ne contient pas de doublons
# (il faut que python arrive à trier... pas des choses trop inhomogènes donc)
# Des listes ou tuples d'entiers sont triés dans l'ordre lexicographique
def TousDifferents(L):
    LL = sorted(L)
    m = len(LL)
    for i in range(m-1):
        if LL[i]==LL[i+1]:
            return False
    return True

# Venant du TM1

def EntierVersBinaire(k,n):
    L=[]
    for i in range(k):
        L.append(0)
    for i in range(k-1,-1,-1):
        if (n%2)==1:
            L[i]=1
        n = n//2
    return tuple(L)

def ToutesSequencesBinaires(k):
    L=[]
    for n in range(2**k):
        L.append(EntierVersBinaire(k,n))
    return L

# Version récursive de ToutesSequencesBinaires
def ToutesSequencesBinairesRec(n):
    if n<=0: #on peut accepter à partir de n=1
        return [()]
    L=ToutesSequencesBinairesRec(n-1)
    M=[]
    for t in L:
        M.append(t+(0,))
        M.append(t+(1,))
    return  M

# Exercice 2

def SequenceVersPartie(t):
    k = len(t)
    L=[]
    for i in range(k):
        if t[i]==1:
            L.append(i+1)
    return tuple(L)

# Version alternative, SANS passer par les séquences binaires
# On exploite (récursivement) la preuve bijective de la formule
# du triangle de Pascal: Bin(n,k) = Bin(n-1,k) + Bin(n-1,k-1)
def PartiesRec(n,k):
    if k<=0:
        return [()]
    if k==n:
        s = tuple([i+1 for i in range(n)])
        return [s]
    L=PartiesRec(n-1,k)
    M=PartiesRec(n-1,k-1)
    for s in M:
        # s+(n,), c'est la partie s (k-1 éléments parmi n-1) avec n à la fin
        L.append(s+(n,))
    return L


# Permutations

# Toutes les séquences sous-diagonales, version récursive
def ToutesSeqSousDiagoRec(n):
    if n<=0:
        return [()]
    L = ToutesSeqSousDiagoRec(n-1)
    M = []
    for s in L:
        for k in range(1,n+1):
            M.append(s+(k,))
    return M

# Version itérative, à écrire
def ToutesSeqSousDiago(n):
    return None

# Conversion d'une sequence sous-diagonale en permutation
# S'inspirer de l'ex. 1.8 feuille TD1
def SeqVersPerm(s):
    return None

# Toutes les permutations: par conversion depuis les séquences
# sous-diagonales
def ToutesPermutations(n):
    L = [SeqVersPerm(s) for s in ToutesSeqSousDiago(n)]
    return L

# Nombre de points fixe d'une permutation s
def PointsFixes(s):
    return None

# Nombre total de points fixes dans toutes les permutations
# de taille n
def ComptePointsFixes(n):
    L = ToutesPermutations(n)
    total = 0
    for s in L:
        total = total + PointsFixes(s)
    return total


# Compte les 2-cycles
def DeuxCycles(s):
    return None

# Compte tous les 2-cycles dans toutes les permutations de [[1,n]]
def CompteDeuxCycles(n):
    return None


#### Mots de Dyck

def EstDyck(s):
    return None

def TousMotsDyck(n):
    return None

def Pics(s):
    return None

# Compte tous les pics dans tous les mots de Dyck de longueur 2n
def ComptePics(n):
    return None

########################################
# Generateurs (ou iterateurs)          #
#######################################

# "range(n)" simplifie
def my_range(n):
    k=0
    while k<n:
        yield k
        k = k+1

def genSeqBinRec(n):
    if n<=0:
        yield ()
    else:
        for s in genSeqBinRec(n-1):
            yield s+(0,)
            yield s+(1,)

# Toutes parties de [[1,n]]: par conversion des séquences binaires?
def genToutesParties(n):
    return None

# Parties de taille n: idées...
# - filtrage depuis genToutesParties (un peu long)
# - récursif sur k
def genParties(n,k):
    return None

# Tous mots de Dyck de longueur 2n:
# par filtrage sur les séquences binaires (pas horrible)
def genDyck(n):
    return None

# mots de Dyck, version récursive: penser au codage des arbres
def genDyckRec(n):
    return None

# mots de Dyck, version récursive, bis: récursivement, un générateur
# pour les mots positifs de longueur m, avec |w|_1-|w|_0=k (k=0,m=2n: Dyck)
def genPositifRec(m,k):
    if m==0 and k==0:
        yield ()
    if k>m or k<0:
        pass #pas de mots possibles
    else:
        if m>0 and k>=0:
        # deux facons possibles d'obtenir un mot:
        # - ajouter un b (0) à un mot de longueur m-1 finissant à hauteur k+1
        # - ajouter un a (1) à un mot de longueur m-1 finissant à hauteur k-1
            return None # A remplacer...
