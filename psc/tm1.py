#!/usr/bin/python3

# Probabilités, Statistiques, Combinatoire (Licence 2)
# TM1: génération exhaustive

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

# Exercice 1

def EntierVersBinaire(k,n):
    if n >= 2**k:
        return None
    def aux(k, n, acc):
        if k <= 0:
            return acc
        if n - (2**(k-1)) >= 0 and n > 0:
            return aux(k-1, n - 2**(k-1), acc + (1,))
        else:
            return aux(k-1, n, acc + (0,))
    return aux(k, n, ())

def BinaireVersEntier(t):
    def aux(k, t, acc):
        if t == ():
            return acc
        return aux(k-1, t[1:], acc + (t[0] * 2**k))

    return aux(len(t)-1, t, 0)

def ToutesSequencesBinaires(k):
    acc = ()
    for v in range(2**k-1, 0, -1):
        acc += (EntierVersBinaire(k, v),)
    return acc

def Consecutifs(t):
  def aux(t, tc, mc):
    if t == ():
      return mc
    if t[0] == 1:
      if tc+1 > mc :
        return aux(t[1:], tc+1, tc+1)
      else:
        return aux(t[1:], tc+1, mc)
    else:
      return aux(t[1:], 0, mc)
  return aux(t, 0, 0)

# Nombre de séquences sans k 1 consecutifs
# On donne la liste pour éviter de la calculer une fois pour chaque k
def CompteConsecutifs(L,k):
    seq = ToutesSequencesBinaires(k)
    seqlen = len(seq)
    res = 0;
    for i in range(0, seqlen):
        x = Consecutifs(seq[i])
        if x == L:
            if res < seqlen/2:
                res += 1
    return res


# Pour chaque valeur k de 1 à m: affiche combien de séquences binaires
# de longueur n ont au moins k 1 consécutifs
def Comptages(n,m):
    for k in range(1, m):
        print("On trouve", CompteConsecutifs(n, k), "séquences binaires consécutive de 1 de longeur", n, "pour la valeur", k)
Comptages(3,20)# TODO: test
# Version récursive de ToutesSequencesBinaires
def ToutesSequencesBinairesRec(n):
    def aux(v):
      if v < 0:
        return ()
      return (EntierVersBinaire(n, v),) + aux(v-1)
    return aux(2**n-1)

def ToutesSequencesBinairesRecT(n):
    def aux(v, acc):
      if v < 0:
        return acc
      return aux(v-1, acc + (EntierVersBinaire(n, v),))
    return aux(2**n-1, ())

# Exercice 2

def SequenceVersPartie(t):
    return None

def Parties(n,k):
    return None

def PremierePartie(n,k):
    return None

# Retourne la partie qui suit s dans l'ordre lexicographique; on n'a pas
# besoin de k (longueur de s) mais on a besoin de n
def Suivante(n,s):
    return None

# Comme la fonction Parties, mais avec PremierePartie() et Suivante()
def PartiesBis(n,k):
    L=[]
    s = PremierePartie(n,k)
    while s!=None:
        L.append(s)
        s = Suivante(n,s)
    return L

# Exercice 3
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

# Nombre de dérangements parmi les permutations de taille n
def NbDerangements(n):
    return None
