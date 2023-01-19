#

## Sujet

- look for numerical behavior
- Gram-Schmidt orthogonalization process.
- the orthogonality of computed vectors can be lost with Gram-Schmidt (basic or edited)
- use these results in the context of Arnoldi process for constructing an orthogonal basis of a sequence of associated Krylov subspaces.
- important conclusions about :
  - parallel implementation
  - efficiency of computational
- fréquemment utilisé car simple
- problème de stabilité numérique (perte d'orthogonalité)
- possibilité d'être parallélisé
- avec la version Gram-Schmidt modifiée il a été montré que la perte d'orthogonalité peut être borné à $\|I-\bar{Q}^T_i\bar{Q}_j\|\leq\zeta_1(m,j)\epsilon{}k(A_j)$

$\epsilon$ is the machine precision

- avec la version Gram-Schmidt modifiée avec reorthogonalization il a été montré que la perte d'orthogonalité peut être borné à $\|I-\bar{Q}^T_i\bar{Q}_j\|\leq\zeta_3(m,j)\epsilon{}$
- la version réorthogonalisée fonctionne très bien.
- 

L'algo orthogonalise puis normalise.

a = base de base
a^(1) = la base orthogonalisé
q = la base orthonormé (orthogonalisé + normé)

## Notes

- la factorisation QR de Gram-Schmidt modifiée
- la factorisation QR de Householder
- matrices inversibles : ??
- valeur singulière : ??

## Definitions

### Gram-Schmidt (CGS)


### Gram-Schmidt modifier (MGS)


### Gram-Schmidt réorthogonalisée

la méthode de Gram-Schmidt réorthogonalisée, qui utilise des itérations pour corriger les erreurs d'orthogonalité et maintenir une base orthonormale

### orthonormalization

L'orthonormalisation est un ensemble de techniques utilisées en analyse mathématique et physique pour rendre une base de vecteurs normaux et orthogonaux les uns aux autres. Cela permet de simplifier les calculs et de rendre les résultats plus faciles à interpréter. Il existe plusieurs méthodes d'orthonormalisation, comme la méthode de Gram-Schmidt et la factorisation QR.

### orthogonalization

L'orthogonalisation est un sous-ensemble de l'orthonormalisation, c'est une technique utilisée pour rendre une base de vecteurs orthogonaux les uns aux autres, c'est-à-dire que le produit scalaire de deux vecteurs différents de cette base est nul. Cela permet de simplifier les calculs et de rendre les résultats plus faciles à interpréter. Il existe plusieurs méthodes d'orthogonalisation, comme la méthode de Gram-Schmidt, qui permet de construire une base orthogonale à partir d'une base quelconque.

### réorthogonalisation

La réorthogonalisation est une technique utilisée pour corriger les erreurs d'orthogonalisation dans les méthodes numériques telles que la méthode de Gram-Schmidt classique. Elle est utilisée pour maintenir l'orthogonalité des vecteurs dans une base en cours de construction lorsque les erreurs numériques sont importantes. Il existe plusieurs méthodes de réorthogonalisation, comme la méthode de Gram-Schmidt réorthogonalisée, qui utilise des itérations pour corriger les erreurs d'orthogonalité et maintenir une base orthonormale.
