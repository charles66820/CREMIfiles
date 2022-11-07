# TP1 : Apprentissage non supervisé.

Une page max pour le rapport, à rendre au + tard le 7 novembre (avant 8h).

## Analyse en composantes principales

Le chois entre l'ACP et l'ACP normé ce fair on fonction de si on veut gardé ou non des variables. L'ACP normé donne le même pois à toute les variables.

La valeur propre `λ_d` avec `d` la dimension (la variance empirique).

Avec `1` composante on couvre `68.65%` des données significative et une valeur propre de `λ_1=8.9`.
Avec `2` composantes on couvre `83.57%` des données significative et une valeur propre de `λ_2=1.9`.
Avec `3` composantes on couvre `94.77%` des données significative et une valeur propre de `λ_3=1.5`.
Avec `4` composantes on couvre `97.73%` des données significative et une valeur propre de `λ_4=0.3`.

Avlec la règle de Kaiser ont garde `3` composantes car les 3 premier valeurs propre sont supérieur à `1`.
Avlec la règle du coude ont garde `2` composante. (ou `3`).

Différences premières :

- $ε_1 = (λ_1-λ_2), ε_2 = (λ_2-λ_3), ε_3 = (λ_3-λ_4)$
- $ε_1 = (8.92-1.93), ε_2 = (1.93-1.45), ε_3 = (1.45-0.38)$
- $ε_1 = 7.99, ε_2 = 0.48, ε_3 = 1.07$

Différence secondes:

- $δ_1 = (ε_1-ε_2), δ_2 = (ε_2-ε_3)$
- $δ_1 = (7.99-0.48), δ_2 = (0.48-1.07)$
- $δ_1 = 7.51, δ_2 = -0.59$

## Partitionnement

