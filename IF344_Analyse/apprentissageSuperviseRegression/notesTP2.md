# TP2 : Apprentissage supervisé (Régression)

## Régression simple

- On vois que la régression suis plutôt bien les données sauf au début.
- On vois que les résidus sont centré ???
- On vois que les valeurs prédites corresponde bien au valeur observées sauf quelle sont inversé.

### régression linéaire [SIMPLE]

- La **p-value** (Pr(>|t|)) est bien inférieur à 5% (`< 2.2e-16`). Donc on rejette (A0) car `p-value` $\leq \alpha$, $< 2.2e-16 < 0.05$.
- La **t-value** (student test) vos `-24.53` et **qt** (*alphaT*) vos `1.964682`. Donc on rejette (A0) car $|-24.53| > 1.964682$.
- La **statistique de Fisher** (F-statistic) vos `601.6` et le **qf** (*alphaFisher*) vos `5.054041`. Donc on rejette (A0) car $601.6 > 5.054041$
- L'intervalle de confiance est `[-1.026148 ; -0.8739505]`. Donc on rejette (A0) car `0` n'est pas dans l'intervale. `0` car la courbe peut pas être à la fois ver la gauche et vers la droite.
- Le $R^2$ vos `0.5441` et le $R^2$ ajusté vos `0.5432`, l’adéquation du modèle aux données est donc bien car a mis chemin entre `0` et `1`.
- Pour `10` on vois que l'intervalle de confiance est très resserré $[24.47413 ; 25.63256]$ et que l'intervalle de prédiction est bien plus large $[12.82763 ; 37.27907]$.
- Avec le test de normalité (`Shapiro-Wilk`) on obtient bien la même `p-value` (`< 2.2e-16`). `W = 0.87857` ???
- La validation croisée, un moindre carré (MSE : la moyenne des résidu au carré), nous donne `38.8901`.

### régression non linéaire

- La **p-value** : on rejette (A0) car $< 2.2e-16 < 0.05$.
- La **t-value** :
On as plusieurs $\beta$ (e.g. `poly(x1, degpoly)1`, `poly(x1, degpoly)2`) on peut peut faire le test sur chacun des $\beta$ pour dir si il sont significatif ou non.
  - on rejette (A0) pour $\beta_1$ car $|-27.60| > 1.964682$.
  - on rejette (A0) pour $\beta_2$ car $|11.63| > 1.964682$.
- La **statistique de Fisher** (F-statistic) : on rejette (A0) car $448.5 > 5.054041$
- L'intervalle de confiance :
  - on rejette (A0) pour $\beta_1$ car `0` n'est pas dans l'intervale `[-163.31194 ; -141.60716]`.
  - on rejette (A0) pour $\beta_2$ car `0` n'est pas dans l'intervale `[53.37485 ; 75.07963]`.
- Le $R^2$ vos `0.6407` et le $R^2$ ajusté vos `0.6393`.
- Avec le test de normalité (`Shapiro-Wilk`) on obtient une `p-value` plus élevé mais toujours en dessous de 5% (`6.101e-14`). `W = 0.93583` ???
- La validation croisée, un moindre carré (MSE : la moyenne des résidu au carré), nous donne `30.73622`.

TODO: comparaison
