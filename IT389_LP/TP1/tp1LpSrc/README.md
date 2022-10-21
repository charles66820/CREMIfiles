
# Calcul du champ proche

Fichiers :

- my_types.hpp les types/classes utilisés dans le TP
- utils.hpp contient les méthodes pour calculer les listes, vérifier les résultats
- direct.hpp contient les méthodes pour calculer les forces
- main.cpp le programme principal

## Pour compiler

Sur PlaFRIM, utilisez le module gcc 12

```bash
module add compiler/gcc/12.2.0
```

- en séquentiel

  ``` bash
  g++ -O3 -o main main.cpp
  ```

- en parallèle

  ``` bash
   g++ -fopenmp -O3 -o main main.cpp
  ```

  La macro `RANDOM_NB_PARTICLES` permet de générer un nombre variable de particules dans chaque boîte.
