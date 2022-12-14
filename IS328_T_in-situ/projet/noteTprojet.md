# Traitement de données in-situ

## notes

Un paramètre supplémentaire qui s'appelle `rcut_inc`. (INCrément du Rayon de coupure (CUT)). Garantie qu'une particule n'est pas dans le voisinage si elle ce déplace de moins de $rcut_inc / 2$.

équation de plan : donne une distantce (en gros pour donné la distance d'un point à un plan). $A\cdot x+B\cdot y+C\cdot z+D$.

```c++
double A = direction.x, B=direction.y, C=direction.z;
double D = -(A*origin.x + B*origin.y + C*origin.z);
// assert(A*origin.x + B*origin.y + C*origin.z + D == 0);
```

## instructions

1. Git :

```bash
  # créer sa branche "enseirb-${USER}"
  # mettre à jour sa branche
  git pull origin master
```

2. build / exec

```bash
source ~/build/setup-env.sh
salloc -proutage -n1 -c12 make -j24
# copier le fichier de test dans sont dossier
cd ..
cp exaStamp/data/samples/tutorial_slice_plot.msp project/
# plus grosse simu
cp exaStamp/data/samples/tutorial_slice_plot_mj.msp project/
cd project
salloc -proutage --exclusive -N4 -n4 -c24 mpirun ~/build/xstampv2 tutorial_slice_plot.msp --profiling-vite trace.vite
```

Le fichier où ce trouve le code est `exaStamp/src/tutorial/slice_plot.cpp`

Le paramètre `domain` défini la géométrie du domain.

les consignes sont sur le moodle.

Rendre le code et le rapport par email avant le 31 janvier 23h59.
Pour le code utilisé les `git bundle`. On peut aussi push ça branche en plus.

```bash
git bundle create <nom>-<prenom>.bundle master~20..enseirb-${USER}
```

l'email du prof `thierry.carrard@cea.fr`.
