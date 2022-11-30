# Traitement de données in-situ

## notes

Un paramètre supplémentaire qui s'appelle `rcut_inc`. (INCrément du Rayon de coupure (CUT)). Garantie qu'une particule n'est pas dans le voisinage si elle ce déplace de moins de $rcut_inc / 2$.

équation de plan : donne une distantce (en gros pour donné la distance d'un point à un plan). $A\cdot x+B\cdot y+C\cdot z+D$.

```c++
double A = direction.x, B=direction.y, C=direction.z;
double D = -(A*origin.x + B*origin.y + C*origin.z);
// assert(A*origin.x + B*origin.y + C*origin.z + D == 0);
```
