#include "limits.h"

/* Cette fonction assure que pour tout i, si i est entre 0 et n-1, tab[i] = 0.
Elle assure également qu’il existe au moins une telle case.
En ACSL, pour tout i s’écrit "\forall int i;", et il existe i, "\exists int i;".
*/
void set_to_zero(int *tab, int n);