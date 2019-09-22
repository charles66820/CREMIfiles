12 + 30 (* donne 42 *)
12.0 + 30 (* erreur de type car on additionne un foat a un int *)
12.0 + 30.0 (* erreur de type car on utilise l'opérateur d'addition de int *)
int_of_float(12.5) + 30 (* donne 42 *)
float_of_int(12) + 30.0 (* donne 42.0 *)

