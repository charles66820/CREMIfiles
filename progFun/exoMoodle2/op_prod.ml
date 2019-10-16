(* n et p = délimiteur de l'interval *)
(* f une fonction qui donne 'a *)
(* op = opération qui prend 'a et un élément de type 'b *)
(* natural = type netre de 'b (ser  çà rien
) *)
let rec op_prod n p f op neutral = 
    if n > p then neutral (* quant il n'y a pas d'interval *)
    else op (f n) (op_prod (n+1) p f op neutral)
    (* opération de (f )n avec l'element suivant n-1 *)

