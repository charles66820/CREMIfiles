(* Charles Goedefroit A11 *)
(* Implementation avec listes *)
(* Exercice 1 *)
(* 1. *)
type ('a,'b) dict = D of ('a * 'b) list;;

(* 2. *)
let dict_empty = [];;

(* Exercice 2 *)
(* 3. *)
let rec dict_add key value dict =
    match dict with
        [] -> (key, value)::[]
        | (k, v)::t -> if key = k then (k, value)::t
                    else (k, v)::(dict_add key value t);;

(* Exercice 3 *)
(* 4. *)
let rec dict_find key dict =
    match dict with
        [] -> None
        | (k, v)::t -> if k = key then Some v
                       else dict_find key t;;

(* Exercice 4 *)
(* 5. *)
let rec dict_remove key dict =
    match dict with
        [] -> []
        | (k, v)::t -> if k = key then dict_remove key t
                       else (k, v)::(dict_remove key t);;

