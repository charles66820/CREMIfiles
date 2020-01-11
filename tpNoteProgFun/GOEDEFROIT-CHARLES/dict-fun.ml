(* Charles Goedefroit A11 *)
(* Implementation avec fonctions *)
(* Exercice 9 *)
(* 10 *)
type ('a,'b) dict = D of 'a * 'b option | None;;

(* 11. *)
let dict_empty = None;;

(* 12 *)
let rec dict_add key value dict =
    match dict with
        None -> D((key, value), None)
        | D((k, v), t) -> if key = k then D((k, value), t)
                    else D((k, v), (dict_add key value t));;


(* 13 *)
let rec dict_find key dict =
    match dict with
        None -> None
        | D((k, v), t) -> if k = key then Some v
                       else dict_find key t;;

(* 14 *)
let rec dict_remove key dict =
    match dict with
        None -> None
        | D((k, v), t) -> if k = key then dict_remove key t
                       else D((k, v), (dict_remove key t));;

