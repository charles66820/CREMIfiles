(* Charles Goedefroit A11 *)
(* Exercice 5 *)
(* 6. *)
let rec dict_adds couples dict =
    match couples with
        dict_empty -> dict
        | (k, v)::t -> if (dict_find k dict) = None then dict_adds t dict
                       else dict_adds t (dict_add k v dict);;

(* Exercice 6 *)
(* 7. *)
let make_dict couples =
    let rec aux c d =
        match c with
            dict_empty -> d
            | (k, v)::t -> aux t (dict_add k v d)
    in aux couples dict_empty;;

(* Exercice 7 *)
(* 8. *)
let rec find_word word dict =
    match dict with
        dict_empty -> "?"^word^"?"
        | (k, v)::t -> if k = word then v
                       else if v = word then k
                       else find_word word t;;

(* Exercice 8 *)
(* 9. *)
let rec translate words dict =
    match words with
        dict_empty -> dict_empty
        | h::t -> (find_word h dict)::(translate t dict);;
