let rec f n l l1 l2 = 
    match l with 
        [] -> l1, l2
        | h::t -> if h <= n then f n t (h::l1) l2
            else f n t l1 (h::l2);;

let split n l =
    f n l [] [];;


(* for test *)
let l = [1;6;2;7;8;5;3];;

split 5 l;;
