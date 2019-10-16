let rec sum n p = 
    if n>p then 
        0
    else
        n + sum (n+1) p
;;

