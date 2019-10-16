let rec sum_product f g i j =
    if i > j then 0
    else (f i) * (g (j-i)) + sum_product f g (i+1) j;;
         (* le calcule     +  l'élément suivant *)


let convolution f g = fun x -> sum_product f g 0 x;;

