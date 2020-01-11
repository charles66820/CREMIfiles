let dict = dict_empty
let dict = dict_add 0 "zero" (dict_add 1 "un" dict)
let _ = dict_find 1 dict
let _ = dict_find 2 dict
let _ = dict_find 0 (dict_remove 0 dict)
let _ = dict_find 1 (dict_remove 0 dict)
let dict = dict_add 3 "trois" (dict_add 4 "quatre" dict)

let fr_en = make_dict
              [("bleu", "blue"); ("le", "the"); ("ciel", "sky");
               ("jaune", "yellow"); ("rouge", "red") ; ("vert", "green")]

let _ = find_word "ciel" fr_en
let _ = find_word "est" fr_en
let words = ["le"; "ciel"; "est"; "bleu"]
let _ = translate words fr_en
