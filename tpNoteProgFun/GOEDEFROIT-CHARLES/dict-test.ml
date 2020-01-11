let dict = make_dict [(1, "un"); (2, "deux"); (3, "trois"); (4, "quatre")]

let _ = assert(dict_find 1 dict = Some "un")         
let _ = assert(dict_find 2 dict = Some "deux")         
let _ = assert(dict_find 3 dict = Some "trois")         
let _ = assert(dict_find 4 dict = Some "quatre")
let _ = assert(dict_find 5 dict = None)
let _ = assert(None = dict_find 4 (dict_remove 4 dict))
let _ = assert(Some "cinq" = dict_find 5 (dict_add 5 "cinq" dict))

