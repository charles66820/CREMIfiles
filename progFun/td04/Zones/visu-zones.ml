(* visualisation *)

#load "graphics.cma"
#load "images.cmo"

open Graphics
open Images

let nb_pixels_per_unit = 10

let int_to_nb_pixels n = nb_pixels_per_unit * n

let pixel_to_float x = (float_of_int x) /. (float_of_int nb_pixels_per_unit)

let pixel_to_point px py =
  (C(pixel_to_float px, pixel_to_float py))

let zone_to_pixels zone n =
  let nb_pixels = int_to_nb_pixels n in
  let pixels = Array.make_matrix nb_pixels nb_pixels black in
  for x = 0 to nb_pixels - 1 do
    for y = 0 to nb_pixels - 1 do
      pixels.(x).(y) <- if point_in_zone_p (pixel_to_point x y) zone then red else white
    done
  done;
  pixels

let pixels_to_png pixels filename = sauver_image pixels filename

let zone_to_png zone n filename = pixels_to_png (zone_to_pixels zone n) filename

let random_point z = C(Random.float z, Random.float z)

let gruyere n =
  let zone = ref (make_disk0 40.)
  and disk = make_disk0 5. in
  for i = 0 to n do
    zone := zone_difference !zone (move_zone disk (random_point 40.))
  done;
  !zone

let uname () =
  let (inchannel, outchannel) = Unix.open_process "uname" in
  let name = input_line inchannel in
  close_in inchannel;
  close_out outchannel;
  name

let viewer =
  let uname = uname() in
  if uname = "Linux" then "eog "
  else if uname = "Darwin" then "open "
  else failwith "Viewer not set under windows"

let view_file file =
  Unix.system (viewer ^ file ^ "&")

   (* zone_to_png (gruyere 5) 50 "monimage.png" *)

let view_zone_size zone pixel_size =
  begin
    let filename = "monimage.png" in
    zone_to_png zone pixel_size filename;
    view_file filename;
  end

let view_zone zone = view_zone_size zone 50

(*
let z1 = zone_difference (make_disk0 10.) (move_zone (make_rectangle 5. 10.) (C(3., 3.)))
let z2 = move_zone (make_disk 15. (C(5., 5.))) (C(10., 10.))
let _ = view_zone (zone_union z1 z2)

view_zone (zone_union z1 z2)
 *)
