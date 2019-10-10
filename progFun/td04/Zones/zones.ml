type mycomplex = C of float * float

let realpart c = match c with C(x,_) -> x
let imagpart c = match c with C(_,y) -> y

let c_origin = C(0.,0.)
let p_12 = C(1.,2.)

let square x = x * x

let c_abs c = C(sqrt square (realpart c) , sqrt square (imagpart c))
let c_sum c1 c2 = C(realpart c1 +. realpart c2, imagpart c1 +. imagpart c2)
let c_dif c1 c2 = 
let c_mul c1 c2 = C(realpart c1 *. realpart c2, imagpart c1 *. imagpart c2)
let c_sca lambda c = 
let c_exp c = C(exp realpart c, exp imagpart c)

(* A zone is represented as a function that takes a point in 2-dimensional
   space as a parameter (represented as C(x, y)), and returns true
   if and only if the point is in the zone, and false otherwise.
*)

(* A zone that contains no points.  A point is never in this zone. *)
let nowhere = fun point -> false

(*  To determine whether a point is in a zone, just call this function. *)
let point_in_zone_p point zone = 

(* Create a circular zone with center in (0,0) with the indicated radius. *)
let make_disk0 radius = fun point -> c_abs point <= radius

(* Given a zone, move it by a vector indicated as a complex number
   passed as the argument. *)
let move_zone zone vector = fun p -> point_in_zone_p (c_dif p vector) zone

(* A zone that contains every point. A point is always in this zone. *)
let everywhere = 

(* Make a rectangle in the first quadrant. *)
let make_rectangle width height =

(* Given two zones, create a zone that behaves as the intersection of the two. *)
let zone_intersection zone1 zone2 =

(* Test all zone_manipulating code. *)
let test = 
  let c = make_disk0 1. in
  let c1 = move_zone c (C(1., 0.)) in
  assert (point_in_zone_p (C(0.0, 0.5)) c);
  assert (not (point_in_zone_p (C(1.0, 0.5)) c));
  assert (point_in_zone_p (C(0.5, 0.)) (zone_intersection c c1))

(* point_in_zone_p (C(3.5 2.1)) everywhere *)

(* Given two zones, create a zone that behaves as the union of the two. *)
let zone_union zone1 zone2 =

(* Given a zone, create a zone that contains every point not in zone. *)
let zone_complement zone =
					       
let make_disk radius center =
  fun p -> (c_abs (c_dif center p)) <= radius

(* point_in_zone_p (C(2., 2.)) (make_disk 1. (C((1.5, 1.5))) *)

(* Scale a zone in two dimensions *)
let scale_zone0 zone coeff =

(* Test scale_zone *)
(* point_in_zone_p (C(5., 5.)) (scale_zone0 (make_disk0 2.) (10., 10.)) *)

let scale_zone zone coeff origin =

let _ = point_in_zone_p (C(6.0, 0.5)) (scale_zone0 (make_disk 1. (C(4.0, 4.0))) (C(6.0, 4.0)))

let c_i = C(0., -1.)

let rotate_zone0 zone angle =

(* point_in_zone_p (C(0.5, 8.)) (rotate_zone0 (make_rectangle 10. 2.) (3.1416 /. 2.)) *)

let rotate_zone zone angle center =


