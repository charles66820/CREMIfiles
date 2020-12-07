(* Logique du premier ordre *)

(** Tactiques :
  pour forall :
    introduction :
            intro, intros.
    elimination :
            apply, eapply, specialize.
            (H x ...)

  pour exists :
     introduction: exists (fournir le terme)
     elimintation: destruct.

  pour = reflexivity (introduction),
         rewrite H   [in H\u00d8] (elimination)
         rewrite <- H [in H0]
 *)

(* tactique maison pour eliminer un forall *)
(* il faut fournir le terme temoin *)
(* c'est une alternative \u00e0 specialize *)

Ltac forall_e H t := (generalize (H t); intro).

(* Exemple *)

Example E0 : ~(forall x:nat, x <> x).
Proof.
  intro H.
  specialize H with (x:=42) as H0.
  (* equivalent: forall_e H 42. *)
  apply H0.
  reflexivity.
Qed.

Section Syllogismes. (* Entrainements *)
  Variable Etre: Type.
  Variables humain mortel animal : Etre -> Prop.

  Variable Socrate : Etre.
  Variable Rhino : Etre.

  Hypothesis HM : forall x,  humain x -> mortel x.
  Hypothesis HSocrate : humain Socrate.
  Hypothesis Etre_disj : forall x:Etre,  humain x \/ animal x.
  Hypothesis Hrhino : ~ humain Rhino.

  (** comme humain est de type Etre->Prop, Coq voit tout seul que
      le "x" dans HM est forc\u00e9ment de type Etre, et on n'est pas
      oblig\u00e9 de le pr\u00e9ciser *)

  Lemma Syllogisme :  mortel Socrate.
  Proof.
    apply HM. (* elimination du forall et modus-ponens *)
    assumption.
  Qed.

  Lemma contraposee : forall x, ~ mortel x -> ~ humain x.
  Proof.
  Admitted.

  Check Syllogisme.
  Lemma Existence_mortel: exists x, mortel x.
  Proof.
   exists Socrate.  (* introduction de l'existentiel *)
   apply Syllogisme. (* les lemmes d\u00e9montr\u00e9s sont utilisables *)
  Qed.

  Lemma Existence_animal: exists x, animal x.
  Proof.
    exists Rhino.
    destruct (Etre_disj Rhino).
    (* elimination sur la disjonction "Etre_disj" appliqu\u00e9e \u00e0 Rhino *)
    - contradiction. (* equivalent: exfalso; apply Hrhino; apply H. *)
      (* Coq cherche une contradiction \u00e9vidente dans les hypoth\u00e8ses *)
    - assumption.
  Qed.

  Lemma on_n_est_pas_des_animaux : ~(exists x:Etre,  ~ humain x /\ ~ animal x).
  Proof.
    intro H.
    destruct H as [e He]. (* elimination de l'existentiel *)
  Admitted.

End Syllogismes.

Section Egalite. (* Entrainements, sur l'egalite *)
  Variable A : Set.
  Variable f : A -> A.

  Lemma Egalite1 : forall x:A, exists y: A, x=y.
  Proof.
   intros x.
   exists x.
   reflexivity.
   Qed.

  Lemma Egalite2 : forall x y z: A, x = y -> y = z -> x = z.
  Proof.
    intros x y z H H0.
    rewrite H.
    assumption.
  Qed.

  (* x <> y est une abr\u00e9viation de ~ (x = y) *)

  Lemma Difference_egalite : forall x y z:A, x <> y -> y = z -> x <> z.
  Proof.
    intros.
    rewrite H0 in H.
    assumption.
  Qed.

  Lemma Egalite_difference : forall x y z: A , x = y -> x <> z -> z <> y.
  Proof.
    intros.
    intro.
     rewrite H1 in H0.
    apply H0; assumption.
   Qed.

   Lemma Difference_fonction : forall x y:A, f x <> f y -> x <> y.
   Proof.
     intros.
     intro H0.
     rewrite <- H0 in H.
     apply H.
     reflexivity.
   Qed.

End Egalite.

(* Supprimer les "Admitted" (on admet la preuve compl\u00e8te) et les "admit"
   (on admet le but courant) dans toutes les preuves qui suivent, et les
   compl\u00e9ter *)

Section ExercicesIntuitionnistes.
  Variable A B: Set.
  Variables P Q : A -> Prop.
  Variable R : A -> B -> Prop.
  Variable X : Prop.
  Variable f: A->A.

  Lemma Forall_and_eq : (forall x:A, P x /\ Q x) <->
                    (forall x:A, P x) /\ (forall x:A, Q x).
  Proof.
    split; intro H.
    - split.
      + intro x.
        specialize H with x.
        destruct H.
        assumption.
      + admit.
  Admitted.

  Lemma Forall_or_impl : (forall x, P x) \/ (forall x, Q x) ->
                    forall x, P x \/ Q x.
  Proof.
  Admitted.

  Lemma Exists_and_impl : (exists x:A, P x /\ Q x) ->
                    (exists x:A, P x) /\  (exists x:A, Q x).
  Proof.
  Admitted.

  Lemma Exists_or_eq : (exists x:A, P x \/ Q x) <->
                    (exists x:A, P x) \/   (exists x:A, Q x).
  Proof.
  Admitted.

  Section Forall_exists_exists.
    Hypothesis H : forall x, P x -> Q x.
    Hypothesis H0 : exists x, P x.

    Lemma L7 : exists x, Q x.
    Proof.
    Admitted.

  End Forall_exists_exists.

  Lemma L8 : forall x,  (P x -> exists y,  P y).
  Proof.
  Admitted.

  Lemma NonExists_ForallNot_eq : ~(exists x, P x) <-> forall x, ~ P x.
  Proof.
  Admitted.

  Lemma Exists_forall_eq : ((exists x, P x) -> X) <->
                     forall x, P x -> X.
  Proof.
  Admitted.

  Lemma ExForall_ForallEx :  (exists x:A, forall y:B, R x y)
                      -> (forall y:B, exists x:A, R x y).
  Proof.
  Admitted.

  (* Sur l egalite *)
  (* Vu en cours *)
  Lemma eq_sym : forall x y:A, x = y -> y = x.
  Proof.
    intros x y H.
    rewrite H.
    reflexivity.
  Qed.

  (* Vu en cours aussi *)
  Lemma eq_trans : forall x y z:A, x = y -> y = z -> x = z.
  Proof.
  Admitted.

  (* Pas vu en cours *)
  Lemma eq_function: forall x y: A, x = y -> f x = f y.
  Proof.
  Admitted.

  (* Sur les types vides *)

  Definition A_est_vide := forall x:A, x <> x.


  Lemma Vide_forall : A_est_vide -> forall x:A, P x.
  Proof.
    unfold A_est_vide. (* A compl\u00e9ter *)
  Admitted.

  Lemma TousDifferents_vide : (forall x y:A, x <> y) -> A_est_vide.
  Proof.
  Admitted.

End ExercicesIntuitionnistes.

(* On passe en logique classique *)

Require Import Classical.

Section ExercicesClassiques.
  Variable A B: Set.
  Variables P Q : A -> Prop.
  Variable R : A -> B -> Prop.
  Variable X : Prop.

  Hypothesis exm : forall X : Prop, X \/ ~X.

  Ltac add_exm  P :=
  let hname := fresh "exm" in
  assert(hname := exm  P).

(** Deux facons de faire un tiers exclus:
    - add_exm (formule)
    - destruct (classic <formule>)
 *)

(* ne pas essayer de comprendre :
   la tactique absurdK applique le raisonnement par l'absurde classique:

  Transforme un but  "Gamma |- P " en
                     "Gamma, ~P |- False" *)
  Ltac absurdK :=
    match goal with |- ?X =>
                    let hname := fresh "exm" in
                    assert(hname := exm  X);
                      destruct hname;[assumption| elimtype False]
    end.

  Lemma NonForall_Exists_eq : ~ (forall x, P x) <-> exists x, ~ P x.
  Proof.
    split.
    - intro H.
      absurdK.
      apply H.
      admit.    (*   remplacer le admit *)
    -
  Admitted. (*   finir la preuve *)

  Section Empty_again.

    Definition A_vide := forall x:A, x <> x.
    Hypothesis H : ~ A_vide.
    Hypothesis H0 : forall x:A, P x.

    Lemma NonVide_Forall_Exists : exists x:A, P x. (* difficile *)
    Proof.
      unfold A_est_vide in H.
      assert (exists x:A, x = x).
      {  absurdK.
         admit.
      }
    Admitted.

  End Empty_again.
End ExercicesClassiques.

Section drinkers_problem.
  (* Dans tout bar non vide, il existe quelqu'un qui, s'il boit,
     tout le monde boit *)
  (* On se place en logique classique: on reprend donc le tiers exclus
     et l'absurde classique *)
  Hypothesis exm : forall X : Prop, X \/ ~X.
  Ltac add_exm  P :=
  let hname := fresh "exm" in
  assert(hname := exm  P).

  Ltac absurdK :=
    match goal with |- ?X =>
                    let hname := fresh "exm" in
                    assert(hname := exm  X);
                      destruct hname;[assumption| elimtype False]
    end.

  Variable people : Type.
  Variable patron : people.

  Variable boit : people -> Prop.
  Theorem buveurs :
    exists p:people, boit p -> forall q, boit q.
  Proof.
    add_exm (forall q, boit q).
    destruct exm0.
    - exists patron;auto.
    - assert (exists x:people, ~ boit x).
       {
         admit.
       }
 Admitted. (* compl\u00e9ter cette preuve *)

End drinkers_problem.
