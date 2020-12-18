Ltac forall_e H t := (generalize (H t); intro).

Require Import Setoid.

Section Gaulois.
  Variable personnage: Set.
  Variables humain gaulois romain animal: personnage -> Prop.
  Variables Idefix Panoramix: personnage.

  Hypothesis Hngr: forall p:personnage, ~(gaulois p /\ romain p).
  Hypothesis Hpers: forall p:personnage, animal p \/ gaulois p \/ romain p.
  Hypothesis Hhum: forall p:personnage, humain p <-> (gaulois p \/ romain p).
  Hypothesis Hnon_humain_animal: forall p:personnage, ~(humain p /\ animal p).

  Hypothesis Hidef: animal Idefix.
  Hypothesis Hpano: gaulois Panoramix.
  Hypothesis Hrom: exists p:personnage, romain p.

  Theorem Exemple: exists x:personnage, humain x /\ ~gaulois x.
  Proof.
    destruct Hrom as [y Hy].
    exists y.
    split.
    - rewrite Hhum.
      right.
      assumption.
    - forall_e Hngr y.
      intro Hgy.
      apply H.
      split.
      + assumption.
      + assumption.
  Qed.

(* Dans les 5 th\u00e9or\u00e8mes ci-dessous, remplacez "Admitted." par un
   script de preuve complet, que vous terminerez par "Qed."

   Bar\u00e8me indicatif: pour chaque th\u00e9or\u00e8me, 3.5 points pour un
   script qui prouve le th\u00e9or\u00e8me sans rien admetter; 0.5 point
   suppl\u00e9mentaire si la preuve est structur\u00e9e de mani\u00e8re \u00e0
   permettre de la suivre sans interagir avec Coq (prendre exemple
   sur la preuve pr\u00e9c\u00e9dente).
  *)

  Theorem Exercice1: ~ gaulois Idefix.
  Proof.
    destruct Hrom as [y Hy].
    forall_e Hpers y.
    destruct H.
    - forall_e Hnon_humain_animal y.
      intro.
      apply H0.
      split.
      * forall_e Hhum y.
        destruct H2.
        apply H3.
        right.
        assumption.
      * assumption.
   - forall_e Hnon_humain_animal Idefix.
      intro.
      apply H0.
      split.
      * forall_e Hhum Idefix.
        destruct H2.
        apply H3.
        left.
        assumption.
      * assumption.
   Qed.

  Theorem Exercice2:
    forall p:personnage, humain p -> ~romain p -> gaulois p.
  Proof.
    intros p Hp NRp.
    forall_e Hpers p.
    destruct H.
    - 
 
  Qed.

  Theorem Exercice3:
    exists p:personnage, humain p /\ ~gaulois p.
  Proof.
  Admitted.

  Theorem Exercice4:
    forall p, ~animal p -> gaulois p \/ romain p.
  Proof.
  Admitted.

  Theorem Exercice5: Idefix <> Panoramix.
  Proof.
    (* Indication: on peut utiliser n'importe quel th\u00e9or\u00e8me,
       y compris un qui a \u00e9t\u00e9 prouv\u00e9 comme exercice *)
  Admitted.

End Gaulois.