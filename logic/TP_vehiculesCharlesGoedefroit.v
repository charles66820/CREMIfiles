Require Import Setoid.

(* forall_e Hypothese t: introduit une nouvelle hypoth\u00e8se qui est ce
   qu'on obtient quand on applique Hypothese de la forme
   forall x:T, ...
   au terme t:T   *)
Ltac forall_e H t := generalize (H t); intro.

(* Alternativement:
   specialize Hypothese with t
   specialize Hypotheses with (x:=t)
   specialize Hypothese with t as Ht
   specialize Hypothese with (x:=t) as Ht
   les deux premi\u00e8res formes remplacent l'hypoth\u00e8se;
   specifier la variable \u00e0 lier avec (x:=t) permet de s\u00e9lectionner
   la variable x s'il y a plusieurs quantificateurs cons\u00e9cutifs,
   forall z y x, ...  *)

Section Vehicules.

(* Prenez le temps de vous familiariser avec les hypoth\u00e8ses, et de les
   interpr\u00e9ter mentalement: cela vous aidera pour vos preuves *)

  Variable vehicule: Type.

  Variable voiture: vehicule -> Prop.
  Variable velo: vehicule -> Prop.
  Variable rouge: vehicule -> Prop.
  Variable noir: vehicule -> Prop.
  Variable blanc: vehicule -> Prop.
  Variable ford: vehicule -> Prop.
  Variables totoche T: vehicule.

  Hypothesis Htot: rouge totoche.
  Hypothesis HT: ford T.
  Hypothesis Hford: forall v, ford v -> noir v.

  Hypothesis Hcouleur: forall v, rouge v \/ noir v \/ blanc v.
  Hypothesis Hrn: forall v, ~(rouge v /\ noir v).
  Hypothesis Hnb: forall v, ~(noir v /\ blanc v).
  Hypothesis Hrb: forall v, ~(rouge v /\ blanc v).

  Hypothesis Htype: forall v, voiture v \/ velo v.
  Hypothesis Hvv: forall v, ~(voiture v /\ velo v).

  Variable propulsion: Type.
  Variable categorie: vehicule -> propulsion -> Prop.
  Variable essence : propulsion.
  Variable electrique: propulsion.
  Variable pedale: propulsion.

  Hypothesis Hve: ~(exists v, velo v /\ categorie v essence).


(* Un exemple *)

  Lemma Exemple: noir T.
  Proof.
    forall_e Hford T.
    apply H; assumption.
  Qed.

(* Un second exemple *)
  Lemma Exemple_bis: forall v:vehicule, ~voiture v -> velo v.
  Proof.
    intros.
    forall_e Htype v.
    destruct H0.
    - contradiction.
    - assumption.
  Qed.

(* Les exercices commencent ici *)

(* Dans chaque Lemme, remplacer "Admitted" par votre propre script de preuve.
   N'oubliez pas de terminer chaque preuve par "Qed". *)

  Lemma Exercice1: ~ ford totoche.
  Proof.
    forall_e Hford totoche.
    intro.
    forall_e Hcouleur totoche.
    destruct H1.
    - forall_e Hrn totoche.
      apply H2.
      split.
      * assumption.
      * apply H; assumption.
    - forall_e Hrn totoche.
      apply H2.
      split.
      * assumption.
      * apply H; assumption.
  Qed.

  Lemma Exercice2: forall v:vehicule, noir v -> ~(rouge v \/ blanc v).
  Proof.
    intro v.
    intro Nv.
    intro VrOb.
    destruct VrOb as [Vr | Vb].
    - forall_e Hcouleur v.
      destruct H as [r | nOb].
      * forall_e Hrn v.
        apply H.
        split.
        + assumption.
        + assumption.
      * forall_e Hrb v.
        apply H.
        split.
        + assumption.
        + exfalso.
          forall_e Hrn v.
          apply H0.
          split; assumption.
    - forall_e Hnb v.
      apply H.
      split; assumption.
  Qed.

  Lemma Exercice3: (forall v, velo v-> rouge v) -> (forall w, ford w -> voiture w).
  Proof.
    intros Velo_r w HW.
    forall_e Velo_r w.
    forall_e Htype w.
    destruct H0.
    - assumption.
    - forall_e Hford w.
      forall_e Hrn w.
      exfalso.
      apply H2.
      split.
      * apply H; assumption.
      * apply H1; assumption.
  Qed.

  Lemma Exercice4: ~(totoche=T).
  Proof.
    intro.
    apply Exercice1.
    rewrite H.
    assumption.
  Qed.

  Lemma Exercice5: (exists v, blanc v /\ voiture v) -> exists v, voiture v /\ ~ford v.
  Proof.
    intro VbAV.
    destruct VbAV as [v VbAV].
    destruct VbAV.
    exists v.
    split.
    - forall_e Htype v.
      assumption.
    - intro HV.
      forall_e Hford v.
      forall_e Hnb v.
      apply H2.
      split.
      * apply H1; assumption.
      * assumption.
  Qed.

  Lemma Exercice6 : forall v,  categorie v essence -> voiture v.
  Proof.
    intros v Cve.
    forall_e Htype v.
    forall_e Hvv v.
    destruct H.
    - assumption.
    - exfalso.
      apply Hve.
      exists v.
      split; assumption.
  Qed.

End Vehicules.

(* Fin des exercices *)