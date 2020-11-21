Require Import Setoid.
(*  Logique intuitionniste *)

Section LJ.
  Variables P Q R S T : Prop.
  (*  Tactiques pour la conjonction

    Introduction : pour prouver A /\ B : split (il faudra prouver A, puis B)
    Elimination : destruct H, si H : A /\ B
                  variante : destruct H as [H1 H2].
        Dans les deux cas, on r\u00e9cup\u00e8re deux hypoth\u00e8ses pour A et B (et on
        choisit leurs noms, pour la variante "as..")
  *)
  Lemma and_comm : P /\ Q -> Q /\ P.
  Proof.
    intro H.
    destruct H as [H0 H1].

    split.
    - assumption.
    - assumption.
  Back 5.
    split; assumption. (* "assumption" r\u00e9sout les deux sous-buts *)
  Qed.

  (* tactiques pour la disjonction
    Introduction:
     pour prouver A \/ B a partir de A : left
     pour prouver A \/ B a partir de B : right

    Elimination:
     preuve par cas : destruct H, si H: A \/ B
                      variante : destruct H as [H1 | H2]
        On aura a faire deux preuves, une pour chaque cas (cas A, cas B)
  *)

  Lemma or_not : P \/ Q -> ~P -> Q.
  Proof.
   intros H H0.
   destruct H.
   - exfalso.
     apply H0; assumption.
     (* alternative: 
     assert (f:False).
     {
       apply H0; trivial.
     }
     destruct f. *)
     (* "destruct f" sur f:False r\u00e9soud n'importe quel but *)
   - assumption.
   Qed.

  (* Structuration de la preuve: +,*,+
     utiles quand on a plusieurs sous-preuves non triviales;
     am\u00e9liorent la lisibilit\u00e9 du script *)
  
   (*  equivalence logique (<->, iff):
       unfold iff transforme A <-> B en
                             (A -> B) /\ (B -> A).
       donc split, destruct, etc, marchent

       (iff pour "if and only if", le "si et seulement si" en anglais)
    *)

  Lemma iff_comm : (P <-> Q) -> (Q <-> P).
  Proof.
    intro H.
    unfold iff in H.
    destruct H.
    split.
    - assumption.
    - assumption.
    (* "assumption" r\u00e9soud les deux sous-buts engendr\u00e9s par "split"
    donc on peut remplacer les trois derni\u00e8res lignes par
    split; assumption.
    *)
  Back 5.
    split; assumption.
  Qed.

  (* la regle de remplacement est implant\u00e9e en Coq *)
  (* "rewrite H" fait un remplacement uniforme quand H est une
     \u00e9quivalence *)
  (* "rewrite H" r\u00e9\u00e9crit le but courant avec H *)
  (* "rewrite H in H'" fait la r\u00e9\u00e9criture de H dans une autre hypoth\u00e8se H' *)
  (* "rewrite <- H" r\u00e9\u00e9crit dans l'autre sens, le membre droit par le gauche *)
  Lemma L1 : (P <-> Q) -> ~(Q <-> ~P).
  Proof.  
    intro H.
    unfold iff. (* Car on a l'hypoth\u00e8se P <-> Q donc tous les p peuve \u00eatre remplacer par Q *)
    rewrite H.
    intro H0.
    destruct H0.
    assert (~Q).
    { intro H2.
      unfold not in H0.
      unfold not in H1.
      apply H0; assumption.
    }
    apply H2. apply H1. assumption. 
  Qed.

  (* Fin des exemples, d\u00e9but des exercices *)

  (* Exercice : remplacer tauto par des vraies preuves 
     interactives *)
  (*  Exercices de la feuille 4 *)

  Lemma and_false : P /\ False -> False.
  Proof. 
    intro.
    destruct H as [p F].
    apply F.
  Qed.

  Lemma and_assoc : (P /\ Q) /\ R <-> P /\ (Q /\ R).
  Proof.
    split.
    - intro H.
      destruct H as [pAq r].
      destruct pAq as [p q].
      split.
      * assumption.
      * split; assumption.
    - intro H.
      destruct H as [p qAr].
      destruct qAr as [q r].
      split.
      * split; assumption.
      * assumption.
  Qed.

  (* Ex. 2 *)
  Lemma or_to_imp: ~ P \/ Q -> P -> Q.
  Proof.
    intros NpOq p.
    destruct NpOq as [Np | q].
    - exfalso.
      apply Np; assumption.
    - assumption.
  Qed.

  Lemma not_or_and_not: ~(P\/Q) -> ~P /\ ~Q.
  Proof.
    intro NpOq.
    split.
    - intro p.
      apply NpOq.
      left; assumption.
    - intro q.
      apply NpOq.
      right; assumption.
  Qed.

  (* Exercice 4 *)

  Lemma absorption_or: P \/ False <-> P.
  Proof.
    split.
    - intro pOF.
      destruct pOF as [H | p].
      * assumption.
      * exfalso; assumption.
    - intro.
      left; assumption.
  Qed.

  Lemma and_or_dist : P /\ (Q \/ R) <-> P /\ Q \/ P /\ R.
  Proof.
    split.
    - intro H.
      destruct H as [p qOr].
      destruct qOr.
      * left.
        split; assumption.
      * right.
        split; assumption.
    - intro H.
      destruct H as [pAq | pAr].
      * destruct pAq as [p q].
        split.
        + assumption.
        + left; assumption.
      * destruct pAr as [p r].
        split.
        + assumption.
        + right; assumption.
  Qed.

  Lemma or_and_dist : P \/ (Q /\ R) <-> (P \/ Q) /\ (P \/ R).
  Proof.
    split.
    - intro pOqAr.
      destruct pOqAr as [p | qAr].
      * split; left; assumption.
      * destruct qAr as [q a].
        split; right; assumption.
    - intro H.
      destruct H as [pOq pOr].
      destruct pOq as [p | q].
      * left; assumption.
      * destruct pOr as [p | r].
        + left; assumption.
        + right.
          split; assumption.
  Qed.

  Lemma and_not_not_impl: P /\ ~ Q -> ~(P -> Q).
  Proof.
    intro pANq.
    destruct pANq as [p Nq].
    intro pq.
    apply Nq.
    apply pq.
    assumption.
  Qed.

  Lemma de_morgan1 : ~ (P \/ Q) <-> ~P /\ ~Q.
  Proof.
    split.
    - split.
      + intro.
        apply H.
        left; assumption.
      + intro.
        apply H.
        right; assumption.
    - intros NpONq pOq.
      destruct NpONq as [Np Nq].
      destruct pOq as [p | q].
      + apply Np; assumption.
      + apply Nq; assumption.
  Qed.

  Lemma reductio_ad_absurdum: (P -> ~P) -> ~P.
  Proof.
    intros pNp p.
    apply pNp; assumption.
  Qed.

  Lemma np_p_nnp: (~P -> P) -> ~~P.
  Proof.
    intros Npp Np.
    apply Np.
    apply Npp; assumption.
  Qed.

  (* Exercice: reprendre toutes les preuves pr\u00e9c\u00e9dentes, 
     en simplifiant et clarifiant les scripts:
     - structurer les sous-preuves avec +/-/*
     - inversement, quand c'est possible, factoriser avec 
       l'enchainement de tactiques (par ";")

     Le but est de faire que le script soit plus facile \u00e0 lire
     par un humain, pas pour la machine.
   *)
   (* TODO: here *) 

End LJ.

(*  Logique classique
    On peut sauter les 4 commandes suivantes 
*)

(* un peu de magie noire *)
Definition EXM :=   forall A:Prop, A \/ ~A.

Ltac add_exm  A :=
  let hname := fresh "exm" in
  assert(hname : A \/ ~A);[auto|].

Section LK.

  Hypothesis  exm :  EXM.

  (* 
   Pour ajouter une instance du tiers-exclu de la forme  A \/ ~A 
   il suffit d'ex\u00e9cuter la commande "add_exm A"
   *)

  Variables P Q R S T : Prop.

  Lemma double_neg : ~~ P -> P.
  Proof.
    intro H.
    add_exm  P. (* "je fais un tiers exclus sur P " *)
    destruct exm0. (* Presque toujours, destruct suit add_exm *)
    - assumption.
    - assert (f:False).
      {
        apply H; assumption.
      }
      destruct f. (* ou: exfalso, etc. *)
  Restart.
    intro H.
    add_exm  P. (* "je fais un tiers exclus sur P " *)
    destruct exm0. (* Presque toujours, destruct suit add_exm *)
    - assumption.
    - exfalso.
      apply H; assumption.
  Qed.

  (* Exercice: completer toutes les preuves, en rempla\u00e7ant les
     "Admitted" par des preuves termin\u00e9es par "Qed."; et 
     sans utiliser ni auto, ni tauto.  *)

  Lemma de_morgan : ~ ( P /\ Q) <-> ~P \/ ~Q.
  Proof.
    split.
      - intro H.
        add_exm P.
        destruct exm0 as [p | Np].
        * right.
          intro q.
          apply H.
          split; assumption.
        * left; assumption.
      - intros NpONq pAq.
        destruct pAq as [p q].
        destruct NpONq as [Np | Nq].
        * apply Np; assumption.
        * apply Nq; assumption.
  Qed.

  Lemma not_impl_and : ~(P -> Q) <-> P /\ ~ Q.
  Proof.
    split.
    - intro Npq.
      add_exm P.
      destruct exm0 as [p | Np].
      * split.
        + assumption.
        + intro q.
          apply Npq.
          intro p1; assumption.
      * split.
        + exfalso.
          apply Npq.
          intro p.
          exfalso.
          apply Np; assumption.
        + intro q.
          apply Npq.
          intro p; assumption.
    - intros pANq pq.
      destruct pANq as [p Nq].
      apply Nq.
      apply pq.
      assumption.
  Qed.

  Lemma contraposee: (P -> Q) <-> (~Q -> ~P).
  Proof.
    split.
    - intros pq Nq p.
      apply Nq.
      apply pq.
      assumption.
    - intros NqNp p.
      add_exm Q.
      destruct exm0 as [q | Nq].
      assumption.
      exfalso.
      apply NqNp; assumption.
  Qed.

  Lemma exm_e : (P -> Q) -> (~P -> Q) -> Q.
  Proof.
    intros pq Npq.
    add_exm P.
    destruct exm0 as [p | Np].
    - apply pq; assumption.
    - apply Npq; assumption.
  Qed.

  Lemma exo_16 : (~ P -> P) -> P.
  Proof.
    intro Npp.
    add_exm P.
    destruct exm0 as [p | Np].
    assumption.
    apply Npp; assumption.
  Qed.

  Lemma double_impl : (P -> Q) \/ (Q -> P).
  Proof.
    add_exm P.
    destruct exm0 as [p | Np].
    - right.
      intro q; assumption.
    - left.
      intro p.
      exfalso.
      apply Np; assumption.
  Qed.

  Lemma imp_translation : (P -> Q) <-> ~P \/ Q.
  Proof.
    split.
    - add_exm P.
      destruct exm0 as [p | Np].
      intro pq.
      * right.
        apply pq; assumption.
      * left; assumption.
    - add_exm Q.
      destruct exm0 as [q | Nq]. 
      * intros NpOq p; assumption.
      * intros NpOq p.
        destruct NpOq as [Np | q].
        + exfalso.
          apply Np; assumption.
        + assumption.
  Qed.

  Lemma Peirce : (( P -> Q) -> P) -> P.
  Proof.
    add_exm Q.
    destruct exm0 as [q | Nq].
    - intro pqp.
      apply pqp.
      intro p.
      apply q.
    - add_exm P.
      destruct exm0 as [p | Np].
      * intro pqp; assumption.
      * intro pqp.
        apply pqp.
        intro p.
        exfalso.
        apply Np; assumption.
  Qed.

  (* Quelques exercices d'anciens tests *) 
  Lemma test_1: (P->Q)->(~P->R)->(R->Q)->Q.
  Proof.
    add_exm R.
    destruct exm0 as [r | Nr].
    - intros pq Npr rq.
      apply rq; assumption.
    - add_exm P.
      destruct exm0 as [p | Np].
      * intros pq Npr rq.
        apply pq; assumption.
      * intros pq Npr rq.
        apply rq.
        apply Npr; assumption.
  Qed.

  Lemma test__2: (P \/ (Q\/R))-> (~P) -> (~R) -> (P\/Q).
  Proof.
    add_exm Q.
    destruct exm0 as [q | Nq].
    - intros pOqOr Np Nr.
      right; assumption.
    - intros pOqOr Np Nr.
      destruct pOqOr as [p | qOr].
      * left.
        assumption.
      * destruct qOr as [q | r].
        + right; assumption.
        + exfalso.
          apply Nr; assumption.
  Qed.

  Lemma test_3: (~P-> Q/\R)->(Q->~R)->P.
  Proof.
    add_exm P.
    destruct exm0 as [p | Np].
    - intros NpqAr qNr; assumption.
    - intros NpqAr qNr.
      destruct NpqAr as [Npq r].
      * assumption.
      * exfalso.
        apply qNr; assumption.
  Qed.

  Lemma test_4: (~P->Q)->(~Q\/R)->(P->R)->R.
  Proof.
    add_exm Q.
    destruct exm0 as [q | Nq].
    - intros Npq NqOr pr.
      destruct NqOr as [Nq | r].
      exfalso.
      * apply Nq; assumption.
      * assumption.
    - add_exm P.
      destruct exm0 as [p | Np].
      * intros Npq NqOr pr.
        apply pr; assumption.
      * intros Npq NqOr pr.
        exfalso.
        apply Nq.
        apply Npq; assumption.
  Qed.

  Lemma test_5: (P->Q)->(~P->~Q)->((P/\Q) \/ ~(P\/Q)).
  Proof.
    add_exm P.
    destruct exm0 as [p | Np].
    - intros pq NpNq.
      left.
      split.
      * assumption.
      * apply pq; assumption.
    - intros pq NpNq.
      right.
      intro pOq.
      apply NpNq.
      * assumption.
      * destruct pOq.
        + apply pq; assumption.
        + assumption.
  Qed.

  Lemma test_6: (P->Q)->(~P->Q)->(Q->R)->R.
  Proof.
    add_exm P.
    destruct exm0 as [p | Np].
    - intros pq Npq qr.
      apply qr.
      apply pq; assumption.
    - intros pq Npq qr.
      apply qr.
      apply Npq; assumption.
  Qed.

End LK.

Section Club_Ecossais. (* version propositionnelle *)
  Variables E R D M K: Prop.
  (* Ecossais, chaussettes Rouges, sort le Dimanche, Mari\u00e9, Kilt *)

  Hypothesis h1: ~E -> R.
  (* Tout membre non ecossais porte des chaussettes rouges *)
  Hypothesis h2: M -> ~D.
  (* Les membres maries ne sortent pas le dimanche *)
  Hypothesis h3: D <-> E.
  (* Un membre sort le dimanche si et seulement si il est ecossais *)
  Hypothesis h4: K -> E /\ M.
  (* Tout membre qui porte un kilt est ecossais et est marie *)
  Hypothesis h5: R -> K.
  (* Tout membre qui porte des chaussettes rouges porte un kilt *)
  Hypothesis h6: E -> K.
  (* Tout membre ecossais porte un kilt. *)

  Lemma personne: False. (* Le club est vide! *)
  Proof.
  Admitted.

End Club_Ecossais.  
  
(** On peut sauter cette section *)

(* Au sens strict, cette partie est hors programme; il s'agit de voir que 
   diverses hypoth\u00e8ses (toutes formul\u00e9es "au second ordre": avec des 
   quantificateurs universels sur des propositions)
   sont \u00e9quivalentes, et correspondent \u00e0 la logique classique *)
Section Second_ordre. 
  Definition PEIRCE := forall A B:Prop, ((A -> B) -> A) -> A.
  Definition DNEG := forall A, ~~A <-> A.
  Definition IMP2OR := forall A B:Prop, (A->B) <-> ~A \/ B.

  Lemma L2 : IMP2OR -> EXM.
  Proof.
    unfold IMP2OR, EXM.
    intros.
    assert (~ A \/ A).
    rewrite <- H. (* Coq "voit" qu'il suffit de prendre B=A; il va falloir prouver A->A *)
  Admitted.
  

  Lemma L3 : EXM -> DNEG.
  Proof.
    unfold DNEG , EXM.
    intros.
    (* H permet de faire un tiers exclus sur A *)
    assert (H0: A \/ ~A).
    {
      admit.
    }
  Admitted.

  Lemma L4 : PEIRCE -> DNEG.
  Proof.
    unfold DNEG , PEIRCE.
  Admitted.
  
  Lemma L5 : EXM -> PEIRCE.
  Proof.
  Admitted.

End Second_ordre.
