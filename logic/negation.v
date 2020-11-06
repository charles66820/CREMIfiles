Section Negation.
  Variables P Q R S T: Prop.

  (* unfold not: expansion de la n\u00e9gation dans le but *)
  (* unfold not in X: expansion de la n\u00e9gation dans l'hypoth\u00e8se X *)
  (* exfalso: transforme le but courant en False; c'est l'\u00e9quivalent
     de la r\u00e8gle d'\u00e9limination de la contradiction *)

  (* Executez cette preuve en essayant de comprendre le sens de chacune des nouvelles tactiques utilis\u00e9es. *)
  Lemma absurde_exemple: P -> ~P -> S.
  Proof.
    intros p np.
    unfold not in np.
    exfalso.
    apply np.
    assumption.
  Qed.

  Lemma triple_neg_e : ~~~P -> ~P.
  Proof.
    intro H.
    intro H0.
    apply H.
    intro H1.
    apply H1; assumption.
    Restart.
    unfold not.
    intro H.
    intro H0.
    apply H.
    intro H1.
    apply H1.
    assumption.
    Restart.  (* Annule la preuve en cours, et en commence un autre *)
    unfold not.
    auto.
    (* auto est une tactique qui est capable de beaucoup, mais qu'on
      s'interdira d'utiliser dans nos preuves *)
  Qed.

  (* Remplacer les Admitted par des scripts de preuve *)
  Lemma absurde: (P -> Q) -> (P -> ~Q) -> (P -> S).
  Proof.
    intros pq pNq p.
    assert (q:Q).
    apply pq.
    assumption.
    assert (Nq:~Q).
    apply pNq.
    assumption.
    exfalso.
    apply Nq.
    assumption.
    Restart.
    intros pq pNq p.
    unfold not in pNq.
    exfalso.
    apply pNq.
    assumption.
    apply pq.
    assumption.
  Qed.

  Lemma triple_abs: ~P -> ~~~P.
  Proof.
    unfold not.
    intros Np NNp.
    apply NNp.
    apply Np.
  Qed.

  Lemma absurd' : (~P -> P) -> ~~P.
  Proof.
    unfold not.
    intros Npp Np.
    apply Np.
    apply Npp. (* Tautologie *)
    apply Np.
  Qed.

  Definition Peirce  := ((P -> Q) -> P) -> P.

  (* On va prouver non-non-Peirce *)
  Lemma Peirce_2 : ~~ Peirce.
  Proof.
    (* Strategie: sous hypothese ~Peirce [par intro], montrer ~P, puis s'en 
       servir pour montrer Peirce, et on aura une contradiction
       entre Peirce et ~Peirce *)
    intro.
    assert (np: ~P).
    - unfold not.
     intro p.
     apply H.
     intro peirce.
     assert (nq: (P->Q)->P).
     intro pq.
     assumption.
     assumption.
    - assert (pirce: Peirce).
     intro peirce.
     apply peirce.
     intro p.
     exfalso.
     apply np.
     assumption.
    apply H.
    assumption.
    Qed.

  (* Une s\u00e9rie de s\u00e9quents \u00e0 prouver; \u00e0 chaque fois, il faut
  l'\u00e9noncer, en introduisant les hypoth\u00e8ses au moyen d'une
  sous-section... *)

  (* P->Q, R->~Q, P->R |- P->S *)
  Section s1.
    Hypothesis pq: P->Q.
    Hypothesis rq: R->~Q.
    Hypothesis pr: P->R.

    Lemma s1: P->S.
    Proof.
      intro.
      exfalso.
      apply rq.
      - apply pr.
        assumption.
      - apply pq.
        assumption.
    Qed.
  End s1.


  (* ~P->~Q |- ~~Q->~~P *)
  Section s2.
    Hypothesis NpNq: ~P->~Q.

    Lemma s2: ~~Q->~~P.
    Proof.
      intro NNq.
      intro Np.
      unfold not in NNq.
      apply NNq.
      unfold not in NpNq.
      apply NpNq.
      assumption.
    Qed.
  End s2.

  (* P->~P |- ~P *)
  Section s3.
    Hypothesis pNp: P->~P.

    Lemma s3: ~P.
    Proof.
      intro p.
      apply pNp.
      assumption.
      assumption.
      Restart.
      intro p.
      apply pNp; assumption.
    Qed.
  End s3.

  (* ~~P |- ~P->~Q *)
  Section s4.
    Hypothesis NNp: ~~P.

    Lemma s4: ~P->~Q.
    Proof.
      intro.
      exfalso.
      apply NNp.
      assumption.
    Qed.
  End s4.

  (* P->~Q, R->Q |- P->~R *)
  Section s5.
    Hypothesis pNq: P->~Q.
    Hypothesis rq: R->Q.

    Lemma s5: P->~R.
    Proof.
      intros p r.
      apply pNq.
      - assumption.
      - apply rq; assumption.
    Qed.
  End s5.

  (* ~(P->Q) |- ~Q *)
  Section s6.
    Hypothesis Npq: ~(P->Q).

    Lemma s6: ~Q.
    Proof.
      intro q.
      apply Npq.
      intro p; assumption.
    Qed.
  End s6.

  (* S\u00e9quents propos\u00e9s dans le test de la semaine 42 *)

  Section TestMercredi.
    
    Hypothesis H: P->Q.

    Lemma Mercredi: ~(~Q->~P) -> R.
    Proof.
      intro FNpNq.
      exfalso.
      apply FNpNq.
      intro Nq.
      intro p.
      apply Nq.
      apply H.
      assumption.
    Qed.
  End TestMercredi.

  Section TestJeudi.
    Hypothesis H: ~(P->R).

    Lemma Jeudi: Q->(P->Q->R)->P.
    Proof.
      intros q pqr.
      exfalso.
      apply H.
      intro p.
      apply pqr; assumption.
    Qed.
  End TestJeudi.

  Section TestVendrediMatin.
    Hypothesis H: ~(Q->R).

    Lemma VendrediMatin: (P->Q->R)->(P->Q).
    Proof.
      intros pqr p.
      exfalso.
      apply H.
      apply pqr; assumption.
    Qed.
  End TestVendrediMatin.

  Section TestVendrediAM.
    Hypothesis H: ~~P.

    Lemma VendrediAM: Q->(P->Q->False)->P.
    Proof.
      intros q pqF.
      exfalso.
      apply H.
      intro p.
      apply pqF; assumption.
    Qed.
  End TestVendrediAM.
    
End Negation.


