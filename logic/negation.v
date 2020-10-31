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
   Restart.  (* Annule la preuve en cours, et en commence un autre *)
   unfold not.
   auto.
   (* auto est une tactique qui est capable de beaucoup, mais qu'on
      s'interdira d'utiliser dans nos preuves *)
   Qed.


    (* Remplacer les Admitted par des scripts de preuve *)
  Lemma absurde: (P -> Q) -> (P -> ~Q) -> (P -> S).
  Proof.
  Admitted.

  Lemma triple_abs: ~P -> ~~~P.
  Proof.
  Admitted.
  
  Lemma absurd' : (~P -> P) -> ~~P.
  Proof.
  Admitted.

  Definition Peirce  := ((P -> Q) -> P) -> P.

  (* On va prouver non-non-Peirce *)
  Lemma Peirce_2 : ~~ Peirce.
  Proof.
    (* Strategie: sous hypothese ~Peirce [par intro], montrer ~P, puis s'en 
       servir pour montrer Peirce, et on aura une contradiction
       entre Peirce et ~Peirce *)
    intro.
    assert (np: ~P).
  Admitted. (* \u00c0 vous de finir *)

  (* Une s\u00e9rie de s\u00e9quents \u00e0 prouver; \u00e0 chaque fois, il faut
  l'\u00e9noncer, en introduisant les hypoth\u00e8ses au moyen d'une
  sous-section... *)

  (* P->Q, R->~Q, P->R |- P->S *)

  (* ~P->~Q |- ~~Q->~~P *)

  (* P->~P |- ~P *)

  (* ~~P |- ~P->~Q *)

  (* P->~Q, R->Q |- P->~R *)

  (* ~(P->Q) |- ~Q *)
  

  (* S\u00e9quents propos\u00e9s dans le test de la semaine 42 *)

  Section TestMercredi.
    
    Hypothesis H: P->Q.

    Lemma Mercredi: ~(~Q->~P) -> R.
    Admitted.
  End TestMercredi.

  Section TestJeudi.
    Hypothesis H: ~(P->R).

    Lemma Jeudi: Q->(P->Q->R)->P.
    Admitted.
  End TestJeudi.

  Section TestVendrediMatin.
    Hypothesis H: ~(Q->R).

    Lemma VendrediMatin: (P->Q->R)->(P->Q).
    Admitted.
  End TestVendrediMatin.

  Section TestVendrediAM.
    Hypothesis H: ~~P.

    Lemma VendrediAM: Q->(P->Q->False)->P.
    Admitted.
  End TestVendrediAM.
    
End Negation.


