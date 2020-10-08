(* Raccourcis clavier de coqide *)
(* CTRL-fl\u00e8che bas: avancer d'une commande *)
(* CTRL-fl\u00e8che haut: revenir en arri\u00e8re d'une commande *)
(* CTRL-fl\u00e8che droit: avancer ou revenir en arri\u00e8re jusqu'au curseur *) 

(** premiers pas en Coq *)

(* Logique minimale "pure": pas de n\u00e9gation/contradiction *)

Section Declarations.

  Variables P Q R S T : Prop.

  (* Une Section peut contenir d'autres [sous-]Sections. C'est le bon endroit
     pour definir des hypotheses (et donc prouver des sequents avec hypotheses).

     Dans ce fichier, la section "Declarations" va jusqu'au bout \u2620\ufe0f *)
  
  Lemma imp_dist: (P -> Q -> R) -> (P -> Q) -> P -> R.
  Proof.
    intro H.
    intro H0.
    intro H1.
    apply H.
    - assumption.
    - apply H0.
      assumption.
  Qed.

  (* Explication des tactiques utilis\u00e9es: *)
  (* intro: utilisation de la regle d'introduction pour changer de but *)
  (* apply: utilisation d'une implication qui a la bonne conclusion
     (il va falloir prouver ses hypotheses) *)
  (* Note: on ne peut faire "apply" que sur une propri\u00e9t\u00e9 d\u00e9j\u00e0 prouv\u00e9e,
     contrairement au modus ponens des preuves en arbres *)
  (* assumption: utilisation de la regle d'hypothese *)

  Check imp_dist.  (* On voit la formule prouv\u00e9e *)
  Print imp_dist.  (* Pour voir le "terme de preuve" calcul\u00e9 *)

  (* exemple de preuve d'un sequent avec hypoth\u00e8ses *)

  Section S1.
    Hypothesis H : P -> Q.
    Hypothesis H0 : P -> Q -> R.

    Lemma L2 : P -> R.
    (* le sequent est: P->Q, P->Q->R |- P->R *)
    Proof.
      intro p.
      apply H0.
      - assumption.
      - apply H.
        assumption.
    Qed.

    Check L2. (* Les hypoth\u00e8se font partie de la section *)
  End S1.

  (* Quand on ferme la section, ses hypotheses sont "exportees" sous la
     forme d'hypotheses supplementaires pour L2                         *)
  Check L2.

  
  Section About_cuts.
    Hypothesis H : P -> Q.
    Hypothesis H0 : P -> Q -> R.
    Hypothesis H1 : Q -> R -> S.
    Hypothesis H2 : Q -> R -> S -> T.

    (* preuve sans lemme (coupure) *)
    Lemma L3 : P -> T.
    (* Quel est le s\u00e9quent qu'on s'appr\u00eate \u00e0 prouver? *)

    (* Faites-en une preuve papier AVANT de continuer *)
    Proof.
      intro p.
      apply H2.
      apply H.
      assumption.
      apply H0.
      assumption.
      apply H.
      assumption.
      apply H1.
      apply H; assumption.
      apply H0.
      assumption.
      apply H;assumption.
    Qed.
    (* R\u00e9\u00e9crivez le script ci-dessus en introduisant des tirets 
       (-, *, +) \u00e0 chaque fois qu'une tactique engendre plus d'un 
       sous-but *)
    
    (* preuve avec coupures: on prouve Q et R une seule fois chacun,
       puis on les utilise *)

     Lemma L'3 : P -> T.
     Proof.
       intro p.
       assert (q: Q). { 
         apply H; assumption.
         }   
       assert (r: R). {
         apply H0.
         - assumption.
         - assumption.
          }
       assert (s : S). {
        apply H1; assumption.
       }
       apply H2; assumption.
     Qed.

     (* assert: permet d'ouvrir une nouvelle sous-preuve, *)
     (* dans laquelle on se d\u00e9finit un nouveau but; c'est *)
     (* une coupure. Les accolades sont optionnelles mais *)
     (* facilitent la lecture humaine                     *)
     
     Check L'3.

(* remarquez la diff\u00e9rence entre les termes de preuves avec coupure et sans coupure. *)
     Print L'3.
     Print L3.

  End About_cuts.


 (* Exercices 

    Reprendre les exemples vus en TD, en utilisant les tactiques 
    assumption, apply, assert et intro/intros.

    Remplacer chaque commande Admitted par un script correct de preuve,
    suivi de la commande Qed.

  *)

  Lemma IdP : P -> P.
  Proof.
  Admitted.

  Lemma IdPP : (P -> P) -> P -> P.
  Proof.
  Admitted.

  (* sans regarder le fichier de demo, c'est de la triche *)
  Lemma imp_trans : (P -> Q) -> (Q -> R) -> P -> R.
  Proof.
  Admitted.

  Section proof_of_hyp_switch.
    Hypothesis H : P -> Q -> R.
    Lemma hyp_switch : Q -> P -> R.
    Proof.
    Admitted. 

  End proof_of_hyp_switch.

  Check hyp_switch.

  Section proof_of_add_hypothesis.
    Hypothesis H : P -> R.

    Lemma add_hypothesis : P -> Q -> R.
    Proof.
    Admitted.

  End proof_of_add_hypothesis.

  (* prouver le sequent (P -> P -> Q) |- P -> Q  
     (il faut l'\u00e9noncer, et faire la preuve) 
      *)
  Section proof_of_remove_dup_hypothesis.

  End proof_of_remove_dup_hypothesis.

  (* m\u00eame exercice avec le s\u00e9quent P->Q |- P->P->Q *)
  Section proof_of_dup_hypothesis.

  End proof_of_dup_hypothesis.

  (* meme exercice avec 
     P -> Q , P -> R , Q -> R -> T |- P -> T  
   *)
  Section proof_of_distrib_impl.

  End proof_of_distrib_impl.

  (* m\u00eame exercice, avec 
     P->Q, Q->R, (P->R)->T->Q, (P->R)->T |- Q   
     (ex. 9 de la feuille TD2)
   *)
  Section proof_of_ex9.

  End proof_of_ex9.
  
  (* exercice 12 de la feuille 1 *)
  Section Proof_of_weak_Peirce.

    Hypothesis H: (((P->Q)->P)->P)->Q.
    Lemma weak_Peirce : Q.
    Proof.
    Admitted.

  End Proof_of_weak_Peirce.
  Check weak_Peirce.
  Print weak_Peirce. (* Pas facile \u00e0 d\u00e9chiffrer *)
End Declarations.

