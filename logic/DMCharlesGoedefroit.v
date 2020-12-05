Section section3_1.
Variables A B : Prop.

Theorem section3_1 : ~(A -> B) <-> ~~A /\ ~B.
Proof.
  split.
  - intro Nab.
    split.
    * intro Na.
      apply Nab.
      intro a.
      exfalso.
      apply Na; assumption.
    * intro b.
      apply Nab.
      intro a; assumption.
  - intros NNaANb ab.
    destruct NNaANb as [NNa Nb].
    apply NNa.
    intro a.
    apply Nb.
    apply ab; assumption.
Qed.

End section3_1.

Section section3_2.
Variables A B : Prop.

Theorem section3_2 : ~(A /\ B) <-> (~~A -> ~B).
Proof.
  split.
  - intros NaAb NNa b.
    apply NNa.
    intro a.
    apply NaAb.
    split; assumption.
  - intros NNaNb aAb.
    destruct aAb as [a b].
    apply NNaNb.
    * intro Na.
      apply Na; assumption.
    * assumption.
Qed.

End section3_2.

Section section3_3.
Variables A B : Prop.

Theorem section3_3 : ~(A \/ B) <-> ~A /\ ~B.
Proof.
  split.
  - intro NaOb.
    split.
    * intro.
      apply NaOb.
      left; assumption.
    * intro b.
      apply NaOb.
      right; assumption.
  - intros NaANb aOb.
    destruct NaANb as [Na Nb].
    destruct aOb as [a | b].
    * apply Na; assumption.
    * apply Nb; assumption.
Qed.

End section3_3.

(* Preuve que chacune des traductions n\u00e9gatives est une r\u00e8gle d\u00e9riv\u00e9e de la logique intuitionniste *)
 
Section ModusPonensRule.
Variables A B : Prop.

Hypothesis H1 : ~~(A -> B).
Hypothesis H2 : ~~A.
Theorem ModusPonensRule : ~~B.
Proof.
  intro Nb.
  apply H2.
  intro a.
  apply H1.
  intro ab.
  apply Nb.
  apply ab; assumption.
Qed.

End ModusPonensRule.

Section IntroductionRule.
Variables A B : Prop.

Hypothesis H1 : ~~A -> ~~B.
Theorem IntroductionRule : ~~(A -> B).
Proof.
  intro Nab.
  apply H1.
  - intro Na.
    apply Nab.
    intro a.
    exfalso.
    apply Na; assumption.
  - intro b.
    apply Nab.
    intro a; assumption.
Qed.

End IntroductionRule.

Section ExfalsoRule.
Variables A : Prop.

Hypothesis H1 : ~~False.
Theorem ExfalsoRule : ~~A.
Proof.
  intro Na.
  apply H1.
  intro F; assumption.
Qed.

End ExfalsoRule.

Section ConjunctionIntroductionRule.
Variables A B : Prop.

Hypothesis H1 : ~~A.
Hypothesis H2 : ~~B.
Theorem ConjunctionIntroductionRule : ~~(A /\ B).
Proof.
  intro NaAb.
  apply H1.
  intro a.
  apply H2.
  intro b.
  apply NaAb.
  split; assumption.
Qed.

End ConjunctionIntroductionRule.

Section ConjunctionEliminationEditLeftRule.
Variables A B : Prop.

Hypothesis H1 : ~~(A /\ B).
Theorem ConjunctionEliminationEditLeftRule : ~~A.
Proof.
  intro Na.
  apply H1.
  intro aAb.
  destruct aAb as [a b].
  apply Na; assumption.
Qed.

End ConjunctionEliminationEditLeftRule.

Section ConjunctionEliminationEditRightRule.
Variables A B : Prop.

Hypothesis H1 : ~~(A /\ B).
Theorem ConjunctionEliminationEditRightRule : ~~B.
Proof.
  intro Nb.
  apply H1.
  intro aAb.
  destruct aAb as [a b].
  apply Nb; assumption.
Qed.

End ConjunctionEliminationEditRightRule.

Section DisjunctionIntroductionLeftRule.
Variables A B : Prop.

Hypothesis H1 : ~~A.
Theorem DisjunctionIntroductionLeftRule : ~~(A \/ B).
Proof.
  intro NaOb.
  apply H1.
  intro a.
  apply NaOb.
  left; assumption.
Qed.

End DisjunctionIntroductionLeftRule.

Section DisjunctionIntroductionRightRule.
Variables A B : Prop.

Hypothesis H1 : ~~B.
Theorem DisjunctionIntroductionRightRule : ~~(A \/ B).
Proof.
  intro NaOb.
  apply H1.
  intro b.
  apply NaOb.
  right; assumption.
Qed.

End DisjunctionIntroductionRightRule.

Section DisjunctionEliminationRule.
Variables A B C : Prop.

Hypothesis H1 : ~~(A \/ B).
Hypothesis H2 : ~~A -> ~~C.
Hypothesis H3 : ~~B -> ~~C.
Theorem DisjunctionEliminationRule : ~~C.
Proof.
  intro Nc.
  apply H1.
  intro aOb.
  apply H2.
  - intro Na.
    destruct aOb as [a | b].
    * apply Na; assumption.
    * apply H3.
      + intro Nb.
        apply Nb; assumption.
      + assumption.
  - assumption.
Qed.

End DisjunctionEliminationRule.

Section ExmE.
Variables A B : Prop.

Hypothesis H1 : ~~A -> ~~B.
Hypothesis H2 : ~~~A -> ~~B.
Theorem ExmE : ~~B.
Proof.
  intro Nb.
  apply H2.
  - intro NNa.
    apply H1.
    * intro Na.
      apply NNa; assumption.
    * assumption.
  - assumption.
Qed.

End ExmE.
