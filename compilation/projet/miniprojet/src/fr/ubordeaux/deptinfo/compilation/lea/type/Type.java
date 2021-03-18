package fr.ubordeaux.deptinfo.compilation.lea.type;

// Type:
// Représentation d'un type simple ou d'un type complexe

//Pour l'instant cet arbre contient:
//- Le TType
//- Sa composition (left, right)

// Pour vérifier un type:
// assertEqual vérifie que deux types sont égaux ou
// qu'un type est égal à un type donné

// implémentation connue
// TypeExpr
//

public interface Type {

	TType getTType();
	void assertEqual(Type t) throws TypeException;
	int getSize();
	int getOffset();
	Type getLeft();
	Type getRight();
}
