package fr.ubordeaux.deptinfo.compilation.lea.environment;

// Environement:
// Outil pour enregistrer les symboles.

// Implémentation connue:
// MapEnvironment
//
public interface Environment<T> {

	// Ajoute une variable à l'environnement
	void put(String id, T value);
	
	// Retrouve une variable dans l'environnement
	T get(String id) throws EnvironmentException;

}
