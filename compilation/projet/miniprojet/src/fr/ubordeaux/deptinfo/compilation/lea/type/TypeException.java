package fr.ubordeaux.deptinfo.compilation.lea.type;

public class TypeException extends Exception {
	
	public TypeException(String msg) {
		super(msg);
	}

	public String getMessage() {
		return "--- Erreur de typage " + super.getMessage();
	}
}
