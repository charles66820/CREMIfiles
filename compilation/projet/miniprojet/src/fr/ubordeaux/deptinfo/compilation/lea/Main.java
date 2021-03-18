package fr.ubordeaux.deptinfo.compilation.lea;

import java.io.FileReader;
import java.io.IOException;

import fr.ubordeaux.deptinfo.compilation.lea.environment.EnvironmentException;
import fr.ubordeaux.deptinfo.compilation.lea.parser.Parser;
import fr.ubordeaux.deptinfo.compilation.lea.parser.ParserLexer;
import fr.ubordeaux.deptinfo.compilation.lea.type.TypeException;

public class Main {

	public static void main(String[] args) {
		FileReader inputFile = null;
	    System.out.println("*** begin compilation");
		try {
			inputFile = new FileReader(args[0]);
		}
		catch (Exception e){
		    System.err.println("invalid file");
		}
		try {
			Parser.Lexer Lexer = new ParserLexer(inputFile);
			Parser parser = new Parser(Lexer);
			parser.parse();
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
		catch (EnvironmentException e) {
			System.err.println(e.getMessage());
		}
		catch (TypeException e) {
			System.err.println(e.getMessage());
		}
	    finally {
	    		System.out.println("*** end compilation");
	    }
	}

}
