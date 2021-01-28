import java.io.*;

%%

%public
%class Lexer
%standalone
%8bit

%{
	int lineno = 0;
	int keywords = 0;
	int identifiers = 0;
	int operators = 0;

	int integers = 0;
	int floats = 0;
	int comments = 0;
	boolean DEBUG = true;
%}

%eof{
	System.out.printf("Lines: %d\nKeywords: %d\nIdentifiers: %d\nOperators: %d",
		this.lineno, this.keywords, this.identifiers, this.operators);
	if (DEBUG)
	    System.out.printf("\nIntegers: %d\nFloats: %d\nComments: %d",
        		this.integers, this.floats, this.comments);
%eof}


/* Définition des expressions régulières */

%%

/* Keywords */
"if"|"else"|"while" {++this.keywords;}

/* Identifier */
[a-zA-Z][a-zA-Z0-9]* {++this.identifiers;}

/* Integer */
[0-9]* {++this.integers;}

/* Float */
[0-9]*("."[0-9]+((e|E)(-|"+")[0-9]*)?|(e|E)(-|"+")[0-9]+) {++this.floats;}

/* Operators (Attention d'utiliser les doubles quotes pour les caract`eres UTF8) */

/* Separators */

/* Strings */

/* Comments */
"//"[^\n]* {++this.comments;}
"/*"([^"*"]|"*"[^"/"])*"*"?"*/" {++this.comments;}

/* Attention d'utiliser [^] plut^ot que . pour l'encodage des caractères unicodes */
\n  {++this.lineno;}
[^] {}
