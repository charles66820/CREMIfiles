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
%}

%eof{
	System.out.printf("Lines: %d\nKeywords: %d\nIdentifiers: %d\nOperators: %d", 
		this.lineno, this.keywords, this.identifiers, this.operators);
%eof}


/* Définition des expressions régulières */

%%

/* Keywords */

/* Identifier */

/* Integer */

/* Float */

/* Operators (Attention d'utiliser les doubles quotes pour les caract`eres UTF8) */

/* Separators */

/* Strings */

/* Comments */

/* Attention d'utiliser [^] plut^ot que . pour l'encodage des caractères unicodes */
[^] {}
\n  {++this.lineno;}
