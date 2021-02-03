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
	int separators = 0;
	int strings = 0;
	int comments = 0;
	int nbCodeSigns = 0;
%}

%eof{
	System.out.printf(
	    "Lines: %d\nKeywords: %d\nIdentifiers: %d\nOperators: %d\nIntegers: %d\nFloats: %d\nComments: %d\nSeparators: %d\nStrings: %d\nNb code signs: %d",
		this.lineno, this.keywords, this.identifiers, this.operators, this.integers,
		this.floats, this.comments, this.separators, this.strings, this.nbCodeSigns
	);
%eof}


/* Définition des expressions régulières */

%%

/* Keywords */
"bool"|"break"|"case"|"catch"|"char"|"class"|"const"|"continue"|"default"|"delete"|"do"|"double"|"else"|"enum"|"false"|"float"|"for"|"friend"|"goto"|"if"|"inline"|"int"|"long"|"namespace"|"new"|"operator"|"private"|"protected"|"public"|"register"|"return"|"short"|"signed"|"sizeof"|"static"|"struct"|"switch"|"template"|"this"|"throw"|"true"|"try"|"typedef"|"typeid"|"typename"|"union"|"unsigned"|"using"|"virtual"|"void"|"while" {++this.keywords; this.nbCodeSigns += yytext().length();}

/* Identifier */
[a-zA-Z][a-zA-Z0-9]* {++this.identifiers; this.nbCodeSigns += yytext().length();}

/* Integer */
[-+]?[0-9]+ {++this.integers; this.nbCodeSigns += yytext().length();}

/* Float */
[-+]?([0-9]*(\.[0-9]+((e|E)(-|\+)?[0-9]+)?|[0-9]+(e|E)(-|\+)?[0-9]+)|([0-9]+\.)) {++this.floats; this.nbCodeSigns += yytext().length();}

/* Operators (Attention d'utiliser les doubles quotes pour les caract`eres UTF8) */
"++"|"+="|"+"|"--"|"-="|"-"|"*="|"*"|"/="|"/"|"%="|"%"|"<<="|"<<"|"<="|"<"|">>="|">>"|">="|">"|"&&"|"&="|"&"|"||"|"|="|"|"|"!="|"!"|"^="|"^"|"=="|"="|"~" {++this.operators; this.nbCodeSigns += yytext().length();}
// BUG: error with ** ? and < > in include ?

/* Separators */
","|";"|":"|"("|")"|"["|"]"|"{"|"}" {++this.separators; this.nbCodeSigns += yytext().length();} // BUG: error with scope qualifier ?

/* Strings */
\"[^"]*\" {++this.strings; this.nbCodeSigns += yytext().length();} // FIXME: error with \"

/* Comments */
"//"[^\n]* {++this.comments;}
"/*"([^"*"]|"*"[^"/"])*"*"?"*/" {++this.comments;}

/* Attention d'utiliser [^] plut^ot que . pour l'encodage des caractères unicodes */
\n  {++this.lineno; this.nbCodeSigns += yytext().length();}
[^] {this.nbCodeSigns += yytext().length();} // NOTE: # is not process
