import java.io.*;

%%

%public
%class Lexer
%type Token
%line
%column
%8bit

%{
  private Token token(Sym type, Object value) {
    return new Token(type, value, yyline, yycolumn);
  }
%}

%eof{
%eof}


/* Définition des expressions régulières */

%%

/* Keywords */
"bool"|"break"|"case"|"catch"|"char"|"class"|"const"|"continue"|"default"|"delete"|"do"|"double"|"else"|"enum"|"false"|"float"|"for"|"friend"|"goto"|"if"|"inline"|"int"|"long"|"namespace"|"new"|"operator"|"private"|"protected"|"public"|"register"|"return"|"short"|"signed"|"sizeof"|"static"|"struct"|"switch"|"template"|"this"|"throw"|"true"|"try"|"typedef"|"typeid"|"typename"|"union"|"unsigned"|"using"|"virtual"|"void"|"while" {return token(Sym.KEYWORDS, this.yytext());}

/* Identifier */
[a-zA-Z][a-zA-Z0-9]* {return token(Sym.IDENTIFIER, this.yytext());}

/* Integer */
[-+]?[0-9]+ {return token(Sym.INTEGER, this.yytext());}

/* Float */
[-+]?([0-9]*(\.[0-9]+((e|E)(-|\+)?[0-9]+)?|[0-9]+(e|E)(-|\+)?[0-9]+)|([0-9]+\.)) {return token(Sym.FLOAT, this.yytext());}

/* Operators (Attention d'utiliser les doubles quotes pour les caract`eres UTF8) */
"++"|"+="|"+"|"--"|"-="|"-"|"*="|"*"|"/="|"/"|"%="|"%"|"<<="|"<<"|"<="|"<"|">>="|">>"|">="|">"|"&&"|"&="|"&"|"||"|"|="|"|"|"!="|"!"|"^="|"^"|"=="|"="|"~" {return token(Sym.OPERATORS, this.yytext());}

/* Separators */
","|";"|":"|"("|")"|"["|"]"|"{"|"}" {return token(Sym.SEPATATORS, this.yytext());}

/* Strings */
\"(\\\"|[^\"])*\" {return token(Sym.STRINGS, this.yytext());}

/* Comments */
"//"[^\n]* {}
"/*"([^"*"]|"*"[^"/"])*"*"?"*/" {}

\n  {}
[^] {}
