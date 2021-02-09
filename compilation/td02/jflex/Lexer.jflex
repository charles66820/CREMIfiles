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
  private Token token(Sym type) {
    return token(type, null);
  }
%}

%eof{
%eof}


/* Définition des expressions régulières */

%%

/* Keywords */
"bool" {return token(Sym.BOOL);}
"break" {return token(Sym.BREAK);}
"case" {return token(Sym.CASE);}
"catch" {return token(Sym.CATCH);}
"char" {return token(Sym.CHAR);}
"class" {return token(Sym.CLASS);}
"const" {return token(Sym.CONST);}
"continue" {return token(Sym.CONTINUE);}
"default" {return token(Sym.DEFAULT);}
"delete" {return token(Sym.DELETE);}
"do" {return token(Sym.DO);}
"double" {return token(Sym.DOUBLE);}
"else" {return token(Sym.ELSE);}
"enum" {return token(Sym.ENUM);}
"false" {return token(Sym.FALSE);}
"float" {return token(Sym.FLOAT);}
"for" {return token(Sym.FOR);}
"friend" {return token(Sym.FRIEND);}
"goto" {return token(Sym.GOTO);}
"if" {return token(Sym.IF);}
"inline" {return token(Sym.INLINE);}
"int" {return token(Sym.INT);}
"long" {return token(Sym.LONG);}
"namespace" {return token(Sym.NAMESPACE);}
"new" {return token(Sym.NEW);}
"operator" {return token(Sym.OPERATOR);}
"private" {return token(Sym.PRIVATE);}
"protected" {return token(Sym.PROTECTED);}
"public" {return token(Sym.PUBLIC);}
"register" {return token(Sym.REGISTER);}
"return" {return token(Sym.RETURN);}
"short" {return token(Sym.SHORT);}
"signed" {return token(Sym.SIGNED);}
"sizeof" {return token(Sym.SIZEOF);}
"static" {return token(Sym.STATIC);}
"struct" {return token(Sym.STRUCT);}
"switch" {return token(Sym.SWITCH);}
"template" {return token(Sym.TEMPLATE);}
"this" {return token(Sym.THIS);}
"throw" {return token(Sym.THROW);}
"true" {return token(Sym.TRUE);}
"try" {return token(Sym.TRY);}
"typedef" {return token(Sym.TYPEDEF);}
"typeid" {return token(Sym.TYPEID);}
"typename" {return token(Sym.TYPENAME);}
"union" {return token(Sym.UNION);}
"unsigned" {return token(Sym.UNSIGNED);}
"using" {return token(Sym.USING);}
"virtual" {return token(Sym.VIRTUAL);}
"void" {return token(Sym.VOID);}
"while" {return token(Sym.WHILE);}

/* Identifier */
[a-zA-Z][a-zA-Z0-9]* {return token(Sym.IDENTIFIER, this.yytext());}

/* Integer */
[-+]?[0-9]+ {return token(Sym.INTEGER, this.yytext());}

/* Float */
[-+]?([0-9]*(\.[0-9]+((e|E)(-|\+)?[0-9]+)?|[0-9]+(e|E)(-|\+)?[0-9]+)|([0-9]+\.)) {return token(Sym.NUMBER, this.yytext());}

/* Operators (Attention d'utiliser les doubles quotes pour les caract`eres UTF8) */
"++" {return token(Sym.INCREMENT);}
"+=" {return token(Sym.PLUS_EQ);}
"+" {return token(Sym.PLUS);}
"--" {return token(Sym.DECREMENT);}
"-=" {return token(Sym.MINUS_EQ);}
"-" {return token(Sym.MINUS);}
"*=" {return token(Sym.MULTIPLICATION_EQ);}
"*" {return token(Sym.MULTIPLICATION);}
"/=" {return token(Sym.DIVISION_EQ);}
"/" {return token(Sym.DIVISION);}
"%=" {return token(Sym.MODULO_EQ);}
"%" {return token(Sym.MODULO);}
"<<=" {return token(Sym.SHL_EQ);}
"<<" {return token(Sym.SHL);}
"<=" {return token(Sym.LESS_EQ);}
"<" {return token(Sym.LESS);}
">>=" {return token(Sym.SHR_EQ);}
">>" {return token(Sym.SHR);}
">=" {return token(Sym.OGREATER_EQ);}
">" {return token(Sym.GREATER);}
"&&" {return token(Sym.AND);}
"&=" {return token(Sym.AND_EQ);}
"&" {return token(Sym.BiTAND);}
"||" {return token(Sym.OR);}
"|=" {return token(Sym.OR_EQ);}
"|" {return token(Sym.BITOR);}
"!=" {return token(Sym.NOT_EQ);}
"!" {return token(Sym.NOT);}
"^=" {return token(Sym.XOR_EQ);}
"^" {return token(Sym.XOR);}
"==" {return token(Sym.EQ);}
"=" {return token(Sym.ASSIGNMENT);}
"~" {return token(Sym.COMPL);}

/* Separators */
"," {return token(Sym.COMMA);}
";" {return token(Sym.SEMICOLON);}
"::" {return token(Sym.SCOPE_RESOLUTION);}
"(" {return token(Sym.LPAR);}
")" {return token(Sym.RPAR);}
"[" {return token(Sym.LBRA);}
"]" {return token(Sym.RBRA);}
"{" {return token(Sym.LBRACE);}
"}" {return token(Sym.RBRACE);}

/* Ternary conditional */
"?" {return token(Sym.QUESTION);}
":" {return token(Sym.COLON);}

/* Strings */
\"(\\\"|[^\"])*\" {return token(Sym.STRING, this.yytext());}

/* Comments */
"//"[^\n]* {}
"/*"([^"*"]|"*"[^"/"])*"*"?"*/" {}

\n  {}
[^] {}
