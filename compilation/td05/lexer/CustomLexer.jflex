import java.io.*;

%%

%public
%class CustomLexer
%line
%column

%implements Parser.Lexer

%int

%function nextToken
%yylexthrow java.io.IOException

%{
private Object lastVal;

private Position startPos = new Position(0, 0);
private Position endPos = new Position(0, 0);

@Override
public Position getStartPos() {
  return startPos;
}

@Override
public Position getEndPos() {
  return endPos;
}

@Override
public void yyerror(Parser.Location loc, String msg) {
  if (loc == null) System.err.println(msg);
  else System.err.println(loc + ": " + msg);
}

@Override
public Object getLVal() {
  return lastVal;
}

public int yylex() throws IOException {
  startPos = new Position(yyline, yycolumn);
  int ttype = nextToken();
  endPos = new Position(yyline, yycolumn);
  return ttype;
}
%}

%eof{
%eof}

Digit = [[:digit:]]
Integer = {Digit}+
Float = {Integer}"."?{Integer}?([eE][-+]?{Integer})?|"."{Integer}
Letter = [[:letter:]]
Identifier = ({Letter}|_)({Letter}|{Integer}|_)*
Space = [\t\s]

%%
"+" { return PLUS; }
- { return MINUS; }
"*" | \u00D7 { return MULTIPLICATION; } // ×
"/" { return DIVISION; }
- { return UMINUS; }

":=" { return ASSIGN; }

"(" { return PARB; }
")" { return PARE; }

{Identifier} { lastVal = yytext(); return IDENTIFIER; }
{Integer} { lastVal = Integer.parseInt(yytext()); return INTEGER; }
//{Float} { lastVal = Double.parseDouble(yytext()); return FLOAT; }

/* Strings */
\"(\\\"|[^\"])*\" {lastVal = Double.parseDouble(yytext()); return STRING; }

/* Comments */
"//"[^\n]* {}
"/*"([^"*"]|"*"[^"/"])*"*"?"*/" {}

\n { return '\n'; }
<<EOF>> { System.out.println("EOF"); return YYEOF;}
[^] {}
