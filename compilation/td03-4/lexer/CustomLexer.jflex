import java.io.*;

%%

%public
%class CustomLexer
%line
%column

%implements Parser.Lexer

%int

%function yylex
%yylexthrow java.io.IOException

%{
private Object lastVal;

private Position startPos;
private Position endPos;

@Override
public Position getStartPos() {
  return new Position(yyline, yycolumn);
}

@Override
public Position getEndPos() {
  return new Position(yyline, yycolumn);
}

@Override
public void yyerror(Parser.Location loc, String msg) {
  System.err.println(msg + " start at " + loc.begin + " end at " + loc.end);
}

@Override
public Object getLVal() {
  return lastVal;
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
1 { return TRUE; }
0 { return FALSE; }

OR | \u2228 { return OR; } // ∨
AND | \u2227 { return AND; } // ∧
NOT | \u00AC { return NOT; } // ¬

"(" { return PARB; }
")" { return PARE; }

{Identifier} { lastVal = yytext(); return IDENTIFIER; }

\n { return '\n'; }
<<EOF>> { System.out.println("EOF"); return YYEOF;}
[^] {}
