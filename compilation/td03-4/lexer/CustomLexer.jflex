import java.io.*;

%%

%public
%class CustomLexer
%line
%column
%8bit

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
  return this.startPos;
}

@Override
public Position getEndPos() {
  return this.endPos;
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

OR { return OR; } // \u2228 ∨
AND { return AND; } // \u2227 ∧
NOT { return NOT; } // \u00AC ¬

"(" {
  this.startPos = new Position(yyline, yycolumn);
  this.endPos = new Position(yyline, yycolumn);
  return PARB;
}
")" {
  this.endPos = new Position(yyline, yycolumn);
  if (this.startPos == null) this.startPos = new Position(yyline, yycolumn);
  return PARE;
}

{Identifier} { lastVal = yytext(); return IDENTIFIER; }

\n { return '\n'; }
<<EOF>> { System.out.println("EOF"); return YYEOF;}
[^] {}
