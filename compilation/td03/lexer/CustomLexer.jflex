import java.io.*;

%%

%public
%class CustomLexer
%line
%column
%8bit

%implements Parser.Lexer

%int

%{
private Position startPos;
private Position endPos;

public CustomLexer (InputStream inputStream) {
  this(new InputStreamReader(inputStream));
}

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
  System.err.println(msg + " at " + loc);
}

@Override
public Object getLVal() {
  return null;
}
%}

%eof{
%eof}

%%
1 { return TRUE; }
0 { return FALSE; }

p { return P; }

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

/*
a { return 'a'; }
b { return 'b'; }
*/

\n { System.out.println("NewLine"); return '\n'; }
<<EOF>> { System.out.println("EOF"); return YYEOF;}
[^] {}
