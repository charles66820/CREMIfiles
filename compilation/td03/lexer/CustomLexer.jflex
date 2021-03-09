import java.io.*;

%%

%public
%class CustomLexer
%8bit

%implements Parser.Lexer

%int

%{
//private StreamTokenizer streamTokenizer;
public CustomLexer (InputStream inputStream) {
  this(new InputStreamReader(inputStream));
}

@Override
public void yyerror(String s) {
  System.err.println(s);
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

"(" { return PARB; }
")" { return PARE; }

/*
a { return 'a'; }
b { return 'b'; }
*/

\n { System.out.println("NewLine"); return '\n'; }
<<EOF>> { System.out.println("EOF"); return YYEOF;}
[^] {}
