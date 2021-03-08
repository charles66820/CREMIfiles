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
"toto" { System.out.println("Toto"); }

a { return 'a'; }
b { return 'b'; }

\n { System.out.println("NewLine"); return '\n'; }
<<EOF>> { System.out.println("EOF"); return YYEOF;}
[^] {}
