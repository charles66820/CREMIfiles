%language "Java"

%define api.parser.class {Parser}
%define api.parser.public
%define lr.type canonical-lr

%define parse.error verbose

%code imports {
  import java.io.IOException;
  import java.io.InputStream;
  import java.io.InputStreamReader;
  import java.io.Reader;
  import java.io.StreamTokenizer;
}

%code {
  public static void main(String[] args) throws IOException {
    Lexer Lexer = new CustomLexer(System.in);
    Parser parser = new Parser(Lexer);
    if (!parser.parse()) {
		System.exit(1);
      }
    System.out.println("OK");
  }
}

// Grammar follows
%%

// My language
L: R '\n';

// a^n b^n
R: //%empty
  | 'a' R 'b' { /* Code */ };

// $$ is vars for value ?
// $0, $1 ... is for select part ?
// YYACCEPT or YYABORT
// yyclearin ?
// yyerror("msg"); is for print error
%%
