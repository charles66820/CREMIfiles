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
}

%code {
  public static void main(String[] args) throws IOException {
    Lexer Lexer = new CustomLexer(new InputStreamReader(System.in));
    Parser parser = new Parser(Lexer);
    if (!parser.parse()) System.exit(1);
    System.out.println("OK");
  }
}

%locations
%token TRUE FALSE IDENTIFIER NOT OR AND PARB PARE
// Grammar follows
%%
// Logique propositionnelle
T: T L
 | T error
 | L;

L: '\n' { System.out.println("L: NewLine"); }
 | E '\n' { System.out.println("L: e NewLine"); };

E: IDENTIFIER { System.out.println("e: identifier");}
 | TRUE { System.out.println("e: true");}
 | FALSE { System.out.println("e: false");}
 | E OR E { System.out.println("e: e ∨ e");}
 | E AND E { System.out.println("e: e ∧ e");}
 | NOT E { System.out.println("e: ¬e");}
 | PARB E PARE { System.out.println("e: ( e )");};

/*
// My language
L: R '\n';

// a^n b^n
R: //%empty
  | 'a' R 'b'; */

%%
