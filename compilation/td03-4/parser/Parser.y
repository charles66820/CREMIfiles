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
%token TRUE FALSE IDENTIFIER INTEGER FLOAT PARB PARE
%token PLUS MULTIPLICATION MINUS DIVISION
%token LESS LESSTHAN GREATER GREATERTHAN EQUAL NOTEQUAL
%token AND OR NOT
// Grammar follows
%%
// Logique propositionnelle
T: T L
  | L;

L: '\n' { System.out.println("NewLine"); }
 | E '\n' { System.out.println("NewLine"); };

E: IDENTIFIER
 | INTEGER
 | FLOAT
 | TRUE
 | FALSE
 | PARB E PARE
 | exprArith
 | exprComp
 | exprLog;

exprArith: E PLUS E
 | E MULTIPLICATION E
 | E MINUS E
 | E DIVISION E;

exprComp: E LESS E
 | E LESSTHAN E
 | E GREATER E
 | E GREATERTHAN E
 | E EQUAL E
 | E NOTEQUAL E;

exprLog:
 | E OR E
 | E AND E
 | NOT E;

/*
// My language
L: R '\n';

// a^n b^n
R: //%empty
  | 'a' R 'b'; */

// $$ is vars for value ?
// $0, $1 ... is for select part ?
// YYACCEPT or YYABORT
// yyclearin ?
// yyerror("msg"); is for print error
%%
