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
%token <boolean> TRUE FALSE
%token <int> INTEGER
%token <double> FLOAT
%token IDENTIFIER PARB PARE
%token PLUS MULTIPLICATION MINUS DIVISION UMINUS
%token AND OR NOT
%token LESS LESSTHAN GREATER GREATERTHAN EQUAL NOTEQUAL

%type <Integer> E
%type <Integer> exprArith
//%type <Boolean> exprComp
//%type <Boolean> exprLog

// Precedence
%left LESS LESSTHAN GREATER GREATERTHAN EQUAL NOTEQUAL
%precedence OR
%precedence AND
%left PLUS MINUS
%left MULTIPLICATION DIVISION
%nonassoc PARB PARE
%right NOT UMINUS

// Grammar follows
%%
// Logique propositionnelle
T: T L
 | T error
 | L;

L: '\n'
 | E '\n' { System.out.println($1);};

E: //IDENTIFIER
 INTEGER { System.out.println($1); $$ = $1; }
 | FLOAT { System.out.println($1); $$ = $1; }
// | TRUE { System.out.println($1); $$ = true; }
// | FALSE { System.out.println($1); $$ = false; }
 | PARB E PARE { System.out.println($2); $$ = $2; }
 | exprArith;
// | exprComp
// | exprLog;

exprArith: MINUS exprArith %prec UMINUS { System.out.println("U -"); }
 | E PLUS E { System.out.println($1 + " + "); $$ = $1 + $3; }
 | E MULTIPLICATION E { System.out.println("*"); $$ = $1 * $3; }
 | E MINUS E { System.out.println("-"); $$ = $1 - $3; }
 | E DIVISION E { System.out.println("/"); $$ = $1 / $3; };

/*exprComp: E LESS E { System.out.println("<"); $$ = $1 < $3; }
 | E LESSTHAN E { System.out.println("<="); $$ = $1 <= $3; }
 | E GREATER E { System.out.println(">"); $$ = $1 > $3; }
 | E GREATERTHAN E { System.out.println(">="); $$ = $1 >= $3; }
 | E EQUAL E { System.out.println("="); $$ = $1 == $3; }
 | E NOTEQUAL E { System.out.println("!="); $$ = $1 != $3; };*/

/*exprLog:
 E OR E { System.out.println("OR"); $$ = $1 || $3; }
 | E AND E { System.out.println("AND"); $$ = $1 && $3; }
 | NOT E { System.out.println("NOT"); $$ = !$2; };*/

/*
// My language
L: R '\n';

// a^n b^n
R: //%empty
  | 'a' R 'b'; */

// $$ is return valuer to parent
// $1, $2, $3 ... is for select right, middle andleft element
// YYACCEPT or YYABORT
// yyclearin ?
// yyerror("msg"); is for print error
%%
