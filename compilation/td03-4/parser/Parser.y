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
//%token <double> FLOAT
%token PARB PARE //IDENTIFIER
%token PLUS MULTIPLICATION MINUS DIVISION UMINUS
%token AND OR NOT
%token LESS LESSTHAN GREATER GREATERTHAN EQUAL NOTEQUAL

%type <String> E
%type <Integer> FOC FOA exprArith
%type <Boolean> FOL exprComp exprLog

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

E: exprArith { System.out.println($1); $$ = String.valueOf($1); }
 | exprComp { System.out.println($1); $$ = Boolean.toString($1); }
 | exprLog { System.out.println($1); $$ = Boolean.toString($1); };

exprArith: MINUS exprArith %prec UMINUS { System.out.println("-" + $2); }
 | FOA PLUS FOA { System.out.println($1 + " + " + $3); $$ = $1 + $3; }
 | FOA MULTIPLICATION FOA { System.out.println($1 + " * " + $3); $$ = $1 * $3; }
 | FOA MINUS FOA { System.out.println($1 + " - " + $3); $$ = $1 - $3; }
 | FOA DIVISION FOA { System.out.println($1 + " / " + $3); $$ = $1 / $3; };

exprComp: FOC LESS FOC { System.out.println($1 + " < " + $3); $$ = $1 < $3; }
 | FOC LESSTHAN FOC { System.out.println($1 + " <= " + $3); $$ = $1 <= $3; }
 | FOC GREATER FOC { System.out.println($1 + " > " + $3); $$ = $1 > $3; }
 | FOC GREATERTHAN FOC { System.out.println($1 + " >= " + $3); $$ = $1 >= $3; }
 | FOC EQUAL FOC { System.out.println($1 + " = " + $3); $$ = $1 == $3; }
 | FOC NOTEQUAL FOC { System.out.println($1 + " != " + $3); $$ = $1 != $3; };

exprLog: FOL OR FOL { System.out.println($1 + " OR " + $3); $$ = $1 || $3; }
 | FOL AND FOL { System.out.println($1 + " AND " + $3); $$ = $1 && $3; }
 | NOT FOL { System.out.println("NOT " + $2); $$ = !$2; };

// Final Operands Arithmetic operations
FOA: exprArith { $$ = $1; }
 | INTEGER { $$ = $1; }
 | PARB FOA PARE { $$ = $2; };

// Final Operands Comparison operations
FOC: exprArith { $$ = $1; }
 | INTEGER { $$ = $1; }
 | PARB FOC PARE { $$ = $2; };

// Final Operands Logical operators
FOL: exprLog { $$ = $1; }
 | exprComp { $$ = $1; }
 | TRUE { $$ = true; }
 | FALSE { $$ = false; }
 | PARB FOL PARE { $$ = $2; };

/*
// My language
L: R '\n';

// a^n b^n
R: //%empty
  | 'a' R 'b'; */

// $$ is return valuer to parent
// $1, $2, $3 ... is for select right, middle andleft element
// YYACCEPT or YYABORT
%%
