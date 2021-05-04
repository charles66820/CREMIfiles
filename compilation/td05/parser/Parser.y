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

// operators
%token ASSIGN ":="
  PLUS "+"
  MINUS "-"
  MULTIPLICATION "*"
  DIVISION "/"
  UMINUS;

//keywords
%token
  PARB "("
  PARE ")"
  IDENTIFIER;

// constants
%token<String>
	//IDENTIFIER
	STRING;

%token<Integer>
	INTEGER;

%token<Float>
	FLOAT;

//%type <String> E
%type <Integer> expr

// Precedence
%left PLUS MINUS
%left MULTIPLICATION DIVISION
%nonassoc PARB PARE
%right UMINUS

// Grammar follows
%%
/*T: T L
 | T error
 | L;

L: '\n'
 | E '\n' { System.out.println($1);};*/

axiom: affs expr;

affs:
  affs aff
  | aff;

aff: lhs ":=" expr ';';

expr:
  IDENTIFIER
  | INTEGER { $$ = $1; }
  //| FLOAT
  | STRING
  | expr '+' expr { System.out.println($1 + " + " + $3); $$ = $1 + $3; }
  | expr '-' expr { System.out.println($1 + " - " + $3); $$ = $1 - $3; }
  | '-' expr %prec UMINUS { System.out.println("-" + $2); }
  | expr '*' expr { System.out.println($1 + " * " + $3); $$ = $1 * $3; }
  | expr '/' expr { System.out.println($1 + " / " + $3); $$ = $1 / $3; }
  | '(' expr ')' { $$ = $2; };

lhs: IDENTIFIER;

// { System.out.println($1); $$ = String.valueOf($1); }

%%
