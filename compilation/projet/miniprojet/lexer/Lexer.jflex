package fr.ubordeaux.deptinfo.compilation.lea.parser;

import java.io.*;

%%

%public
%class ParserLexer
%implements Parser.Lexer
%int
%function nextToken
%yylexthrow java.io.IOException
%line
%column

%{
  private final boolean DEBUGGING = true;

  private Position startPos = new Position(0, 0);
  private Position endPos = new Position(0, 0);
  private Object lVal;

    /**
     * Method to retrieve the beginning position of the last scanned token.
     * @return the position at which the last scanned token starts.
     */
    public Position getStartPos(){
 	   return startPos;
    }

    /**
     * Method to retrieve the ending position of the last scanned token.
     * @return the first position beyond the last scanned token.
     */
    public Position getEndPos(){
    	   return endPos;
    }

    /**
     * Method to retrieve the semantic value of the last scanned token.
     * @return the semantic value of the last scanned token.
     */
    public Object getLVal(){
        return lVal;
    }

  /**
   * Fetch the next token.  Called yylex in pull parsers.
   */
  public int yylex() throws IOException {
    startPos = new Position(yyline, yycolumn);
    int ttype = nextToken();
    endPos = new Position(yyline, yycolumn);
    return ttype;
  }

  /**
   * Build and emit a syntax error message.
   */
  public void reportSyntaxError(Parser.Context ctx) {
    System.err.print(ctx.getLocation() + ": syntax error");
      final int TOKENMAX = 10;
      Parser.SymbolKind[] arg = new Parser.SymbolKind[TOKENMAX];
      int n = ctx.getExpectedTokens(arg, TOKENMAX);
      for (int i = 0; i < n; ++i)
        System.err.print((i == 0 ? ": expected " : " or ")
                         + arg[i].getName());
      Parser.SymbolKind lookahead = ctx.getToken();
      if (lookahead != null)
        System.err.print(" before " + lookahead.getName());
    System.err.println("");
  }

  public void yyerror(Parser.Location loc, String msg) {
    if (loc == null)
      System.err.println(msg);
    else
      System.err.println(loc + ": " + msg);
  }

  public int token(int yytype){
	if (DEBUGGING) {
		System.err.printf ("*** token(%d): %s\n", yytype, yytext());
	}
	return yytype;
  }

  public int token(int yytype, Object lVal){
  	this.lVal = lVal;
  	return token(yytype);
  }

%}

Identifier = [a-zA-Z_][a-zA-Z_0-9]*
Integer = [0-9]+
Float = {Integer}(\.{Integer})?([eE][+-]?{Integer})?
String = \"~\"
Char = \'[^']\'
CommentLines = "/*"~"*/"
CommentLine = "//".*

%%

"{"		{return token('{');}
"}"		{return token('}');}
"("		{return token('(');}
")"		{return token(')');}
"["		{return token('[');}
"]"		{return token(']');}
"<"		{return token('<');}
">"		{return token('>');}
","		{return token(',');}
":"		{return token(':');}
";"		{return token(';');}
"."		{return token('.');}
"!"		{return token('!');}
"+"		{return token('+');}
"-"		{return token('-');}
"*"		{return token('*');}
"/"		{return token('/');}
"%"		{return token('%');}
"&"		{return token('&');}
"|"		{return token('|');}
":="	{return token(ASSIGN);}
"+="	{return token(PLUS_ASSIGN);}
"-="	{return token(MINUS_ASSIGN);}
"*="	{return token(token(TIMES_ASSIGN));}
"/="	{return token(DIV_ASSIGN);}
"%="	{return token(PERC_ASSIGN);}
"||="	{return token(PIPE_PIPE_ASSIGN);}
"&&="	{return token(AMP_AMP_ASSIGN);}
"&="	{return token(AMP_ASSIGN);}
"|="	{return token(PIPE_ASSIGN);}
"<<="	{return token(LT_LT_ASSIGN);}
">>="	{return token(GT_GT_ASSIGN);}
"++"	{return token(PLUS_PLUS);}
"--"	{return token(MINUS_MINUS);}
"&&"	{return token(AMP_AMP);}
"||"	{return token(PIPE_PIPE);}
"<<"	{return token(LT_LT);}
">>"	{return token(GT_GT);}
".."	{return token(DOTS);}
"=="	{return token(EQ_EQ);}
"<="	{return token(LT_EQ);}
">="	{return token(GT_EQ);}
"!="	{return token(BANG_EQ);}

"boolean"		{return token(BOOLEAN);}
"break"		{return token(BREAK);}
"char"		{return token(CHAR);}
"class"		{return token(CLASS);}
"comparable"	{return token(COMPARABLE);}
"const"		{return token(CONST);}
"continue"	{return token(CONTINUE);}
"do"		{return token(DO);}
"else"		{return token(ELSE);}
"enum"		{return token(ENUM);}
"equivalent"	{return token(EQUIVALENT);}
"extends	"	{return token(EXTENDS);}
"final"		{return token(FINAL);}
"float"		{return token(FLOAT);}
"for"	{return token(FOR);}
"foreach"		{return token(FOREACH);}
"function"	{return token(FUNCTION);}
"if"		{return token(IF);}
"implements"	{return token(IMPLEMENTS);}
"import"		{return token(IMPORT);}
"in"		{return token(IN);}
"integer"		{return token(INTEGER);}
"interface"	{return token(INTERFACE);}
"list"		{return token(LIST);}
"map"		{return token(MAP);}
"main"		{return token(MAIN);}
"new	"	{return token(NEW);}
"null"		{return token(NULL);}
"operator"		{return token(OPERATOR);}
"private"		{return token(PRIVATE);}
"protected"		{return token(PROTECTED);}
"public"		{return token(PUBLIC);}
"procedure"	{return token(PROCEDURE);}
"range"		{return token(RANGE);}
"readln"		{return token(READLN);}
"return"		{return token(RETURN);}
"set"		{return token(SET);}
"string"		{return token(STRING);}
"this"		{return token(THIS);}
"type"		{return token(TYPE);}
"virtual"		{return token(VIRTUAL);}
"while"		{return token(WHILE);}
"write"		{return token(WRITE);}
"writeln"		{return token(WRITELN);}

// constants
{Identifier} {return token(IDENTIFIER, yytext());}
{Integer} {return token(NUMBER_INTEGER, new Integer(yytext()));}
{Float} {return token(NUMBER_FLOAT, new Float(yytext()));}
{String} {return token(LITTERAL_STRING, yytext());}
{Char} {return token(LITTERAL_CHAR, yytext());}

// Comments
{CommentLines} {System.err.println("*** comments");}
{CommentLine} {System.err.println("*** comments");}

// Everything else avoid
\n|[^]  {}
