LEX = flex
YACC = yacc -d -v
CC = gcc
CFLAGS = -std=c99 -Og -g
CPPFLAGS = -D_XOPEN_SOURCE=700
LDLIBS = -lreadline -ll # -ly

Shell: Shell.o Affichage.o Evaluation.o y.tab.o lex.yy.o
Shell.o: Shell.c Shell.h
Affichage.o :  Shell.h Affichage.h Affichage.c
Evaluation.o :  Shell.h Evaluation.h Evaluation.c
lex.yy.o: lex.yy.c y.tab.h Shell.h

y.tab.c y.tab.h: Analyse.y
	$(YACC) Analyse.y

lex.yy.c: Analyse.l Shell.h y.tab.h
	$(LEX) Analyse.l

.PHONY: clean
clean:
	rm -f *.o y.tab.* y.output lex.yy.* Shell
