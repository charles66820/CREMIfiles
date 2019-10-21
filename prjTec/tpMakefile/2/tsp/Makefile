CC = gcc
CFLAGS = -g -Wall -std=c99
CPPFLAGS =
LDFLAGS = -lm
SOURCES = $(wildcard *.c)
OBJETS = $(SOURCES:.c=.o)

all : solve-TSP convert-2D-to-full

# crée la library tsp
libtsp.a : tsp.o memoire.o matrice2d.o
	ar -cr $@ $^

# crée les exécutables
solve-TSP : solve-TSP.o libtsp.a
	$(CC) $< $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -L. -ltsp -o $@

convert-2D-to-full : convert-2D-to-full.o libtsp.a
	$(CC) $< $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -L. -ltsp -o $@

# crée les fichier obj
include depends.txt

# nettoie le dossier des fichiers créés
.PHONY : clean
clean :
	rm -f $(OBJETS) *.a convert-2D-to-full solve-TSP *.dat depends.txt

# défini les dépendance pour les fichier obj
depends.txt :
	$(CC) -M $(CPPFLAGS) $(LDFLAGS) $(SOURCES) > depends.txt
