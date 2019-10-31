#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "memoire.h"

#include "chaine.h"

int
chaine_compter_occurrences(const char s[], char c) {
    int nb = 0;
    int i;

    for (i = 0; s[i] != '\0'; i++)
	if (s[i] == c)
	    nb++;

    return nb;
}

void
chaine_remplacer_occurrences(char s[], char c1, char c2) {
    int i;

    for (i = 0; s[i] != '\0'; i++)
	if (s[i] == c1)
	    s[i] = c2;
}

char *
chaine_dupliquer(const char s[]) {
    char *res = memoire_allouer(strlen(s) + 1);

    strcpy(res, s);

    return res;
}

char *
chaine_concatener(const char s1[], const char s2[]) {
    size_t len1 = strlen(s1);
    char *res = memoire_allouer(len1 + strlen(s2) + 1);

    strcpy(res, s1);
    strcpy(res + len1, s2);

    return res;
}

char *
chaine_vers_majuscules(const char s[]) {
    char *res = chaine_dupliquer(s);
    int i;

    for (i = 0; s[i] != '\0'; i++)
	res[i] = toupper(res[i]);

    return res;
}

char *
chaine_vers_minuscules(const char s[]) {
    char *res = chaine_dupliquer(s);
    int i;

    for (i = 0; s[i] != '\0'; i++)
	res[i] = tolower(res[i]);

    return res;
}

char *
chaine_suffixe(const char s[], char c) {
    int derniere_occurrence = -1;
    int i;

    for (i = 0; s[i] != '\0'; i++)
	if (s[i] == c)
		derniere_occurrence = i;

    return chaine_dupliquer(s + derniere_occurrence + 1);
}

char *
chaine_prefixe(const char s[], char c) {
    int derniere_occurrence = -1;
    char *res;
    int i;

    for (i = 0; s[i] != '\0'; i++)
	if (s[i] == c)
		derniere_occurrence = i;

    if (derniere_occurrence == -1)
	derniere_occurrence = i;

    res = memoire_allouer(derniere_occurrence + 1);

    for (i = 0; i < derniere_occurrence; i++)
	res[i] = s[i];

    res[i] = '\0';

    return res;
}
