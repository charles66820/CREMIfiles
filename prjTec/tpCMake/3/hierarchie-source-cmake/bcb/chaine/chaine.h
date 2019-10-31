#ifndef CHAINE_H_
#define CHAINE_H_

int chaine_compter_occurrences(const char [], char);
void chaine_remplacer_occurrences(char [], char, char);

char *chaine_dupliquer(const char []);
char *chaine_concatener(const char [], const char []);

char *chaine_vers_majuscules(const char []);
char *chaine_vers_minuscules(const char []);

char *chaine_suffixe(const char [], char);
char *chaine_prefixe(const char [], char);

#endif /* CHAINE_H_ */
