#ifndef PAIRE_H
#define PAIRE_H

typedef struct paire *paire;

extern paire paire_creer(void *element, paire suivant);
extern void paire_liberer(paire self);

extern void *paire_car(paire self);
extern paire paire_cdr(paire self);

extern void paire_modifier_car(paire self, void *element);
extern void paire_modifier_cdr(paire self, paire suivant);

extern void paire_iterer(paire self, void (*traitement)());

#endif  /* PAIRE_H */
