#ifndef VEXT_H
#define VEXT_H

typedef struct vext *vext;

extern vext vext_creer(void);
extern void vext_liberer(vext self);
extern unsigned int vext_nombre_d_elements(vext self);
extern void *vext_lire(vext self, unsigned int i);
extern void vext_ecrire(vext self, unsigned int i, void *valeur);
extern void vext_ajouter(vext self, void *valeur);
extern void vext_iterer(vext self, void (*fonction)(void *));
extern void vext_afficher(vext self);
extern void vext_definir_affichage(vext self, void (*afficher)());
extern void vext_definir_liberation(vext self, void (*liberer)());

#endif  /* VEXT_H */
