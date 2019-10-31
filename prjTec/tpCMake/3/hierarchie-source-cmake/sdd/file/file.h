#ifndef FILE_H
#define FILE_H

typedef struct file *file;

extern file file_creer(void);
extern void file_liberer(file self, void (*liberer_element)(void *));

extern void file_enfiler(file self, void *element);
extern void file_defiler(file self);
extern void *file_tete(file self);
extern int file_vide(file self);

extern void file_iterer(file self, void (*traitement)());

#endif  /* FILE_H */
