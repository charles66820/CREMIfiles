#ifndef HEAP_H
#define HEAP_H
#include <stdbool.h>

// Structure de tas binaire:
//
//  array = tableau de stockage des objets à partir de l'indice 1 (au lieu de 0)
//  n     = nombre d'objets (qui sont des void*) stockés dans le tas
//  nmax  = nombre maximum d'objets stockables dans le tas
//  f     = fonction de comparaison de deux objets (min, max, ..., cf. man qsort)
//
// Attention ! "heap" est défini comme un pointeur pour optimiser les
// appels (empilement d'un mot (= 1 pointeur) au lieu de 4 sinon).

typedef struct{
  void* *array;
  int n, nmax;
  int (*f)(const void*, const void*);
} *heap;


// Crée un tas pouvant accueillir au plus k>0 objets avec une fonction
// de comparaison f() prédéfinie. NB: La taille d'un objet pointé par
// un pointeur h est sizeof(*h).
heap heap_create(int k, int (*f)(const void *, const void *));


// Détruit le tas h. On supposera h!=NULL. Attention ! Il s'agit de
// libérer ce qui a été alloué par heap_create(). NB: Les objets
// stockés dans le tas n'ont pas à être libérés.
void heap_destroy(heap h);


// Renvoie vrai si le tas h est vide, faux sinon. On supposera
// h!=NULL.
bool heap_empty(heap h);


// Ajoute un objet au tas h. On supposera h!=NULL. Renvoie vrai s'il
// n'y a pas assez de place, et faux sinon.
bool heap_add(heap h, void *object);


// Renvoie l'objet en haut du tas h, c'est-à-dire l'élément minimal
// selon f(), sans le supprimer. On supposera h!=NULL. Renvoie NULL si
// le tas est vide.
void *heap_top(heap h);


// Comme heap_top() sauf que l'objet est en plus supprimé du
// tas. Renvoie NULL si le tas est vide.
void *heap_pop(heap h);

#endif
