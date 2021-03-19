#include "heap.h"

#include <assert.h>
#include <stdlib.h>

heap heap_create(int k, int (*f)(const void *, const void *)) {
  if (k < 0) return NULL;

  heap h = malloc(sizeof(*h));
  assert(h != NULL);
  h->n = 0;
  h->nmax = k;
  h->f = f;
  h->array = malloc((k + 1) * sizeof(void *));
  assert(h->array != NULL);

  return h;
}

void heap_destroy(heap h) {
  free(h->array);
  free(h);
}

bool heap_empty(heap h) { return h->n <= 0; }

bool heap_add(heap h, void *object) {
  if (h->n >= h->nmax) return true;

  int pos = ++h->n;
  h->array[pos] = object;

  while (pos != 1 && h->f(h->array[pos], h->array[pos / 2]) < 0) {
    void *tmp = h->array[pos / 2];
    h->array[pos / 2] = h->array[pos];
    h->array[pos] = tmp;
    pos = pos / 2;
  }

  return false;
}

void *heap_top(heap h) {
  if (heap_empty(h)) return NULL;
  return h->array[1];
}

void *heap_pop(heap h) {
  void *root = heap_top(h);
  if (root == NULL) return NULL;

  h->array[1] = h->array[h->n];  // Last to first
  h->n--;

  // Re sort
  int pos = 1;
  while (pos <= h->n / 2 &&
         (h->f(h->array[pos * 2], h->array[pos]) < 0 ||
          h->f(h->array[(pos * 2) + 1], h->array[pos]) < 0)) {
    int childPos = (h->f(h->array[pos * 2], h->array[(pos * 2) + 1]) < 0)
                       ? pos * 2
                       : (pos * 2) + 1;
    // swap
    void *tmp = h->array[pos];
    h->array[pos] = h->array[childPos];
    h->array[childPos] = tmp;
    pos = childPos;
  }

  return root;
}
