#ifndef TSP_MST_H
#define TSP_MST_H

#include "tools.h"

graph createGraph(int n);

// Libère un graphe G et ses listes.
void freeGraph(graph G);

// Ajoute l'arête u-v au graphe G de manière symétrique. Les degrés de
// u et v doivent être à jour et les listes suffisamment grandes.
void addEdge(graph G, int u, int v);

// Une arête u-v avec un poids.
typedef struct {
  int u, v;      // extrémités de l'arête u-v
  double weight; // poids de l'arête u-v
} edge;

int compEdge(const void *e1, const void *e2);
void Union(int x, int y, int *parent, int *rank);
int Find(int x, int *parent);
void dfs(graph G, int u, int *P, int p);
double tsp_mst(point *V, int n, int *P, graph T);

#endif /* TSP_MST */
