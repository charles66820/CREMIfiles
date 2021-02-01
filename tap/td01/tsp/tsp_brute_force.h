#ifndef TSP_BRUTE_FORCE_H
#define TSP_BRUTE_FORCE_H

#include "tools.h"

/**
 * @brief Distance between A and B
 * @param A Source point
 * @param B Dest point
 * @return double Distance
 */
double dist(point A, point B);

/**
 * @brief Retrun the distance value
 * @param V Point table
 * @param n Tables length (nb points)
 * @param P Index table for point table list
 * @return double Distance value
 */
double value(point *V, int n, int *P);

/**
 * @brief Search the best tour
 * @param V Point table
 * @param n Tables length (nb points)
 * @param Q Reference for optimal tour
 * @return double The smaller value
 */
double tsp_brute_force(point *V, int n, int *Q);

/**
 * @brief Optimized value fonction
 * @param V Point table
 * @param n Tables length (nb points)
 * @param P Index table for point table list
 * @param wmin Current minimal value
 * @return double Distance value
 */
double value_opt(point *V, int n, int *P, double wmin);

/**
 * @brief 
 * @param P Return the largest permutation, in lexicographic order;
 * @param n Tables length (nb points)
 * @param k 
 */
void MaxPermutation(int *P, int n, int k);

/**
 * @brief 
 * @param V 
 * @param n 
 * @param Q 
 * @return double 
 */
double tsp_brute_force_opt(point *V, int n, int *Q);

double value_opt2(point *V, int n, int *P, double wmin);

double tsp_brute_force_opt2(point *V, int n, int *Q);

#endif /* TSP_BRUTE_FORCE_H */
