#include "limits.h"

/* Cette fonction place le quotient de a et b dans *res et renvoie 0 en cas de succès et -1 en cas d’échec.
Indication: pour les overflow, pensez que INT_MIN = -INT_MAX - 1 !
*/
/**
 * @brief Computes a/b and puts the result in *res. Return 0 in case of success and -1 in case of failure.
 *
 * @param a the dividend
 * @param b the diviser
 * @param res a pointer that will receive the result
 * @return int 0 if the computation succeeds, -1 otherwise.
 *
 * @pre this function is applicable to any context provided res is allocated.
 */
int div(int a, int b, int *res);

/* Cette fonction place la somme de a et b dans *res et renvoie 0 en cas de succès et -1 en cas d’échec.
Dans quel cas peut-il y avoir un overflow ? Que se passe-t’il dans ce cas pour le résultat ?
Indication : attention, vos tests et affectation dans la fonction ne doivent pas faire d’overflow du tout pour passer RTE. Pensez à tester les calculs avant de les faire, quitte à avoir à tester les signes de certaines variables pour assurer que vos tests ne font pas d’overflow.
*/
/**
 * @brief Computes a+b and puts the result in *res. Return 0 in case of success and -1 in case of failure.
 *
 * @param a an integer
 * @param b an integer
 * @param res a pointer that will receive the result
 * @return int 0 if the computation succeeds, -1 otherwise
 *
 * @pre this function is applicable to any context, provided res is allocated.
 */
int add(int a, int b, int *res);

/* Cette fonction place dans *res la valeur maximale parmi tab[index1], tab[index2], tab[index3]. Elle renvoie -1 en cas d’échec.
En cas de succès, elle la fonction renverra 1 si les trois index sont différents, 2 si exactement deux sont égaux et 3 si les trois sont égaux (attention, je parle des indices, PAS de leur contenu).
Indication : faites attention à ce que vos tests soient bien disjoints.

C’est ici un bon exemple de ce qui était discuté au TP1 sur les behaviors : à mon avis ici, il est pertinent de n’avoir que deux comportements : quand les indices sont valides et quand il ne le sont pas. Et ceci car s’il y a bien trois valeurs de retour possibles quand les indices sont valides, tous ont une post-condition commune (placer le maximum dans *res).
*/
/**
 * @brief Puts in *res the maximal value among tab[index1], tab[index2] and tab[index3] if these values are in the array tab (of size size). Moreover, it returns the number of identical indexes in case of success. It returns -1 if this is not possible.
 * @param tab an array
 * @param size the size of tab
 * @param index1 an index
 * @param index2 an index
 * @param index3 an index
 * @param res a pointer that will receive the max between tab[index1], tab[index2] and tab[index3].
 * @return int -1 if at least one of the indexes is outside of the range of tab. Otherwise, it return 1 if all indexes are different, 2 if two of them are equal and 3 if they are all the same.
 */
int max_3_tab(int *tab, int size, int index1, int index2, int index3, int *res);