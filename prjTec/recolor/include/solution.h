#ifndef __SOLUTION_H__
#define __SOLUTION_H__

typedef unsigned int uint;

/**
 * @brief The structure pointer that stores the solution. it will be useful to
 * transfer those data in a file.
 **/
typedef struct solution_s* solution;

/**
 * @brief use to have the length of the solution
 *
 * @param sol a struct solution
 * @return uint the length of the solution
 */
uint len_solution(solution sol);

/**
 * @brief take the arr of int of the sol and turn it into an arr of char
 *
 * @param sol a struct solution
 * @return char* the array of char of the solution
 */
char* string_solution(solution sol);

/**
 * @brief Create a solution object
 *
 * @param tab an array of uint
 * @param length a uint that is the size of the arr
 * @return solution a struct with tab and length
 */
solution create_solution(uint* tab, uint length);

/**
 * @brief free the solution
 *
 * @param sol the solution to be free
 */
void delete_solution(solution sol);

#endif  // __SOLUTION_H__