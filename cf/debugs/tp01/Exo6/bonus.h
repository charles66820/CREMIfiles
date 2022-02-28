#include "limits.h"

/**
 * @brief Computes the min of these arguments.
 * 
 * @param a an integer
 * @param b an integer
 * @param c an integer
 * @return int The smallest value among a, b and c.
 */
int min(int a, int b, int c);

/**
 * @brief Computes the next step of @p a in the Syracuse sequence, i.e., its quotient by 2 if @p a is even, and 3 * @p a +1 otherwise.
 * 
 * @param a an integer.
 * @return int The next value in the Syracuse sequence.
 */
int syracuseStep(int a);

/**
 * @brief Computes the rounding of the (real) quotient of a by b, i.e. the closest integer to the quotient. We will accept a function that does that only for positive integers (remember the C int division is not the Euclidean division). As a bonus, you might implement a function that works for any pair of integer (such that the quotient is defined of course).
 *
 * @param a the dividend
 * @param b the diviser
 * @return int The rounding of a/.b.
 * @pre a and b are positive (or not, depending of your implementation).
 * @pre b is not 0.
 */
int roundedDiv(int a, int b);