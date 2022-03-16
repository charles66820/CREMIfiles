#include "limits.h"

/*@
  ensures \result == a || \result == b || \result == c;
  ensures \result <= a;
  ensures \result <= b;
  ensures \result <= c;
*/
/**
 * @brief Computes the min of these arguments.
 * @param a an integer
 * @param b an integer
 * @param c an integer
 * @return int The smallest value among a, b and c.
 */
int min(int a, int b, int c);

/*@
  requires 3 * a + 1 > INT_MIN;
  requires 3 * a + 1 <= INT_MAX;
  ensures \result <= a / 2 || \result <= 3 * a + 1;
*/
/**
 * @brief Computes the next step of @p a in the Syracuse sequence, i.e., its quotient by 2 if @p a is even, and 3 * @p a +1 otherwise.
 *
 * @param a an integer.
 * @return int The next value in the Syracuse sequence.
 */
int syracuseStep(int a);

/*@
  requires a / b <= INT_MAX;
  requires b != 0;
  ensures \result == a / b;
*/
/**
 * @brief Computes the rounding of the (real) quotient of a by b, i.e. the closest integer to the quotient. We will accept a function that does that only for positive integers (remember the C int division is not the Euclidean division). As a bonus, you might implement a function that works for any pair of integer (such that the quotient is defined of course).
 *
 * @param a the dividend
 * @param b the diviser
 * @return int The rounding of a/.b.
 */
int roundedDiv(int a, int b);