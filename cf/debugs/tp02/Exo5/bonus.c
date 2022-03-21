#include "bonus.h"

int div(int a, int b, int *res) {
  if (b == 0 || (a == INT_MIN && b == -1)) return -1;
  *res = a / b;
  return 0;
}

int add(int a, int b, int *res) {
  if ((a >= 0 && b >= 0 && (a > INT_MAX - b)) ||
      (a < 0 && b < 0 && (a < INT_MIN - b)))
    return -1;

  *res = a + b;
  return 0;
}
