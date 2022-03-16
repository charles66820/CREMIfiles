#include "bonus.h"

int min(int a, int b, int c) {
  if (a <= b && a <= c) return a;
  if (b <= a && b <= c) return b;
  if (c <= a && c <= b) return c;
}

int syracuseStep(int a) {
  if (a & 1) return a / 2;
  return 3 * a + 1;
}

int roundedDiv(int a, int b) {
  return a / b;
}
