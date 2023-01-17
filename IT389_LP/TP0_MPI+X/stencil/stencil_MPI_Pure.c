
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STENCIL_SIZE_X 25
#define STENCIL_SIZE_Y 30

/** number of buffers for N-buffering; should be at least 2 */
#define STENCIL_NBUFFERS 2

/** conduction coeff used in computation */
static const double alpha = 0.02;

/** threshold for convergence */
static const double epsilon = 0.0001;

/** max number of steps */
static const int stencil_max_steps = 10000;

static double values[STENCIL_NBUFFERS][STENCIL_SIZE_X][STENCIL_SIZE_Y];

/** latest computed buffer */
static int current_buffer = 0;

#pragma region utils

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void) {
  int b, x, y;
  for (b = 0; b < STENCIL_NBUFFERS; b++) {
    for (x = 0; x < STENCIL_SIZE_X; x++) {
      for (y = 0; y < STENCIL_SIZE_Y; y++) {
        values[b][x][y] = 0.0;
      }
    }
    for (x = 0; x < STENCIL_SIZE_X; x++) {
      values[b][x][0] = x;
      values[b][x][STENCIL_SIZE_Y - 1] = STENCIL_SIZE_X - x;
    }
    for (y = 0; y < STENCIL_SIZE_Y; y++) {
      values[b][0][y] = y;
      values[b][STENCIL_SIZE_X - 1][y] = STENCIL_SIZE_Y - y;
    }
  }
}

/** display a (part of) the stencil values */
static void stencil_display(int b, int x0, int x1, int y0, int y1) {
  int x, y;
  for (y = y0; y <= y1; y++) {
    for (x = x0; x <= x1; x++) {
      printf("%8.5g ", values[b][x][y]);
    }
    printf("\n");
  }
}

#pragma endregion utils

/** compute the next stencil step, return 1 if computation has converged */
static int stencil_step(void) {
  int convergence = 1;
  int prev_buffer = current_buffer;
  int next_buffer = (current_buffer + 1) % STENCIL_NBUFFERS;
  int x, y;
  for (x = 1; x < STENCIL_SIZE_X - 1; x++) {
    for (y = 1; y < STENCIL_SIZE_Y - 1; y++) {
      values[next_buffer][x][y] =
          alpha * values[prev_buffer][x - 1][y] +
          alpha * values[prev_buffer][x + 1][y] +
          alpha * values[prev_buffer][x][y - 1] +
          alpha * values[prev_buffer][x][y + 1] +
          (1.0 - 4.0 * alpha) * values[prev_buffer][x][y];
      if (convergence && (fabs(values[prev_buffer][x][y] -
                               values[next_buffer][x][y]) > epsilon)) {
        convergence = 0;
      }
    }
  }
  current_buffer = next_buffer;
  return convergence;
}

int main(int argc, char** argv) {
  stencil_init();
  // printf("# init:\n");
  // stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y - 1);

  struct timespec t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  int s;
  for (s = 0; s < stencil_max_steps; s++) {
    int convergence = stencil_step();
    if (convergence) {
      break;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &t2);
  const double t_usec =
      (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_nsec - t1.tv_nsec) / 1000.0;
  printf("steps: %d\n", s);
  printf("time: %g usecs\n", t_usec);

  printf("height: %d, width: %d\n", STENCIL_SIZE_X, STENCIL_SIZE_Y);
  const long nbCells =
      (STENCIL_SIZE_X - 2) * (STENCIL_SIZE_Y - 2);
  printf("nbCells: %ld\n", nbCells);

  const double nbOperationsByStep = 10 * nbCells;
  printf("nbOperationsByStep: %g\n", nbOperationsByStep);

  const double gigaflops =
      nbOperationsByStep * s * 1000000 / t_usec / 1000000000;
  printf("%g gigaflop/s\n", gigaflops);

  const double nbCellsByS = nbCells * s * 1000000 / t_usec;
  printf("%g cell/s\n", nbCellsByS);
  printf("%d_%d;%g\n", STENCIL_SIZE_X, STENCIL_SIZE_Y, nbCellsByS);

  // stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y - 1);

  return 0;
}
