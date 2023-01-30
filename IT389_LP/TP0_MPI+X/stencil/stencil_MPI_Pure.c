
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef STENCIL_SIZE_X
#define STENCIL_SIZE_X 25
#endif
#ifndef STENCIL_SIZE_Y
#define STENCIL_SIZE_Y 30
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 6
#endif
#ifndef TILE_HEIGHT
#define TILE_HEIGHT 6
#endif

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

/** compute the next stencil step, return 1 if computation has converged */
static int stencil_step(void) {
  int convergence = 1;
  int prev_buffer = current_buffer;
  int next_buffer = (current_buffer + 1) % STENCIL_NBUFFERS;

  int nbTW = (STENCIL_SIZE_X - 2) / (TILE_WIDTH - 2);
  int nbReminderTW = (STENCIL_SIZE_X - 2) - ((TILE_WIDTH - 2) * nbTW);
  if (nbReminderTW > 0) nbTW += 1;

  int nbTH = (STENCIL_SIZE_Y - 2) / (TILE_HEIGHT - 2);
  int nbReminderTH = (STENCIL_SIZE_Y - 2) - ((TILE_HEIGHT - 2) * nbTH);
  if (nbReminderTH > 0) nbTH += 1;

  int tw, th;
  for (tw = 0; tw < nbTW; tw++) {
    for (th = 0; th < nbTH; th++) {
      int haloX = (tw * (TILE_WIDTH - 2));
      int haloY = (th * (TILE_HEIGHT - 2));

      int x, y;
      for (x = haloX + 1; x < fmin(haloX + TILE_WIDTH, STENCIL_SIZE_X) - 1;
           x++) {
        for (y = haloY + 1; y < fmin(haloY + TILE_HEIGHT, STENCIL_SIZE_Y) - 1;
             y++) {
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
    }
  }

  current_buffer = next_buffer;
  return convergence;
}

int main(int argc, char** argv) {
  stencil_init();
  // printf("# init:\n");
  // stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y -
  // 1);

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
      (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1E3;
  const long nbCells = (STENCIL_SIZE_X - 2) * (STENCIL_SIZE_Y - 2);
  const long nbOperationsByStep = 10 * nbCells;
  const double gigaflops = nbOperationsByStep * s * 1E6 / t_usec / 1E9;
  const double nbCellsByS = nbCells * s * 1E6 / t_usec;

  fprintf(
      stderr,
      "steps,time(µ sec),height,width,nbCells,fpOpByStep,gigaflop/s,cell/s\n");
  fprintf(stderr, "%d,%g,%d,%d,%ld,%ld,%g,\033[0;32m%g\033[0m\n", s, t_usec,
          STENCIL_SIZE_X, STENCIL_SIZE_Y, nbCells, nbOperationsByStep,
          gigaflops, nbCellsByS);

  stencil_display(current_buffer, 0, STENCIL_SIZE_X - 1, 0, STENCIL_SIZE_Y - 1);

  return 0;
}
