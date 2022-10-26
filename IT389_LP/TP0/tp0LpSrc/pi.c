#include <math.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
gcc pi.c -o pi -fopenmp
*/

double compute_pi_manual_reduction(long num_steps) {
  double x, pi, sum = 0.0;

  int nb_threads;
#pragma omp parallel
  nb_threads = omp_get_max_threads();

  double sum_tab[nb_threads];
  for (size_t i = 0; i < nb_threads; i++) sum_tab[i] = 0.0;

  double step = 1.0 / (double)num_steps;
  int start = 1;
  int end = num_steps;

#pragma omp parallel shared(step, sum_tab) firstprivate(sum)
  {
    int thread_num = omp_get_thread_num();
#pragma omp for private(x)
    for (int i = start; i <= end; i++) {
      x = (i - 0.5) * step;
      sum = sum + (4.0 / (1.0 + x * x));
    }
    sum_tab[thread_num] = sum;
  }

  for (size_t i = 0; i < nb_threads; i++)
#pragma omp atomic update
    sum = sum + sum_tab[i];

  pi = step * sum;
  return pi;
}

double compute_pi_omp_reduction(long num_steps) {
  double x, pi, sum = 0.0;
  double step = 1.0 / (double)num_steps;
  int start = 1;
  int end = num_steps;

#pragma omp parallel for shared(step) private(x) reduction(+ : sum)
  for (int i = start; i <= end; i++) {
    x = (i - 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
  return pi;
}

double compute_pi_taskloop(long num_steps) {
  double x, pi, sum = 0.0;
  double step = 1.0 / (double)num_steps;
  int start = 1;
  int end = num_steps;

#pragma omp parallel shared(step)
  {
#pragma omp single
#pragma omp taskloop private(x) reduction(+ : sum)  // grainsize(200)
    for (int i = start; i <= end; i++) {
      x = (i - 0.5) * step;
      sum = sum + 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
  }
  return pi;
}

double compute_pi_seq(long num_steps) {
  double x, pi, sum = 0.0;
  double step = 1.0 / (double)num_steps;
  int start = 1;
  int end = num_steps;

  for (int i = start; i <= end; i++) {
    x = (i - 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
  return pi;
}

int main() {
  double pi;
  double PI25DT = 3.141592653589793238462643;
  static long num_steps = 1000000000;

  double start_timer;
  double end_timer;

  start_timer = omp_get_wtime();
  pi = compute_pi_seq(num_steps);
  end_timer = omp_get_wtime();
  printf("compute_pi_seq : %f seconds\n", end_timer - start_timer);
  printf("pi := %.16e  %.e\n", pi, fabs(pi - PI25DT));

  start_timer = omp_get_wtime();
  pi = compute_pi_omp_reduction(num_steps);
  end_timer = omp_get_wtime();
  printf("\ncompute_pi_omp_reduction : %f seconds\n", end_timer - start_timer);
  printf("pi := %.16e  %.e\n", pi, fabs(pi - PI25DT));

  start_timer = omp_get_wtime();
  pi = compute_pi_manual_reduction(num_steps);
  end_timer = omp_get_wtime();
  printf("\ncompute_pi_manual_reduction : %f seconds\n",
         end_timer - start_timer);
  printf("pi := %.16e  %.e\n", pi, fabs(pi - PI25DT));

  start_timer = omp_get_wtime();
  pi = compute_pi_taskloop(num_steps);
  end_timer = omp_get_wtime();
  printf("\ncompute_pi_taskloop : %f seconds\n", end_timer - start_timer);
  printf("pi := %.16e  %.e\n", pi, fabs(pi - PI25DT));

  return 0;
}
