#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void generer() {
  char chaine[] = "jusqu'ici tout va bien";

  for (int i = 0; i < 10; i++)
#pragma omp task firstprivate(chaine) firstprivate(i) // edit
    printf("tache %d par thread %d >>>> %s <<<< \n", i, omp_get_thread_num(),
           chaine);

  // #pragma omp taskwait

  // When we comment the `taskwait` the function return before all task as
  // execute this provoc the unstack of `chaine` and the not executed task can't
  // acces to `chaine`
}

int main() {
#pragma omp parallel
  {
#pragma omp single
    {
      generer();
      printf("%d est sorti de generer \n", omp_get_thread_num());
    }
  }
  return 0;
}
