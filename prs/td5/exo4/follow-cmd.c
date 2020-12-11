#define _GNU_SOURCE

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define NB_CMD 4

// Returns duration in secs
#define TIME_DIFF(t1, t2) \
  ((t2.tv_sec - t1.tv_sec) + ((double)(t2.tv_usec - t1.tv_usec)) / 1000000)

struct state {
  pid_t pid;
  char command[100];
  char arg[100];
  int in_progress;
  struct timeval start;
  struct timeval end;
} state_tab[NB_CMD];

char *commands[NB_CMD][10] = {{"sleep", "0", NULL},
                               {"sleep", "3", NULL},
                               {"sleep", "4", NULL},
                               {"sleep", "5", NULL}};

void edit_state(pid_t pid) {
  for (int i = 0; i < NB_CMD; i++)
    if (state_tab[i].pid == pid) {
      state_tab[i].in_progress = 0;
      gettimeofday(&state_tab[i].end, NULL);
      return;
    }
  fprintf(stderr, "%d non enregistré\n", pid);
}

void show_state() {
  int i;
  struct timeval now;
  gettimeofday(&now, NULL);
  for (i = 0; i < NB_CMD; i++) {
    if (state_tab[i].pid > 0) {
      printf("%d : %s(%s)", state_tab[i].pid, state_tab[i].command,
             state_tab[i].arg);
      if (state_tab[i].in_progress)
        printf(" en cours depuis : %gs\n",
               TIME_DIFF(state_tab[i].start, now));
      else
        printf(" terminé durée détectée : %gs\n",
               TIME_DIFF(state_tab[i].start, state_tab[i].end));
    }
  }
  printf("\n");
}

int remains_command() {
  for (int i = 0; i < NB_CMD; i++)
    if (state_tab[i].in_progress) return 1;
  return 0;
}

void launch_commands() {
  int i;
  pid_t cpid;

  /* Lancement */
  for (i = 0; i < NB_CMD; i++) {
    cpid = fork();

    if (cpid == -1) {
      perror("fork");
      exit(EXIT_FAILURE);
    }

    if (cpid == 0) {
      execvp(commands[i][0], commands[i]);
      perror(commands[i][0]);
      abort();
    }

    state_tab[i].pid = cpid;
    strcpy(state_tab[i].command, commands[i][0]);
    strcpy(state_tab[i].arg, commands[i][1]);
    state_tab[i].in_progress = 1;
    gettimeofday(&state_tab[i].start, NULL);
  }
}

int main(int argc, char *argv[]) { // TODO: here
  launch_commands();

  for (int cpt = 1; remains_command(); cpt++) {
    pid_t w;
    char buf[1024];

    printf("iteration %d\n", cpt);
    w = waitpid(0, NULL, WNOHANG);
    printf("pid = %d\n", w);
    if (w > 0) edit_state(w);

    int r = read(0, buf, 1024);
    if (r == -1) perror("read");
    show_state();
  }

  printf("Tous les processus se sont terminés !\n");
  show_state();
  exit(EXIT_SUCCESS);
}
