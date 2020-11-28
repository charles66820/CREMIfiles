#define _GNU_SOURCE

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define NCOMMANDES 4

// Returns duration in secs
#define TIME_DIFF(t1, t2) \
  ((t2.tv_sec - t1.tv_sec) + ((double)(t2.tv_usec - t1.tv_usec)) / 1000000)

struct etat {
  pid_t pid;
  char commande[100];
  char arg[100];
  int en_cours;
  struct timeval debut;
  struct timeval fin;
} etat_tableau[NCOMMANDES];

char *commandes[NCOMMANDES][10] = {{"sleep", "0", NULL},
                                   {"sleep", "3", NULL},
                                   {"sleep", "4", NULL},
                                   {"sleep", "5", NULL}};

void modifier_etat(pid_t pid) {
  for (int i = 0; i < NCOMMANDES; i++)
    if (etat_tableau[i].pid == pid) {
      etat_tableau[i].en_cours = 0;
      gettimeofday(&etat_tableau[i].fin, NULL);
      return;
    }
  fprintf(stderr, "%d non enregistré\n", pid);
}

void afficher_etat() {
  int i;
  struct timeval now;
  gettimeofday(&now, NULL);
  for (i = 0; i < NCOMMANDES; i++) {
    if (etat_tableau[i].pid > 0) {
      printf("%d : %s(%s)", etat_tableau[i].pid, etat_tableau[i].commande,
             etat_tableau[i].arg);
      if (etat_tableau[i].en_cours)
        printf(" en cours depuis : %gs\n",
               TIME_DIFF(etat_tableau[i].debut, now));
      else
        printf(" terminé durée détectée : %gs\n",
               TIME_DIFF(etat_tableau[i].debut, etat_tableau[i].fin));
    }
  }
  printf("\n");
}

int reste_commande() {
  for (int i = 0; i < NCOMMANDES; i++)
    if (etat_tableau[i].en_cours) return 1;
  return 0;
}

void lancer_commandes() {
  int i;
  pid_t cpid;

  /* Lancement */
  for (i = 0; i < NCOMMANDES; i++) {
    cpid = fork();

    if (cpid == -1) {
      perror("fork");
      exit(EXIT_FAILURE);
    }

    if (cpid == 0) {
      execvp(commandes[i][0], commandes[i]);
      perror(commandes[i][0]);
      abort();
    }

    etat_tableau[i].pid = cpid;
    strcpy(etat_tableau[i].commande, commandes[i][0]);
    strcpy(etat_tableau[i].arg, commandes[i][1]);
    etat_tableau[i].en_cours = 1;
    gettimeofday(&etat_tableau[i].debut, NULL);
  }
}

void childSigHandler(int sig) {
  pid_t w;
  int exitStatus;
  do {
    w = waitpid(0, &exitStatus, WNOHANG);
    if (w > 0) {
      printf("pid = %d\n", w);
      modifier_etat(w);
    }
    /* // For print status
    if (WIFSIGNALED(exitStatus)) strsignal(exitStatus);
    else WEXITSTATUS(exitStatus); */
  } while (w > 0);
}

int main(int argc, char *argv[]) {
  // Setup handler for child process
  struct sigaction sa;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = childSigHandler;
  sigaction(SIGCHLD, &sa, NULL);
  // mask ?

  lancer_commandes();

  for (int cpt = 1; reste_commande(); cpt++) {
    // Read like a terminal
    char buf[1024];
    printf("iteration %d\n", cpt);
    int r;
    do {
      r = read(0, buf, 1024);
    } while (r == -1);
    afficher_etat();
  }

  printf("Tous les processus se sont terminés !\n");
  afficher_etat();
  exit(EXIT_SUCCESS);
}
