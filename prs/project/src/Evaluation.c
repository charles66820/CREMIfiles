#include "Evaluation.h"

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "Shell.h"
// #include <readline/history.h>
// #include <readline/readline.h>
// extern int yyparse_string(char *);

typedef unsigned int uint;
#define BUFSIZE 1000

void childSigHandler(int sig) {
  int exitStatus;
  int oldpid = wait(&exitStatus);
  if (oldpid != -1) {
    if (WIFSIGNALED(exitStatus))
      printf("\n%d %s\n", oldpid, strsignal(exitStatus));
    else if (WEXITSTATUS(exitStatus) == 0)
      printf("\n%d %s\n", oldpid, "Done");
    else
      printf("\n%d Exit %d\n", oldpid, WEXITSTATUS(exitStatus));
  }
}

int evaluer_expr(Expression* e) {  // chdir
  int status;
  // Process exptression
  switch (e->type) {
    case VIDE:
      status = 0;
      break;
    case SIMPLE: {
      int argc = LongueurListe(e->arguments);
      if (!strcmp(e->arguments[0], "echo")) {
        for (uint i = 1; i < argc; i++) {
          printf("%s", e->arguments[i]);
          if (i == argc - 1)
            printf("\n");
          else
            printf(" ");
        }
        status = 0;
      } else if (!strcmp(e->arguments[0], "source")) {
        if (!e->arguments[1]) {
          fprintf(stderr,
                  "bash: source: filename argument required\n"
                  "source: usage: source filename [arguments]\n");
          status = 1;
          break;
        }

        int fd = open(e->arguments[1], O_RDONLY);
        if (fd == -1) {
          fprintf(stderr, "Cannot open %s file\n", e->arguments[1]);
          status = 1;
          break;
        }

        /*
        // read file
        char buffer[1024];
        int l = 0;
        do {
          l = read(fd, buffer, 1024);
          if (l) printf("%s", buffer);
        } while (l);//*/

        /*
        // try with stdin redirection
        int saveFd = dup(STDIN_FILENO); // save stdin
        int p[2];
        pipe(p);
        dup2(p[0], STDIN_FILENO); // redirect stdin

        char buffer[1024];
        int l = 0;
        do {
          l = read(fd, buffer, 1024);
          if (l) write(p[1], buffer, l);
        } while (l);
        close(p[1]); // close pipe in

        dup2(saveFd, STDIN_FILENO); // restore stdin
        close(p[0]); // close pipe out
        //*/

        /*
        // try with yyparse_string
        char* line = NULL;
        char buffer[1024];
        int l = 0;
        do {
          l = read(fd, buffer, 1024);
          line = readline(buffer);
          if (!line) break; // ERROR
          strncat(line, "\n", 1);
          int ret = yyparse_string(line);
          free(line);
          return ret;
        } while (l);//*/

        close(fd);
        status = 0;
        break;
      } else {
        int pid;
        if (!(pid = fork())) {
          execvp(e->arguments[0], e->arguments);

          fprintf(stderr, "%s: command not found\n", e->arguments[0]);
          exit(127);
        }
        int exitStatus;
        waitpid(pid, &exitStatus, 0);
        status = WIFSIGNALED(exitStatus) ? WTERMSIG(exitStatus) + 128
                                         : WEXITSTATUS(exitStatus);
      }
      break;
    }
    case SEQUENCE: {
      evaluer_expr(e->gauche);
      status = evaluer_expr(e->droite);
      break;
    }
    case SEQUENCE_ET: {
      status = evaluer_expr(e->gauche);
      if (!status) status = evaluer_expr(e->droite);
      break;
    }
    case SEQUENCE_OU: {
      status = evaluer_expr(e->gauche);
      if (status) status = evaluer_expr(e->droite);
      break;
    }
    case BG: {
      if (!fork()) {
        int retStatus = evaluer_expr(e->gauche);
        if (retStatus > 128)
          raise(retStatus - 128);
        else
          exit(retStatus);
      }
      status = 0;
      break;
    }
    case REDIRECTION_O: {
      remove(e->arguments[0]);  // Empty file
      int fd = open(e->arguments[0], O_WRONLY | O_CREAT, 0666);
      if (fd == -1) {
        fprintf(stderr, "Cannot create %s file\n", e->arguments[0]);
        status = 1;
        break;
      }

      int pid;
      if (!(pid = fork())) {
        dup2(fd, STDOUT_FILENO);
        exit(evaluer_expr(e->gauche));
      }

      int exitStatus;
      waitpid(pid, &exitStatus, 0);
      close(fd);
      status = WEXITSTATUS(exitStatus);
      break;
    }
    case REDIRECTION_I: {
      int fd = open(e->arguments[0], O_RDONLY);
      if (fd == -1) {
        fprintf(stderr, "Cannot open %s file\n", e->arguments[0]);
        status = 1;
        break;
      }

      int pid;
      if (!(pid = fork())) {
        dup2(fd, STDIN_FILENO);
        exit(evaluer_expr(e->gauche));
      }

      int exitStatus;
      waitpid(pid, &exitStatus, 0);
      close(fd);
      status = WEXITSTATUS(exitStatus);
      break;
    }
    case REDIRECTION_A: {
      int fd = open(e->arguments[0], O_WRONLY | O_CREAT, 0666);
      if (fd == -1) {
        fprintf(stderr, "Cannot create %s file\n", e->arguments[0]);
        status = 1;
        break;
      }

      int pid;
      if (!(pid = fork())) {
        lseek(fd, 0, SEEK_END);
        dup2(fd, STDOUT_FILENO);
        exit(evaluer_expr(e->gauche));
      }

      int exitStatus;
      waitpid(pid, &exitStatus, 0);
      close(fd);
      status = WEXITSTATUS(exitStatus);
      break;
    }
    case REDIRECTION_E: {
      remove(e->arguments[0]);  // Empty file
      int fd = open(e->arguments[0], O_WRONLY | O_CREAT, 0666);
      if (fd == -1) {
        fprintf(stderr, "Cannot create %s file\n", e->arguments[0]);
        status = 1;
        break;
      }

      int pid;
      if (!(pid = fork())) {
        dup2(fd, STDERR_FILENO);
        exit(evaluer_expr(e->gauche));
      }

      int exitStatus;
      waitpid(pid, &exitStatus, 0);
      close(fd);
      status = WEXITSTATUS(exitStatus);
      break;
    }
    case REDIRECTION_EO: {
      remove(e->arguments[0]);  // Empty file
      int fd = open(e->arguments[0], O_WRONLY | O_CREAT, 0666);
      if (fd == -1) {
        fprintf(stderr, "Cannot create %s file\n", e->arguments[0]);
        status = 1;
        break;
      }

      int pid;
      if (!(pid = fork())) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        exit(evaluer_expr(e->gauche));
      }

      int exitStatus;
      waitpid(pid, &exitStatus, 0);
      close(fd);
      status = WEXITSTATUS(exitStatus);
      break;
    }
    case PIPE: {
      // create pipe
      int pipe1[2];
      int s = pipe(pipe1);
      if (s) {
        status = 1;
        break;
      }

      int pidLeft = fork();
      if (!pidLeft) {
        close(pipe1[0]);  // close pipe read
        // redirect left cmd stdout to pipe write
        dup2(pipe1[1], STDOUT_FILENO);
        // left cmd
        exit(evaluer_expr(e->gauche));
      } else {
        int pidRight = fork();
        if (!pidRight) {
          // close pipe write
          close(pipe1[1]);
          // redirect pipe read to right cmd stdin
          dup2(pipe1[0], STDIN_FILENO);
          // right cmd
          exit(evaluer_expr(e->droite));
        } else {
          // wait left cmd
          waitpid(pidLeft, NULL, 0);
          // on cmd finish close pipe write
          close(pipe1[1]);

          // wait right cmd
          int exitStatus;
          waitpid(pidRight, &exitStatus, 0);
          // on cmd finish close pipe read
          close(pipe1[0]);
          status = WEXITSTATUS(exitStatus);
        }
      }
      break;
    }
    default:
      fprintf(stderr, "not yet implemented\n");
      status = 1;
  }

  // Process backgrand process
  struct sigaction sa;

  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = childSigHandler;
  sigaction(SIGCHLD, &sa, NULL);

  return status;
}
