#include "Evaluation.h"

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "Shell.h"

typedef unsigned int uint;
#define BUFSIZE 1000

void childSigHandler(int sig) {
  int exitStatus;
  int oldpid = waitpid(0, &exitStatus, WNOHANG); // BUG: call by children
  if (oldpid != -1 && oldpid != 0) {
    if (WIFSIGNALED(exitStatus))
      printf("\n\e[ABackground process %d finish with : %s\n", oldpid,
             strsignal(exitStatus));
    else if (WEXITSTATUS(exitStatus) == 0)
      printf("\n\e[ABackground process %d finish with : %s\n", oldpid, "Done");
    else
      printf("\n\e[ABackground process %d finish with : Exit %d\n", oldpid,
             WEXITSTATUS(exitStatus));
  }
}

struct sigaction sa, oldSa;
int evaluer_exprL(Expression* e) {
  int status;
  // Process exptression
  switch (e->type) {
    case VIDE:
      status = 0;
      break;
    case SIMPLE: {
      if (!strcmp(e->arguments[0], "echo")) {
        for (uint i = 1; e->arguments[i] != NULL; i++)
          printf("%s ", e->arguments[i]);
        printf("\e[D\n");
        status = 0;
      } else if (!strcmp(e->arguments[0], "exit")) {
        exit(EXIT_SUCCESS);
      } else if (!strcmp(e->arguments[0], "cd")) {
        if (!e->arguments[1]) {
          fprintf(stderr,
                  "cd: path argument required\n"
                  "usage: cd path [arguments]\n");
          status = 1;
          break;
        }
        status = chdir(e->arguments[1]);
      } else if (!strcmp(e->arguments[0], "source")) {  // don't rally work
        if (!e->arguments[1]) {
          fprintf(stderr,
                  "source: filename argument required\n"
                  "usage: source filename [arguments]\n");
          status = 1;
          break;
        }

        int fd = open(e->arguments[1], O_RDONLY);
        if (fd == -1) {
          fprintf(stderr, "Cannot open %s file\n", e->arguments[1]);
          status = 1;
          break;
        }

        int p[2];
        if (pipe(p)) {
          fprintf(stderr, "Cannot create pipe\n");
          close(fd);
          status = 1;
          break;
        }

        int pid;
        if (!(pid = fork())) {
          close(p[1]);               // close pipe in
          dup2(p[0], STDIN_FILENO);  // redirect stdin
          close(p[0]);               // close pipe out
          return EXIT_SUCCESS;
        }
        close(p[0]);  // close pipe out
        char buffer[1024];
        int l = 0;
        do {
          l = read(fd, buffer, 1024);
          if (l) {
            int n = write(p[1], buffer, l);
          }
        } while (l);
        char exit[6] = "\nexit";
        int n = write(p[1], exit, sizeof(exit));
        close(fd);    // close file descriptor
        close(p[1]);  // close pipe in

        int exitStaus;
        waitpid(pid, &exitStaus, 0);
        status = WEXITSTATUS(exitStaus);
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
      evaluer_exprL(e->gauche);
      status = evaluer_exprL(e->droite);
      break;
    }
    case SEQUENCE_ET: {
      status = evaluer_exprL(e->gauche);
      if (!status) status = evaluer_exprL(e->droite);
      break;
    }
    case SEQUENCE_OU: {
      status = evaluer_exprL(e->gauche);
      if (status) status = evaluer_exprL(e->droite);
      break;
    }
    case BG: {
      if (!fork()) { // FIXME: idk
        sigaction(SIGCHLD, &oldSa, NULL);  // Disable SIGINT handler REVIEW: remove ?
        int retStatus = evaluer_exprL(e->gauche);
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

      int oldFD = dup(STDOUT_FILENO);     // Save stdout
      dup2(fd, STDOUT_FILENO);            // change stdout with file descriptor
      close(fd);                          // close file descriptor
      status = evaluer_exprL(e->gauche);  // Exec left tree
      dup2(oldFD, STDOUT_FILENO);         // Reset stdout
      break;
    }
    case REDIRECTION_I: {
      int fd = open(e->arguments[0], O_RDONLY);
      if (fd == -1) {
        fprintf(stderr, "Cannot open %s file\n", e->arguments[0]);
        status = 1;
        break;
      }

      int oldFD = dup(STDIN_FILENO);      // Save stdin
      dup2(fd, STDIN_FILENO);             // change stdin with file descriptor
      close(fd);                          // close file descriptor
      status = evaluer_exprL(e->gauche);  // Exec left tree
      dup2(oldFD, STDIN_FILENO);          // Reset stdin
      break;
    }
    case REDIRECTION_A: {
      int fd = open(e->arguments[0], O_WRONLY | O_CREAT, 0666);
      if (fd == -1) {
        fprintf(stderr, "Cannot create %s file\n", e->arguments[0]);
        status = 1;
        break;
      }

      lseek(fd, 0, SEEK_END);  // Go to end of file

      int oldFD = dup(STDOUT_FILENO);     // Save stdout
      dup2(fd, STDOUT_FILENO);            // change stdout with file descriptor
      close(fd);                          // close file descriptor
      status = evaluer_exprL(e->gauche);  // Exec left tree
      dup2(oldFD, STDOUT_FILENO);         // Reset stdout
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

      int oldFD = dup(STDERR_FILENO);     // Save stderr
      dup2(fd, STDERR_FILENO);            // change stderr with file descriptor
      close(fd);                          // close file descriptor
      status = evaluer_exprL(e->gauche);  // Exec left tree
      dup2(oldFD, STDERR_FILENO);         // Reset stderr
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

      int oldStdoutFD = dup(STDOUT_FILENO);  // Save stdout
      int oldStderrFD = dup(STDERR_FILENO);  // Save stderr
      dup2(fd, STDOUT_FILENO);            // change stdout with file descriptor
      dup2(fd, STDERR_FILENO);            // change stderr with file descriptor
      close(fd);                          // close file descriptor
      status = evaluer_exprL(e->gauche);  // Exec left tree
      dup2(oldStdoutFD, STDOUT_FILENO);   // Reset stdout
      dup2(oldStderrFD, STDERR_FILENO);   // Reset stderr
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
        exit(evaluer_exprL(e->gauche));
      } else {
        int pidRight = fork();
        if (!pidRight) {
          // close pipe write
          close(pipe1[1]);
          // redirect pipe read to right cmd stdin
          dup2(pipe1[0], STDIN_FILENO);
          // right cmd
          exit(evaluer_exprL(e->droite));
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

  return status;
}

int evaluer_expr(Expression* e) {
  // Process backgrand process
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sa.sa_handler = childSigHandler;
  sigaction(SIGCHLD, &sa, &oldSa);

  sigaddset(&sa.sa_mask, SIGCHLD);            // add to mask
  sigprocmask(SIG_BLOCK, &sa.sa_mask, NULL);  // block SIGCHLD

  evaluer_exprL(e);

  sigprocmask(SIG_UNBLOCK, &sa.sa_mask, NULL);  // unblock SIGCHLD
  sigprocmask(SIG_BLOCK, &sa.sa_mask, NULL);    // reblock SIGCHLD
}