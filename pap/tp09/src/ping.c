#include "mpi.h"
#include <stdio.h>
#include <sys/utsname.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#define TIME_DIFF(t1, t2) \
  ((t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec))


#define NMAX (1024*1024)
int main(int argc, char **argv)
{
  int myid;
  int numprocs;
  char buf[NMAX];
  int N = 1;
  MPI_Status status;
  //MPI_Request request;
  struct utsname name;
  struct timeval t1, t2;

  uname(&name);


#if 0
  /* Utilisation d'un verrou inter-processus pour éviter que vous vous
   * marchiez les uns sur les autres */
  int fd = open("/var/tmp/verrou-PAP", O_WRONLY|O_CREAT, 0777);
  if (fd < 0) {
    perror("open");
    exit(1);
  }
  fchmod(fd, 0777);
  if (lockf(fd, F_TLOCK, 0)) {
    fprintf(stderr,"Lock %s\n",name.nodename);
    perror("lockf");
    exit(1);
  }
#endif

  /* initialiser MPI, à faire _toujours_ en tout premier du programme */
  MPI_Init(&argc,&argv);

  /* récupérer le nombre de processus lancés */
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

  /* récupérer son propre numéro */
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  fprintf(stderr,"I'm %d/%d on %s\n",myid,numprocs,name.nodename);

  if (myid == 0)
    {
      buf[0]='i';
      gettimeofday(&t1,NULL);
      MPI_Send(buf,N,MPI_CHAR,1,0,MPI_COMM_WORLD);
      gettimeofday(&t2,NULL);
      fprintf(stderr,"envoyé %ld\n",TIME_DIFF(t1,t2));
    }
  else
    {
      gettimeofday(&t1,NULL);
      MPI_Recv(buf,N,MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
      gettimeofday(&t2,NULL);
      fprintf(stderr,"reçu %ld\n",TIME_DIFF(t1,t2));
    }

  /* finaliser MPI, à faire _toujours_ à la toute fin du programme */
  MPI_Finalize();
  return 0;
}
