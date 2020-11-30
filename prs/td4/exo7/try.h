#ifndef TRY_H
#define TRY_H

#include <signal.h>

int tryIt(void  (*f)(void*), void *p, int sig);
int tryBefore( void (*f)(void *), void *p, int delay);

#endif
