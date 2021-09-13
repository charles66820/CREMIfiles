/* sort.c 
 *    Test program to sort a large number of integers through BubbleSort.
 *
 *    Intention is to stress virtual memory system.
 *
 *    Ideally, we could read the unsorted array off of the file system,
 *	and store the result back to the file system!
 */

#include "syscall.h"

#define N 1024
int A[N];			/* size of physical memory; with code, we'll run out of space! */

int
main ()
{
    int i, j, tmp;

    /* first initialize the array, in reverse sorted order */
    for (i = 0; i < N; i++)
	A[i] = N - i;

    /* then sort! */
    for (i = 1; i < N; i++)
	for (j = 0; j < N - i; j++)
	    if (A[j] > A[j + 1])
	      {			/* out of order -> need to swap ! */
		  tmp = A[j];
		  A[j] = A[j + 1];
		  A[j + 1] = tmp;
	      }
    Exit (A[0]);		/* and then we're done -- should be 0! */
}
