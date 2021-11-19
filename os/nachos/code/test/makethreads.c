#ifdef CHANGED
#include "syscall.h"

void f (void *arg) {
    char c = *((char*) arg);
    PutChar(c);
    PutChar('\n');
    // ThreadExit(); // TEST: error without this syscall
}

int main () {
    char c = 'a';
    void * arg = &c;
    int newThread;

    PutString("Use console before thread :\n");
    // Creation of a new thread
    newThread = ThreadCreate(*f, arg);
    PutString("Try to use console in parall of the new thread\n");
    // Wait other thread
    // for(;;) ;
    ThreadExit(); // TEST: with an without this syscall

    return 0;
}

// To run it: `cd code` and `userprog/nachos -d s -rs 12 -x test/makethreads`

#endif // CHANGED