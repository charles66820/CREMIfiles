#ifdef CHANGED
#include "syscall.h"

void f (void *arg) {
    char c = *((char*) arg);
    PutChar(c);
    PutChar('\n');
    ThreadExit(); // TEST: error without this syscall
}

int main () {
    char c = 'a';
    void * arg = &c;
    PutString("Use console before thread :\n");
    // Creation of a new thread
    int newThread = ThreadCreate(*f, arg);
    PutString("Try to use console in parall of the new thread\n");
    // Wait other thread
    // for(;;) ;
    ThreadExit(); // TEST: with an without this syscall

    return 0;
}


#endif // CHANGED