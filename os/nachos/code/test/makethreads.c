#ifdef CHANGED
#include "syscall.h"

void f (void *arg) {
    char c = *((char*) arg);
    // PutChar(c);
    ThreadExit();
}

int main () {
    char c = 'a';
    void * arg = &c;
    // Creation of a new thread
    PutString("test\n");
    int newThread = ThreadCreate(*f, arg);
    // PutString("truc");
    // Wait other thread
    for(;;) ;
}


#endif // CHANGED