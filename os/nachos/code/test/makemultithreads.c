#ifdef CHANGED
#include "syscall.h"

void f (void *arg) {
    volatile char c = (char)(int)arg;
    PutString("Thread : ");
    PutChar(c);
    PutChar('!\n');
    ThreadExit(); // TEST: error without this syscall
}

int main () {
    char c = 'a';
    void *arg = (void *)(int)c;
    int i;

    PutString("Main thread !\n");

    // Creation of multiple threads
    for (i = 0; i < 26; i++) {
        int newThread = ThreadCreate(*f, arg);
        c = c + 1;
        arg = (void *)(int)c;
    }

    PutString("Main thread finish\n");
    ThreadExit(); // TEST: with and without this syscall

    return 0;
}


#endif // CHANGED