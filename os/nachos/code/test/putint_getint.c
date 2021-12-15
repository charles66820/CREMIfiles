#ifdef CHANGED
#include "syscall.h"

int main() {
    int i;
    PutString("First int :\n");
    PutInt(6);
    PutString("\nSecond int :\n");
    PutInt(-6);
    PutString("\nThird int :\n");
    PutInt(2147483647);
    PutString("\nFourth int :\n");
    PutInt(-2147483648);
    PutString("\n Put an int :\n");
    GetInt(&i);
    PutInt(i);
    PutChar('\n');
}

#endif // CHANGED