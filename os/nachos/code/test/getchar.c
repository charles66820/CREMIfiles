#ifdef CHANGED
#include "syscall.h"

int main() {
    char c;
    do {
        c = GetChar();
        if (c != -1) PutChar(c);
    } while (c != '\0' && c != -1);

    return 0;
}
#endif // CHANGED