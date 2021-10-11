#ifdef CHANGED
#include "syscall.h"

int main() {
    int n = 6;
    char s[n];

    GetString(s, n);
    PutString(s);

    return 0;
}
#endif // CHANGED