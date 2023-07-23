#include <unistd.h>
// #include <stdio.h>

void main() {
  // toto:
  //   *(long*)sbrk(8) = 2;
  //   goto toto;

  //   register void *p;
  // toto:
  //   p = syscall(12, 0);
  //   syscall(12, p + 8);
  //   *(long*)p = 2;
  //   goto toto;

  // register long edi asm("edi") = 0;
  // register long esi asm("esi") = 0;
  // register long eax asm("eax") = 0;
  // register long rbx asm("rbx") = 0;
  // register long rax asm("rax") = 0;
  // register long rsi asm("rsi") = 0;

  asm("loop:\n"
      // p = syscall(12, 0);
      "movl $12, %edi\n"  // 12 == __NR_brk
      "movl $0, %esi\n"   // 0
      "movl $0, %eax\n"   // p <- 0
      "syscall\n"
      // "cltq\n"
      "movq %rax, %rbx\n"  // save rax
      // syscall(12, p + 8);
      "movl $12, %edi\n"      // 12 == __NR_brk
      "leaq 8(%rbx), %rax\n"  // rax <- p + 8
      "movq %rax, %rsi\n"     // rax
      "movl $0, %eax\n"       // p <- 0
      "syscall\n"
      // *(long*)p = 2;
      // "movq $2, (%rbx)\n"
      // "jmp loop\n"
  );

  // printf("edi= %d\nesi= %p\nrsi= %p\neax= %p\nrax= %p\nrbx= %p\n", edi, esi,
  // rsi, eax, rax, rbx);

  // register int    syscall_no  asm("rax") = 1;
  // register int    arg1        asm("rdi") = 1;
  // register char*  arg2        asm("rsi") = "hello, world!\n";
  // register int    arg3        asm("rdx") = 14;
  // asm("syscall");
}