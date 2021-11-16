#include "userthread.h"

#ifdef CHANGED
#include <malloc.h>

#include "syscall.h"
#include "system.h"

static void StartUserThread(void *args) {
  DEBUG('t', "StartUserThread invoked\n");

  int f = ((int *)args)[0];
  int arg = ((int *)args)[1];
  int stackIndex = ((int *)args)[2];

  free(args);

  // Init registers
  // Reset register value
  int i;
  for (i = 0; i < NumTotalRegs; i++) machine->WriteRegister(i, 0);

  // Set thread stack
  currentThread->SetStackIndex(stackIndex);
  int stackAddress = currentThread->space->NumPages() * PageSize -
                     (UserStackSize * stackIndex + 16);
  machine->WriteRegister(StackReg, stackAddress);
  DEBUG('a', "Initializing stack pointer to 0x%x\n", stackAddress);

  // Initialize program counter to user thread caller
  machine->WriteRegister(PCReg, USER_THREAD_ADDRESS);
  // Put the address of the user thread function in register 2 for the thread caller
  machine->WriteRegister(2, f);

  // Add arg to registers
  machine->WriteRegister(4, arg);

  // Need to also tell MIPS where next instruction is, because
  // of branch delay possibility
  machine->WriteRegister(NextPCReg, machine->ReadRegister(PCReg) + 4);

  // Print the svg
  machine->DumpMem("threads.svg");

  // Start execution of the thread
  machine->Run();
  ASSERT(false);
}

int do_ThreadCreate(int f, int arg) {
  DEBUG('t', "Create a thread\n");

  // Create the new thread
  Thread *newThread = new Thread("forked thread");
  newThread->space = currentThread->space;  // define as same memory space

  int stackIndex = newThread->space->AllocateUserStack();
  if (stackIndex == -1) {
    DEBUG('t', "Not enough memory to allocate the stack of the new thread !");
    delete newThread;
    return 0;
  }

  // Start the new thread
  int *args = (int *)malloc(3 * sizeof(int));
  if (args == NULL) {
    perror("Not enough memory");
    delete newThread;
    return 0;
  }
  args[0] = f;
  args[1] = arg;
  args[2] = stackIndex;
  newThread->Start(StartUserThread, (void *)args);
  return 1;
}

void do_ThreadExit() {
  currentThread->space->DeallocateUserStack();

  currentThread->Finish();
}

#endif  // CHANGED