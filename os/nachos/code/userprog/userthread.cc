#include "userthread.h"

#ifdef CHANGED
#include "system.h"
#include <malloc.h>

static void StartUserThread(void *args) {
  DEBUG('t', "StartUserThread invoked\n");

  int f = ((int *)args)[0];
  int arg = ((int *)args)[1];

  // Init registers
  // Reset register value
  int i;
  for (i = 0; i < NumTotalRegs; i++) machine->WriteRegister(i, 0);

  // Init thread stack
  currentThread->space->AllocateUserStack(); //works for only one thread !

  // Initial program counter to user thread function
  machine->WriteRegister(PCReg, f);

  // Add arg to registers
  machine->WriteRegister(4, arg);

  // Need to also tell MIPS where next instruction is, because
  // of branch delay possibility
  machine->WriteRegister(NextPCReg, machine->ReadRegister(PCReg) + 4);

  free(args);

  // Start execution of the thread
  machine->Run();

}

int do_ThreadCreate(int f, int arg) {
  DEBUG('t', "Create a thread\n");

  // Create the new thread
  Thread *newThread = new Thread("forked thread");
  newThread->space = currentThread->space;  // define as same memory space

  // Start the new thread
  int *args = (int *) malloc(2 * sizeof(int));
  if (args == NULL) {
    perror("Not enough memory");
    return 0;
  }
  args[0] = f;
  args[1] = arg;
  newThread->Start(StartUserThread, (void *)args);
  return 1;
}

void do_ThreadExit() {
  // TODO: update thread stack bitmap to free thread stack for other new thread
  currentThread->Finish();
}

#endif  // CHANGED