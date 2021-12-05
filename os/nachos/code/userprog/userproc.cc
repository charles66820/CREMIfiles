#ifdef CHANGED

#include "userproc.h"

#include "system.h"

static void StartUserProc(void *args) {
  DEBUG('t', "StartUserProc invoked\n");

  (void)args;

  // Init registers
  currentThread->space->InitRegisters();
  currentThread->space->RestoreState();   // load page table register

  // Print the svg
  machine->DumpMem("newProc.svg");

  // Start execution of the thread
  machine->Run();
  ASSERT(false);
}

int do_ProcCreate(char *filename) {
  DEBUG('t', "Create a process\n");

  // Create the new address space
  OpenFile *executable = fileSystem->Open(filename);
  if (executable == NULL) {
    SetColor(stdout, ColorRed);
    SetBold(stdout);
    printf("Unable to open file %s\n", filename);
    ClearColor(stdout);
    return 0;
  }

  AddrSpace *space = new AddrSpace(executable);
  delete executable;  // close file

  // Create the new thread
  Thread *newThreadProc = new Thread("process");
  newThreadProc->space = space;
  int stackIndex = newThreadProc->space->AllocateUserStack(newThreadProc);
  ASSERT(stackIndex != -1); // Not possible
  newThreadProc->SetStackIndex(stackIndex); // Defind as the first stack

  processMutex->Acquire();
  nbProcess += 1;
  processMutex->Release();

  newThreadProc->Start(StartUserProc, NULL);

  return 1;
}

void do_ProcExit() {
  processMutex->Acquire();
  nbProcess -= 1;
  processMutex->Release();

  // When is the last process we do a powerdown interruption
  if (nbProcess == 0) {
    interrupt->Powerdown();
    return;
  }

  // Close all proc threads
  currentThread->space->CloseAllUserThread();

  // remove the current process
  delete currentThread->space;

  currentThread->Finish();
}

#endif  // CHANGED