#ifdef CHANGED

#include "userproc.h"

#include "system.h"

static void StartUserProc(void *args) {
  DEBUG('t', "StartUserProc invoked\n");

  // Init registers
  currentThread->space->InitRegisters();

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
    return;
  }

  AddrSpace *space = new AddrSpace(executable);
  delete executable;  // close file

  // Create the new thread
  Thread *newThreadProc = new Thread("process");
  newThreadProc->space = space;

  // TODO: setMain ???

  newThreadProc->Start(StartUserProc, NULL);

  return 1;
}

#endif  // CHANGED