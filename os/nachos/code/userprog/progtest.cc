// progtest.cc
//      Test routines for demonstrating that Nachos can load
//      a user program and execute it.
//
//      Also, routines for testing the Console hardware device.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation
// of liability and disclaimer of warranty provisions.

#include "addrspace.h"
#include "console.h"
#include "consoledriver.h"
#include "copyright.h"
#include "synch.h"
#include "system.h"

//----------------------------------------------------------------------
// StartProcess
//      Run a user program.  Open the executable, load it into
//      memory, and jump to it.
//----------------------------------------------------------------------

void StartProcess(char *filename) {
  OpenFile *executable = fileSystem->Open(filename);
  AddrSpace *space;

  if (executable == NULL) {
    SetColor(stdout, ColorRed);
    SetBold(stdout);
    printf("Unable to open file %s\n", filename);
    ClearColor(stdout);
    return;
  }
  space = new AddrSpace(executable);
  currentThread->space = space;
#ifdef CHANGED
  int ret = currentThread->space->AllocateUserStack(currentThread);
  ASSERT(ret != -1); // Not possible
#endif  // CHANGED

  delete executable;  // close file

  space->InitRegisters();  // set the initial register values
  space->RestoreState();   // load page table register

  machine->DumpMem("memory.svg");
  machine->Run();  // jump to the user progam
  ASSERT(FALSE);   // machine->Run never returns;
                   // the address space exits
                   // by doing the syscall "exit"
}

// Data structures needed for the console test.  Threads making
// I/O requests wait on a Semaphore to delay until the I/O completes.

static Console *console;
static Semaphore *readAvail;
static Semaphore *writeDone;

//----------------------------------------------------------------------
// ConsoleInterruptHandlers
//      Wake up the thread that requested the I/O.
//----------------------------------------------------------------------

static void ReadAvailHandler(void *arg) {
  (void)arg;
  readAvail->V();
}
static void WriteDoneHandler(void *arg) {
  (void)arg;
  writeDone->V();
}

//----------------------------------------------------------------------
// ConsoleTest
//      Test the console by echoing characters typed at the input onto
//      the output.  Stop when the user types a 'q' or '^d'.
//----------------------------------------------------------------------

void ConsoleTest(const char *in, const char *out) {
  char ch;

  readAvail = new Semaphore("read avail", 0);
  writeDone = new Semaphore("write done", 0);
  console = new Console(in, out, ReadAvailHandler, WriteDoneHandler, NULL);

  for (;;) {
    readAvail->P();  // wait for character to arrive
    ch = console->RX();
#ifdef CHANGED
    if (ch == 'q' || ch == EOF) {
      printf("Au revoir\n");
      break;  // if q, quit
    }
    console->TX('<');
    writeDone->P();  // For TD1 Action II.3

    console->TX(ch);  // echo it!
    writeDone->P();   // wait for write to finish

    console->TX('>');
    writeDone->P();  // For TD1 Action II.3
#else
    console->TX(ch);  // echo it!
    writeDone->P();   // wait for write to finish
    if (ch == 'q') {
      printf("Nothing more, bye!\n");
      break;  // if q, quit
    }
#endif
  }
  delete console;
  delete readAvail;
  delete writeDone;
}

//----------------------------------------------------------------------
// ConsoleDriverTest
//      Test the synchronous console driver by echoing characters typed at the
//      input onto the output.  Stop when the user types `^d'.
//----------------------------------------------------------------------

#ifdef CHANGED

void ConsoleDriverTest(const char *in, const char *out) {
  char ch;
  ConsoleDriver *test_consoledriver = new ConsoleDriver(in, out);
  while ((ch = test_consoledriver->GetChar()) != EOF) {
    test_consoledriver->PutChar('<');
    test_consoledriver->PutChar(ch);
    test_consoledriver->PutChar('>');
  }
  fprintf(stderr, "EOF detected in ConsoleDriver!\n");
  delete test_consoledriver;
}
#endif  // CHANGED