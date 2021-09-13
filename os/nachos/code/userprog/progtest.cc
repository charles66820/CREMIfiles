// progtest.cc 
//      Test routines for demonstrating that Nachos can load
//      a user program and execute it.  
//
//      Also, routines for testing the Console hardware device.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "system.h"
#include "console.h"
#include "addrspace.h"
#include "synch.h"

//----------------------------------------------------------------------
// StartProcess
//      Run a user program.  Open the executable, load it into
//      memory, and jump to it.
//----------------------------------------------------------------------

void
StartProcess (char *filename)
{
    OpenFile *executable = fileSystem->Open (filename);
    AddrSpace *space;

    if (executable == NULL)
      {
	  SetColor (stdout, ColorRed);
	  SetBold (stdout);
	  printf ("Unable to open file %s\n", filename);
	  ClearColor (stdout);
	  return;
      }
    space = new AddrSpace (executable);
    currentThread->space = space;

    delete executable;		// close file

    space->InitRegisters ();	// set the initial register values
    space->RestoreState ();	// load page table register

    machine->DumpMem ("memory.svg");
    machine->Run ();		// jump to the user progam
    ASSERT (FALSE);		// machine->Run never returns;
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

static void
ReadAvailHandler (void *arg)
{
    (void) arg;
    readAvail->V ();
}
static void
WriteDoneHandler (void *arg)
{
    (void) arg;
    writeDone->V ();
}

//----------------------------------------------------------------------
// ConsoleTest
//      Test the console by echoing characters typed at the input onto
//      the output.  Stop when the user types a 'q'.
//----------------------------------------------------------------------

void
ConsoleTest (const char *in, const char *out)
{
    char ch;

    readAvail = new Semaphore ("read avail", 0);
    writeDone = new Semaphore ("write done", 0);
    console = new Console (in, out, ReadAvailHandler, WriteDoneHandler, NULL);

    for (;;)
      {
	  readAvail->P ();	// wait for character to arrive
	  ch = console->RX ();
	  console->TX (ch);	// echo it!
	  writeDone->P ();	// wait for write to finish
	  if (ch == 'q') {
	      printf ("Nothing more, bye!\n");
	      break;		// if q, quit
	  }
      }
    delete console;
    delete readAvail;
    delete writeDone;
}
