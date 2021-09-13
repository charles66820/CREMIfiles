// thread.cc 
//      Routines to manage threads.  There are four main operations:
//
//      Start -- create a thread to run a procedure concurrently
//               with the caller (this is done in two steps -- first
//               allocate the Thread object, then call Start on it)
//      Finish -- called when the forked procedure finishes, to clean up
//      Yield -- relinquish control over the CPU to another ready thread
//      Sleep -- relinquish control over the CPU, but thread is now blocked.
//              In other words, it will not run again, until explicitly 
//              put back on the ready queue.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "thread.h"
#include "switch.h"
#include "synch.h"
#include "system.h"
#include "valgrind.h"
#ifdef USER_PROGRAM
#include "machine.h"
#endif
#ifdef __SANITIZE_ADDRESS__
#include <pthread.h>
#include <sanitizer/asan_interface.h>
#endif


#define STACK_FENCEPOST 0xdeadbeef	// this is put at the top of the
					// execution stack, for detecting 
					// stack overflows

//----------------------------------------------------------------------
// ThreadList
//      List of all threads, for debugging
//----------------------------------------------------------------------
List ThreadList;

//----------------------------------------------------------------------
// Thread::Thread
//      Initialize a thread control block, so that we can then call
//      Thread::Start.
//
//      "threadName" is an arbitrary string, useful for debugging.
//----------------------------------------------------------------------

Thread::Thread (const char *threadName)
{
    name = threadName;
    stackTop = NULL;
    stack = NULL;

    valgrind_id = 0;

#ifdef __SANITIZE_ADDRESS__
    fake_stack = NULL;
#endif
    stack_size = 0;
    main_stack = 0;

    status = JUST_CREATED;
#ifdef USER_PROGRAM
    space = NULL;

    // must be explicitly set to 0 since when Enabling interrupts,
    // DelayedLoad is called !!!
    userRegisters[LoadReg] = 0;
    userRegisters[LoadValueReg] = 0;
#endif
    ThreadList.Append(this);
}

//----------------------------------------------------------------------
// Thread::SetMain
//      Configures the thread as representing the main thread, i.e. the thread
//      initially created by the OS, with an OS-provided stack.
//----------------------------------------------------------------------

void
Thread::SetMain (void)
{
#ifdef __SANITIZE_ADDRESS__
    pthread_attr_t attr;
    void *addr;

    pthread_getattr_np (pthread_self (), &attr);
    pthread_attr_getstack (&attr, &addr, &stack_size);
    pthread_attr_destroy (&attr);
    stack = (unsigned long *) addr;
#endif

    main_stack = 1;

    setStatus (RUNNING);
}

//----------------------------------------------------------------------
// Thread::~Thread
//      De-allocate a thread.
//
//      NOTE: the current thread *cannot* delete itself directly,
//      since it is still running on the stack that we need to delete.
//
//      NOTE: if this is the main thread, we can't delete the stack
//      because we didn't allocate it -- we got it automatically
//      as part of starting up Nachos.
//----------------------------------------------------------------------

Thread::~Thread ()
{
    DEBUG ('t', "Deleting thread %p \"%s\"\n", this, name);

    ASSERT (this != currentThread);
    if (!main_stack && stack != NULL) {
	DeallocBoundedArray ((char *) stack, StackSize * sizeof (unsigned long));
	VALGRIND_STACK_DEREGISTER (valgrind_id);
	stack = NULL;
    }
    ThreadList.Remove(this);
}

//----------------------------------------------------------------------
// ThrashStack
//      Fill the stack with bogus values, to avoid getting 0 values only by luck
//----------------------------------------------------------------------
void
ThrashStack(void)
{
    char c[StackSize];
    memset(c, 0x02, sizeof(c));
}

//----------------------------------------------------------------------
// Thread::Start
//      Invoke (*func)(arg), allowing caller and callee to execute 
//      concurrently.
//
//      NOTE: although our definition allows only a single integer argument
//      to be passed to the procedure, it is possible to pass multiple
//      arguments by making them fields of a structure, and passing a pointer
//      to the structure as "arg".
//
//      Implemented as the following steps:
//              1. Allocate a stack
//              2. Initialize the stack so that a call to SWITCH will
//              cause it to run the procedure
//              3. Put the thread on the ready queue
//      
//      "func" is the procedure to run concurrently.
//      "arg" is a single argument to be passed to the procedure.
//----------------------------------------------------------------------

void
Thread::Start (VoidFunctionPtr func, void *arg)
{
    DEBUG ('t', "Starting thread %p \"%s\" with func = %p, arg = %p\n",
	   this, name, func, arg);

    ASSERT(status == JUST_CREATED);
    StackAllocate (func, arg);

    IntStatus oldLevel = interrupt->SetLevel (IntOff);
    scheduler->ReadyToRun (this);	// ReadyToRun assumes that interrupts 
    // are disabled!
    (void) interrupt->SetLevel (oldLevel);
}

//----------------------------------------------------------------------
// Thread::CheckOverflow
//      Check a thread's stack to see if it has overrun the space
//      that has been allocated for it.  If we had a smarter compiler,
//      we wouldn't need to worry about this, but we don't.
//
//      NOTE: Nachos will not catch all stack overflow conditions.
//      In other words, your program may still crash because of an overflow.
//
//      If you get bizarre results (such as seg faults where there is no code)
//      then you *may* need to increase the stack size.  You can avoid stack
//      overflows by not putting large data structures on the stack.
//      Don't do this: void foo() { int bigArray[10000]; ... }
//----------------------------------------------------------------------

void
Thread::CheckOverflow ()
{
    if (!main_stack && stack != NULL)
#ifdef HOST_SNAKE		// Stacks grow upward on the Snakes
	ASSERT (stack[StackSize - 1] == STACK_FENCEPOST);
#else
	ASSERT (*stack == (unsigned long) STACK_FENCEPOST);
#endif
}

//----------------------------------------------------------------------
// Thread::Finish
//      Called by ThreadRoot when a thread is done executing the 
//      forked procedure.
//
//      NOTE: we don't immediately de-allocate the thread data structure 
//      or the execution stack, because we're still running in the thread 
//      and we're still on the stack!  Instead, we set "threadToBeDestroyed", 
//      so that Scheduler::Run() will call the destructor, once we're
//      running in the context of a different thread.
//
//      NOTE: we disable interrupts, so that we don't get a time slice 
//      between setting threadToBeDestroyed, and going to sleep.
//----------------------------------------------------------------------

//
void
Thread::Finish ()
{
    (void) interrupt->SetLevel (IntOff);
    ASSERT (this == currentThread);

    DEBUG ('t', "Finishing thread %p \"%s\"\n", this, getName ());

    // LB: Be careful to guarantee that no thread to be destroyed 
    // is ever lost 
    ASSERT (threadToBeDestroyed == NULL);
    // End of addition 

    threadToBeDestroyed = currentThread;
    Sleep ();			// invokes SWITCH
    // not reached
}

//----------------------------------------------------------------------
// Thread::Yield
//      Relinquish the CPU if any other thread is ready to run.
//      If so, put the thread on the end of the ready list, so that
//      it will eventually be re-scheduled.
//
//      NOTE: returns immediately if no other thread on the ready queue.
//      Otherwise returns when the thread eventually works its way
//      to the front of the ready list and gets re-scheduled.
//
//      NOTE: we disable interrupts, so that looking at the thread
//      on the front of the ready list, and switching to it, can be done
//      atomically.  On return, we re-set the interrupt level to its
//      original state, in case we are called with interrupts disabled. 
//
//      Similar to Thread::Sleep(), but a little different.
//----------------------------------------------------------------------

void
Thread::Yield ()
{
    Thread *nextThread;
    IntStatus oldLevel = interrupt->SetLevel (IntOff);

    ASSERT (this == currentThread);

    DEBUG ('t', "Yielding thread %p \"%s\"\n", this, getName ());

    nextThread = scheduler->FindNextToRun ();
    if (nextThread != NULL)
      {
	  scheduler->ReadyToRun (this);
	  scheduler->Run (nextThread);
      }
    (void) interrupt->SetLevel (oldLevel);
}

//----------------------------------------------------------------------
// Thread::Sleep
//      Relinquish the CPU, because the current thread is blocked
//      waiting on a synchronization variable (Semaphore, Lock, or Condition).
//      Eventually, some thread will wake this thread up, and put it
//      back on the ready queue, so that it can be re-scheduled.
//
//      NOTE: if there are no threads on the ready queue, that means
//      we have no thread to run.  "Interrupt::Idle" is called
//      to signify that we should idle the CPU until the next I/O interrupt
//      occurs (the only thing that could cause a thread to become
//      ready to run).
//
//      NOTE: we assume interrupts are already disabled, because it
//      is called from the synchronization routines which must
//      disable interrupts for atomicity.   We need interrupts off 
//      so that there can't be a time slice between pulling the first thread
//      off the ready list, and switching to it.
//----------------------------------------------------------------------
void
Thread::Sleep ()
{
    Thread *nextThread;

    ASSERT (this == currentThread);
    ASSERT (interrupt->getLevel () == IntOff);

    DEBUG ('t', "Sleeping thread %p \"%s\"\n", this, getName ());

    status = BLOCKED;
    while ((nextThread = scheduler->FindNextToRun ()) == NULL)
	interrupt->Idle ();	// no one to run, wait for an interrupt

    scheduler->Run (nextThread);	// returns when we've been signalled
}

//----------------------------------------------------------------------
// ThreadFinish, InterruptEnable, ThreadPrint
//      Dummy functions because C++ does not allow a pointer to a member
//      function.  So in order to do this, we create a dummy C function
//      (which we can pass a pointer to), that then simply calls the 
//      member function.
//----------------------------------------------------------------------

static void
ThreadFinish ()
{
    currentThread->Finish ();
}
static void
InterruptEnable ()
{
    interrupt->Enable ();
}

// LB: This function has to be called on starting  a new thread to set
// up the pagetable correctly. This was missing from the original
// version. Credits to Clement Menier for finding this bug!

void 
SetupThreadState ()
{
#ifdef __SANITIZE_ADDRESS__
  __sanitizer_finish_switch_fiber (currentThread->fake_stack, NULL, NULL);
#endif

  // LB: Similar to the second part of Scheduler::Run. This has to be
  // done each time a thread is scheduled, either by SWITCH, or by
  // getting created.
  
  if (threadToBeDestroyed != NULL)
    {
      delete threadToBeDestroyed;
      threadToBeDestroyed = NULL;
    }
  
#ifdef USER_PROGRAM

  // LB: Now, we have to simulate the RestoreUserState/RestoreState
  // part of Scheduler::Run

  // Be very careful! We have no information about the thread which is
  // currently running at the time this function is called. Actually,
  // there is no reason why the running thread should have the same
  // pageTable as the thread just being created.

  // This is definitely the case as soon as several *processes* are
  // running together.

  if (currentThread->space != NULL)
    {				// if there is an address space
      // LB: Actually, the user state is void at that time. Keep this
      // action for consistency with the Scheduler::Run function
      currentThread->RestoreUserState ();	// to restore, do it.
      currentThread->space->RestoreState ();
    }

#endif // USER_PROGRAM

  // LB: The default level for interrupts is IntOn.
  InterruptEnable (); 

}

// End of addition

void
ThreadPrint (void *arg)
{
    Thread *t = (Thread *) arg;
    t->Print ();
}

//----------------------------------------------------------------------
// Thread::StackAllocate
//      Allocate and initialize a kernel execution stack.  The stack is
//      initialized with an initial stack frame for ThreadRoot, which:
//              enables interrupts
//              calls (*func)(arg)
//              calls Thread::Finish
//
//      "func" is the procedure to be forked
//      "arg" is the parameter to be passed to the procedure
//----------------------------------------------------------------------

void
Thread::StackAllocate (VoidFunctionPtr func, void *arg)
{
    stack = (unsigned long *) AllocBoundedArray (StackSize * sizeof (unsigned long));
    stack_size = StackSize * sizeof (unsigned long);
    valgrind_id = VALGRIND_STACK_REGISTER(stack, stack + StackSize);

#ifdef HOST_SNAKE
    // HP stack works from low addresses to high addresses
    stackTop = stack + 16;	// HP requires 64-byte frame marker
    stack[StackSize - 1] = STACK_FENCEPOST;
#else
    // other archs stack works from high addresses to low addresses
#ifdef HOST_SPARC
    // SPARC stack must contains at least 1 activation record to start with.
    stackTop = stack + StackSize - 96;
#elif defined(HOST_PPC)
    stackTop = stack + StackSize - 192;
#elif defined(HOST_i386)
    stackTop = stack + StackSize - 4;	// -4 for the return address
#elif defined(HOST_x86_64)
    stackTop = stack + StackSize - 16;	// room for the return address, and align for SSE*
#elif defined(HOST_MIPS)
    stackTop = stack + StackSize;	// no special need
#else
#error uh??
#endif
    *stack = STACK_FENCEPOST;
#endif // stack direction

    memset(&machineState, 0, sizeof(machineState));

    machineState[PCState] = (unsigned long) ThreadRoot;

    // LB: It is not sufficient to enable interrupts!
    // A more complex function has to be called here...
    // machineState[StartupPCState] = (int) InterruptEnable;
    machineState[StartupPCState] = (unsigned long) SetupThreadState;
    // End of modification
    
    machineState[InitialPCState] = (unsigned long) func;
    machineState[InitialArgState] = (unsigned long) arg;
    machineState[WhenDonePCState] = (unsigned long) ThreadFinish;
}

#ifdef USER_PROGRAM
#include "machine.h"

//----------------------------------------------------------------------
// Thread::SaveUserState
//      Save the CPU state of a user program on a context switch.
//
//      Note that a user program thread has *two* sets of CPU registers -- 
//      one for its state while executing user code, one for its state 
//      while executing kernel code.  This routine saves the former.
//----------------------------------------------------------------------

void
Thread::SaveUserState ()
{
    for (int i = 0; i < NumTotalRegs; i++)
	userRegisters[i] = machine->ReadRegister (i);
}

//----------------------------------------------------------------------
// Thread::RestoreUserState
//      Restore the CPU state of a user program on a context switch.
//
//      Note that a user program thread has *two* sets of CPU registers -- 
//      one for its state while executing user code, one for its state 
//      while executing kernel code.  This routine restores the former.
//----------------------------------------------------------------------

void
Thread::RestoreUserState ()
{
    for (int i = 0; i < NumTotalRegs; i++)
	machine->WriteRegister (i, userRegisters[i]);
}

//----------------------------------------------------------------------
// Thread::DumpThreadState
//      Draw the states for this thread
//----------------------------------------------------------------------
void
Thread::DumpThreadState(FILE *output, int ptr_x, int ptr_y, unsigned virtual_x, unsigned virtual_y, unsigned blocksize)
{
    if (this == currentThread)
	machine->DumpRegs(output, ptr_x, ptr_y, virtual_x, virtual_y, blocksize);
    else
    {
	machine->DumpReg(output, userRegisters[PCReg], "PC", "#000000", ptr_x, ptr_y, virtual_x, virtual_y, blocksize);
	machine->DumpReg(output, userRegisters[StackReg], "SP", "#000000", ptr_x, ptr_y, virtual_x, virtual_y, blocksize);
    }
}

//----------------------------------------------------------------------
// DumpThreadsState
//      Draw the states for threads belonging to a given address space
//----------------------------------------------------------------------
void
DumpThreadsState(FILE *output, AddrSpace *space, unsigned ptr_x, unsigned virtual_x, unsigned virtual_y, unsigned blocksize)
{
    ListElement *element;
    ptr_x += 4*blocksize;
    unsigned nthreads = ThreadList.Length();

    unsigned y_offset;
    unsigned ptr_y;
    if (nthreads == 1)
      {
	y_offset = 0;
	ptr_y = virtual_y;
      }
    else
      {
	y_offset = (blocksize/2) / (nthreads-1);
	ptr_y = virtual_y - (blocksize/4);
      }

    for (element = ThreadList.FirstElement();
	 element;
	 element = element->next)
      {
	Thread *thread = (Thread *) element->item;
	if (thread->space != space)
	    continue;

	thread->DumpThreadState(output, ptr_x, ptr_y, virtual_x, virtual_y, blocksize);

	// Offset names a bit on the left
	ptr_x -= blocksize;

	// And offset lines a bit down
	ptr_y += y_offset;
      }
}

#endif


