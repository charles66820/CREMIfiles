// exception.cc
//      Entry point into the Nachos kernel from user programs.
//      There are two kinds of things that can cause control to
//      transfer back to here from user code:
//
//      syscall -- The user code explicitly requests to call a procedure
//      in the Nachos kernel.  Right now, the only function we support is
//      "Halt".
//
//      exceptions -- The user code does something that the CPU can't handle.
//      For instance, accessing memory that doesn't exist, arithmetic errors,
//      etc.
//
//      Interrupts (which can also cause control to transfer from user
//      code into the Nachos kernel) are handled elsewhere.
//
// For now, this only handles the Halt() system call.
// Everything else core dumps.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "syscall.h"
#include "system.h"

//----------------------------------------------------------------------
// UpdatePC : Increments the Program Counter register in order to resume
// the user program immediately after the "syscall" instruction.
//----------------------------------------------------------------------
static void UpdatePC() {
  int pc = machine->ReadRegister(PCReg);
  machine->WriteRegister(PrevPCReg, pc);
  pc = machine->ReadRegister(NextPCReg);
  machine->WriteRegister(PCReg, pc);
  pc += 4;
  machine->WriteRegister(NextPCReg, pc);
}

#ifdef CHANGED
/**
 * @brief Copy char from mips memory to a table give in args
 * @param from address in mips memory space
 * @param to table pointer to copy string
 * @param size nb char to copy
 * @return int length copy
 */
int copyStringFromMachine(int from, char *to, unsigned size) {
  if (size <= 0) return -1;

  uint cpt = 0;
  for (; cpt < size - 1; cpt++) {
    int tmp;
    if (machine->ReadMem(from + cpt, 1, &tmp)) {
      // copy read tmp char to `to`
      to[cpt] = tmp;
      if (tmp == '\0') return cpt;
    } else break;
  }

  to[cpt] = '\0';
  return cpt;
}
#endif  // CHANGED

//----------------------------------------------------------------------
// ExceptionHandler
//      Entry point into the Nachos kernel.  Called when a user program
//      is executing, and either does a syscall, or generates an addressing
//      or arithmetic exception.
//
//      For system calls, the following is the calling convention:
//
//      system call code -- r2
//              arg1 -- r4
//              arg2 -- r5
//              arg3 -- r6
//              arg4 -- r7
//
//      The result of the system call, if any, must be put back into r2.
//
// And don't forget to increment the pc before returning. (Or else you'll
// loop making the same system call forever!
//
//      "which" is the kind of exception.  The list of possible exceptions
//      are in machine.h.
//----------------------------------------------------------------------

void ExceptionHandler(ExceptionType which) {
  int type = machine->ReadRegister(2);
  int address = machine->registers[BadVAddrReg];

  switch (which) {
    case SyscallException: {
      switch (type) {
        case SC_Halt: {
          DEBUG('s', "Shutdown, initiated by user program.\n");
          interrupt->Powerdown();
          break;
        }
#ifdef CHANGED
        case SC_PutChar: {
          DEBUG('s', "PutChar\n");
          int c = machine->ReadRegister(4);
          consoledriver->PutChar(c);
          break;
        }
        case SC_PutString: {
          DEBUG('s', "PutString\n");
          int p = machine->ReadRegister(4);
          char s[MAX_STRING_SIZE];

          int len = 0;
          while (true) {
            len = copyStringFromMachine(p, s, MAX_STRING_SIZE);
            consoledriver->PutString(s);
            if (len < MAX_STRING_SIZE - 1) break;
            p += len;
          }
          break;
        }
#endif  // CHANGED
        default: {
          printf("Unimplemented system call %d\n", type);
          ASSERT(FALSE);
        }
      }

      // Do not forget to increment the pc before returning!
      UpdatePC();
      break;
    }

    case PageFaultException:
      if (!address) {
        printf("NULL dereference at PC %x!\n", machine->registers[PCReg]);
        ASSERT(FALSE);
      } else {
        printf("Page Fault at address %x at PC %x\n", address,
               machine->registers[PCReg]);
        ASSERT(FALSE);  // For now
      }
      break;

    case ReadOnlyException:
      printf("Read-Only at address %x at PC %x\n", address,
             machine->registers[PCReg]);
      ASSERT(FALSE);  // For now
      break;

    case BusErrorException:
      printf("Invalid physical address at address %x at PC %x\n", address,
             machine->registers[PCReg]);
      ASSERT(FALSE);  // For now
      break;

    case AddressErrorException:
      printf("Invalid address %x at PC %x\n", address,
             machine->registers[PCReg]);
      ASSERT(FALSE);  // For now
      break;

    case OverflowException:
      printf("Overflow at PC %x\n", machine->registers[PCReg]);
      ASSERT(FALSE);  // For now
      break;

    case IllegalInstrException:
      printf("Illegal instruction at PC %x\n", machine->registers[PCReg]);
      ASSERT(FALSE);  // For now
      break;

    default:
      printf("Unexpected user mode exception %d %d %x at PC %x\n", which, type,
             address, machine->registers[PCReg]);
      ASSERT(FALSE);
      break;
  }
}
