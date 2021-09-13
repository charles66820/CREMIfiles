/* switch.h
 *	Definitions needed for implementing context switching.
 *
 *	Context switching is inherently machine dependent, since
 *	the registers to be saved, how to set up an initial
 *	call frame, etc, are all specific to a processor architecture.
 *
 * 	This file currently supports the DEC MIPS and SUN SPARC architectures.
 */

/*
 Copyright (c) 1992-1993 The Regents of the University of California.
 All rights reserved.  See copyright.h for copyright notice and limitation 
 of liability and disclaimer of warranty provisions.
 */

#ifndef SWITCH_H
#define SWITCH_H

#include "copyright.h"

#ifdef HOST_MIPS

/* Registers that must be saved during a context switch. 
 * These are the offsets from the beginning of the Thread object, 
 * in bytes, used in switch.s
 */
#define SP 0
#define S0 4
#define S1 8
#define S2 12
#define S3 16
#define S4 20
#define S5 24
#define S6 28
#define S7 32
#define FP 36
#define PC 40

/* To fork a thread, we set up its saved register state, so that
 * when we switch to the thread, it will start running in ThreadRoot.
 *
 * The following are the initial registers we need to set up to
 * pass values into ThreadRoot (for instance, containing the procedure
 * for the thread to run).  The first set is the registers as used
 * by ThreadRoot; the second set is the locations for these initial
 * values in the Thread object -- used in Thread::AllocateStack().
 */

#define InitialPC	s0
#define InitialArg	s1
#define WhenDonePC	s2
#define StartupPC	s3

#define PCState		(PC/4-1)
#define FPState		(FP/4-1)
#define InitialPCState	(S0/4-1)
#define InitialArgState	(S1/4-1)
#define WhenDonePCState	(S2/4-1)
#define StartupPCState	(S3/4-1)

#endif // HOST_MIPS

#ifdef HOST_SPARC

/* Registers that must be saved during a context switch.  See comment above. */
#define I0 4
#define I1 8
#define I2 12
#define I3 16
#define I4 20
#define I5 24
#define I6 28
#define I7 32

/* Aliases used for clearing code.  */
#define FP I6
#define PC I7

/* Registers for ThreadRoot.  See comment above. */
#define InitialPC       %o0
#define InitialArg      %o1
#define WhenDonePC      %o2
#define StartupPC       %o3

#define PCState         (PC/4-1)
#define InitialPCState  (I0/4-1)
#define InitialArgState (I1/4-1)
#define WhenDonePCState (I2/4-1)
#define StartupPCState  (I3/4-1)
#endif // HOST_SPARC

#ifdef HOST_SNAKE

/* Registers that must be saved during a context switch.  See comment above. */
#define   SP   0
#define   S0   4
#define   S1   8
#define   S2   12
#define   S3   16
#define   S4   20
#define   S5   24
#define   S6   28
#define   S7   32
#define   S8   36
#define   S9   40
#define   S10  44
#define   S11  48
#define   S12  52
#define   S13  56
#define   S14  60
#define   S15  64
#define   PC   68

/* Registers for ThreadRoot.  See comment above. */
#define InitialPC       %r3	/* S0 */
#define InitialArg      %r4
#define WhenDonePC      %r5
#define StartupPC       %r6

#define PCState         (PC/4-1)
#define InitialPCState  (S0/4-1)
#define InitialArgState (S1/4-1)
#define WhenDonePCState (S2/4-1)
#define StartupPCState  (S3/4-1)
#endif // HOST_SNAKE

#ifdef HOST_PPC

#define   R0   4
#define   R1   0 // SP must be the first field of the Thread struct!
#define   R2   8
#define   R3   12
#define   R4   16
#define   R5   20
#define   R6   24
#define   R7   28
#define   R8   32
#define   R9   36
#define   R10  40
#define   R11  44
#define   R12  48
#define   R13  52
#define   R14  56
#define   R15  60
#define   R16  64
#define   R17  68
#define   R18  72

/* Registers for ThreadRoot.  See comment above. */
#define InitialPC       %r5
#define InitialArg      %r6
#define WhenDonePC      %r7
#define StartupPC       %r8

/* WARNING: SP is not part of machineState! */
#define PCState         0
#define InitialPCState  4
#define InitialArgState 5
#define WhenDonePCState 6
#define StartupPCState  7

#endif // HOST_PPC

#ifdef HOST_i386

/* the offsets of the registers from the beginning of the thread object */
#define _ESP     0
#define _EAX     4
#define _EBX     8
#define _ECX     12
#define _EDX     16
#define _EBP     20
#define _ESI     24
#define _EDI     28
#define _PC      32

/* These definitions are used in Thread::AllocateStack(). */
#define PCState         (_PC/4-1)
#define FPState         (_EBP/4-1)
#define InitialPCState  (_ESI/4-1)
#define InitialArgState (_EDX/4-1)
#define WhenDonePCState (_EDI/4-1)
#define StartupPCState  (_ECX/4-1)

#define InitialPC       %esi
#define InitialArg      %edx
#define WhenDonePC      %edi
#define StartupPC       %ecx
#endif

#ifdef HOST_x86_64

/* the offsets of the registers from the beginning of the thread object */
#define _RSP     0
#define _RAX     8
#define _RBX     16
#define _RCX     24
#define _RDX     32
#define _RBP     40
#define _RSI     48
#define _RDI     56
#define _R12     64
#define _R13     72
#define _R14     80
#define _R15     88
#define _PC      96

/* These definitions are used in Thread::AllocateStack(). */
#define PCState         (_PC/8-1)
#define FPState         (_RBP/8-1)
#define InitialPCState  (_RSI/8-1)
#define InitialArgState (_RDX/8-1)
#define WhenDonePCState (_RDI/8-1)
#define StartupPCState  (_RCX/8-1)

#define InitialPC       %rsi
#define InitialArg      %rdx
#define WhenDonePC      %rdi
#define StartupPC       %rcx
#endif

#endif // SWITCH_H
