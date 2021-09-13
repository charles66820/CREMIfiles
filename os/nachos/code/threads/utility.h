// utility.h 
//      Miscellaneous useful definitions, including debugging routines.
//
//      The debugging routines allow the user to turn on selected
//      debugging messages, controllable from the command line arguments
//      passed to Nachos (-d).  You are encouraged to add your own
//      debugging flags.  The pre-defined debugging flags are:
//
//      '+' -- turn on all debug messages
//      't' -- thread system
//      's' -- semaphores, locks, and conditions
//      'i' -- interrupt emulation
//      'm' -- machine emulation (USER_PROGRAM)
//      'd' -- disk emulation (FILESYS)
//      'f' -- file system (FILESYS)
//      'a' -- address spaces (USER_PROGRAM)
//      'n' -- network emulation (NETWORK)
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#ifndef UTILITY_H
#define UTILITY_H

#include "copyright.h"

// Miscellaneous useful routines

#include <bool.h>	// Boolean values
#include <algorithm>	// min

// Divide and either round up or down 
#define divRoundDown(n,s)  ((n) / (s))
#define divRoundUp(n,s)    (((n) / (s)) + ((((n) % (s)) > 0) ? 1 : 0))

enum AnsiColor {
	ColorBlack,
	ColorRed,
	ColorGreen,
	ColorYellow,
	ColorBlue,
	ColorMagenta,
	ColorCyan,
	ColorWhite,
};

// This declares the type "VoidFunctionPtr" to be a "pointer to a
// function taking an integer argument and returning nothing".  With
// such a function pointer (say it is "func"), we can call it like this:
//
//      (*func) (17);
//
// This is used by Thread::Start and for interrupt handlers, as well
// as a couple of other places.

typedef void (*VoidFunctionPtr) (void *arg);
typedef void (*VoidFunctionPtr2) (void *arg, void *arg2);
typedef void (*VoidNoArgFunctionPtr) ();


// Include interface that isolates us from the host machine system library.
// Requires definition of bool, and VoidFunctionPtr
#include "sysdep.h"

// Interface to debugging routines.

extern void DebugInit (const char *flags);	// enable printing debug messages

extern bool DebugIsEnabled (char flag);	// Is this debug flag enabled?

extern void DEBUG (char flag, const char *format, ...);	// Print debug message 
							// if flag is enabled

extern void SetColor(FILE *output, enum AnsiColor color);
extern void SetBold(FILE *output);
extern void ClearColor(FILE *output);

//----------------------------------------------------------------------
// ASSERT
//      If condition is false,  print a message and dump core.
//      Useful for documenting assumptions in the code.
//
//      NOTE: needs to be a #define, to be able to print the location 
//      where the error occurred.
//----------------------------------------------------------------------
#define ASSERT(condition) do {                                                \
    if (!(condition)) {                                                       \
        SetColor(stderr, ColorRed);                                           \
        SetBold(stderr);                                                      \
        fprintf(stderr, "Assertion %s failed: line %d, file \"%s:%d\"\n",     \
                #condition,                                                   \
                __LINE__, __FILE__, __LINE__);                                \
        ClearColor(stderr);                                                   \
        fflush(stderr);                                                       \
        Abort();                                                              \
    }                                                                         \
  } while(0)

/* A lot of classes should not be allowed to be copied, e.g. it doesn't make
 * sense to copy a Semaphore. To enforce this, inherit from this class. */

class dontcopythis {
	private:
		dontcopythis (const dontcopythis &);
		dontcopythis& operator= (const dontcopythis&);
	public:
		dontcopythis() {};
};

#endif /* UTILITY_H */
