// main.cc 
//      Bootstrap code to initialize the operating system kernel.
//
//      Allows direct calls into internal operating system functions,
//      to simplify debugging and testing.  In practice, the
//      bootstrap code would just initialize data structures,
//      and start a user program to print the login prompt.
//
//      Most of this file is not needed until later assignments.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#define MAIN
#include "copyright.h"
#undef MAIN

#include "utility.h"
#include "system.h"

#ifdef USER_PROGRAM
#include "progtest.h"
#endif

#include <malloc.h>


// External functions used by this file

extern void ThreadTest (void), Copy (char *unixFile, char *nachosFile);
extern void Print (char *file), PerformanceTest (void);
extern void MailTest (int networkID);

//----------------------------------------------------------------------
// main
//      Bootstrap the operating system kernel.  
//      
//      Check command line arguments
//      Initialize data structures
//      (optionally) Call test procedure
//
//      "argc" is the number of command line arguments (including the name
//              of the command) -- ex: "nachos -d +" -> argc = 3 
//      "argv" is an array of strings, one for each command line argument
//              ex: "nachos -d +" -> argv = {"nachos", "-d", "+"}
//----------------------------------------------------------------------

int
main (int argc, char **argv)
{
    int argCount;		// the number of arguments 
    // for a particular command

    if (argc > 1 && !strcmp (argv[1], "-h")) // print help
      {
//  NOTE -- flags are ignored until the relevant assignment.
//  Some of the flags are interpreted here; some in system.cc.
//
	printf (
"Usage: nachos -d <debugflags> -rs <random seed #> -z -h\n"
#ifdef USER_PROGRAM
"       -s -x <nachos file> -c <consoleIn> <consoleOut>\n"
#endif
#ifdef FILESYS
"       -f -cp <unix file> <nachos file>\n"
"       -p <nachos file> -r <nachos file> -l -D -t\n"
#endif
#ifdef NETWORK
"       -n <network reliability> -m <machine id>\n"
"       -o <other machine id>\n"
#endif
"\n"
"-d causes certain debugging messages to be printed (cf. utility.h)\n"
"-rs causes Yield to occur at random (but repeatable) spots\n"
"-z prints the copyright message\n"
"-h prints some help about options\n"
"\n"
#ifdef USER_PROGRAM
"USER_PROGRAM\n"
"-s causes user programs to be executed in single-step mode\n"
"-x runs a user program\n"
"-c tests the console\n"
#endif
#ifdef FILESYS
"FILESYS\n"
"-f causes the physical disk to be formatted\n"
"-cp copies a file from UNIX to Nachos\n"
"-p prints a Nachos file to stdout\n"
"-r removes a Nachos file from the file system\n"
"-l lists the contents of the Nachos directory\n"
"-D prints the contents of the entire file system\n"
"-t tests the performance of the Nachos file system\n"
#endif
#ifdef NETWORK
"NETWORK\n"
"-n sets the network reliability\n"
"-m sets this machine's host id (needed for the network)\n"
"-o runs a simple test of the Nachos network software"
#endif
	);
	return (0);
      }

    DEBUG ('t', "Entering main");
    (void) Initialize (argc, argv);

#ifdef THREADS
    ThreadTest ();
#endif

    for (argc--, argv++; argc > 0; argc -= argCount, argv += argCount)
      {
	  argCount = 1;
	  if (!strcmp (*argv, "-z"))	// print copyright
	      printf ("%s", copyright);
#ifdef USER_PROGRAM
	  if (!strcmp (*argv, "-x"))
	    {			// run a user program
		ASSERT (argc > 1);
		StartProcess (*(argv + 1));
		argCount = 2;
	    }
	  else if (!strcmp (*argv, "-c"))
	    {			// test the console
		if (argc == 1)
		    ConsoleTest (NULL, NULL);
		else
		  {
		      ASSERT (argc > 2);
		      ConsoleTest (*(argv + 1), *(argv + 2));
		      argCount = 3;
		  }
	    }
#endif // USER_PROGRAM
#ifdef FILESYS
	  if (!strcmp (*argv, "-cp"))
	    {			// copy from UNIX to Nachos
		ASSERT (argc > 2);
		Copy (*(argv + 1), *(argv + 2));
		argCount = 3;
	    }
	  else if (!strcmp (*argv, "-p"))
	    {			// print a Nachos file
		ASSERT (argc > 1);
		Print (*(argv + 1));
		argCount = 2;
	    }
	  else if (!strcmp (*argv, "-r"))
	    {			// remove Nachos file
		ASSERT (argc > 1);
		fileSystem->Remove (*(argv + 1));
		argCount = 2;
	    }
	  else if (!strcmp (*argv, "-l"))
	    {			// list Nachos directory
		fileSystem->List ();
	    }
	  else if (!strcmp (*argv, "-D"))
	    {			// print entire filesystem
		fileSystem->Print ();
	    }
	  else if (!strcmp (*argv, "-t"))
	    {			// performance test
		PerformanceTest ();
	    }
#endif // FILESYS
#ifdef NETWORK
	  if (!strcmp (*argv, "-o"))
	    {
		ASSERT (argc > 1);
		Delay (2);	// delay for 2 seconds
		// to give the user time to 
		// start up another nachos
		MailTest (atoi (*(argv + 1)));
		argCount = 2;
	    }
#endif // NETWORK
      }

    currentThread->Finish ();	// NOTE: if the procedure "main" 
    // returns, then the program "nachos"
    // will exit (as any other normal program
    // would).  But there may be other
    // threads on the ready list.  We switch
    // to those threads by saying that the
    // "main" thread is finished, preventing
    // it from returning.
    return (0);			// Not reached...
}
