// console.cc 
//	Routines to simulate a serial port to a console device.
//	A console has input (a keyboard) and output (a display).
//	These are each simulated by operations on UNIX files.
//	The simulated device is asynchronous,
//	so we have to invoke the interrupt handler (after a simulated
//	delay), to signal that a byte has arrived and/or that a written
//	byte has departed.
//
//  DO NOT CHANGE -- part of the machine emulation
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "console.h"
#include "system.h"
#include <langinfo.h>

#define NOCHAR (-42)

// Dummy functions because C++ is weird about pointers to member functions
static void ConsoleReadPoll(void *c) 
{ Console *console = (Console *)c; console->CheckCharAvail(); }
static void ConsoleWriteDone(void *c)
{ Console *console = (Console *)c; console->WriteDone(); }

//----------------------------------------------------------------------
// Console::Console
// 	Initialize the simulation of a hardware console device.
//
//	"readFile" -- UNIX file simulating the keyboard (NULL -> use stdin)
//	"writeFile" -- UNIX file simulating the display (NULL -> use stdout)
// 	"readAvailHandler" is the interrupt handler called when a character arrives
//		from the keyboard
// 	"writeDoneHandler" is the interrupt handler called when a character has
//		been output, so that it is ok to request the next char be
//		output
//----------------------------------------------------------------------

int Console::stdin_busy;

Console::Console(const char *readFile, const char *writeFile, VoidFunctionPtr readAvailHandler, 
		VoidFunctionPtr writeDoneHandler, void *callArg)
{
    if (readFile == NULL)
    {
	ASSERT(!stdin_busy);
	stdin_busy = 1;
	readFileNo = 0;					// keyboard = stdin
    }
    else
    	readFileNo = OpenForReadWrite(readFile, TRUE);	// should be read-only
    if (writeFile == NULL)
	writeFileNo = 1;				// display = stdout
    else
    	writeFileNo = OpenForWrite(writeFile);

    // set up the stuff to emulate asynchronous interrupts
    writeHandler = writeDoneHandler;
    readHandler = readAvailHandler;
    handlerArg = callArg;
    putBusy = FALSE;
    incoming = NOCHAR;

    // start polling for incoming packets
    interrupt->Schedule(ConsoleReadPoll, this, ConsoleTime, ConsoleReadInt);
}

//----------------------------------------------------------------------
// Console::~Console
// 	Clean up console emulation
//----------------------------------------------------------------------

Console::~Console()
{
    if (readFileNo != 0)
	Close(readFileNo);
    else
	stdin_busy = 0;
    readFileNo = -1;
    if (writeFileNo != 1)
	Close(writeFileNo);
    writeFileNo = -1;

    /* Wait for last interrupts to happen */
    while (readFileNo != -2)
	currentThread->Yield();
    while (putBusy)
	currentThread->Yield();
}

//----------------------------------------------------------------------
// Console::CheckCharAvail()
// 	Periodically called to check if a character is available for
//	input from the simulated keyboard (eg, has it been typed?).
//
//	Only read it in if there is buffer space for it (if the previous
//	character has been grabbed out of the buffer by the Nachos kernel).
//	Invoke the "read" interrupt handler, once the character has been 
//	put into the buffer. 
//----------------------------------------------------------------------

void
Console::CheckCharAvail()
{
    unsigned char c, d;
    int n;
    int cont = 1;

    if (readFileNo == -1) {
	// Termination, don't schedule any other interrupt
	readFileNo = -2;
	n = 0;
	cont = 0;
    } else if ((incoming != NOCHAR) || !PollFile(readFileNo)) {
	// do nothing if character is already buffered, or none to be read
	n = 0;
    } else {
	// otherwise, read character and tell user about it
	n = ReadPartial(readFileNo, &c, sizeof(c));
	if (n == 0) {
	    incoming = EOF;
	    (*readHandler)(handlerArg);
	} else if (strcmp(nl_langinfo(CODESET),"UTF-8")) {
	    /* Not UTF-8, assume 8bit locale */
	    incoming = c;
	} else
	    /* UTF-8, decode */
	if (!(c & 0x80)) {
	    /* ASCII */
	    incoming = c;
	} else {
	    if ((c & 0xe0) != 0xc0)
		/* continuation char or more than three bytes, drop */
		return;
	    if (c & 0x1c)
		/* Not latin1, drop */
		return;
	    /* latin1 UTF-8 char, read second char */
	    n = ReadPartial(readFileNo,  &d, sizeof(d));
	    if (n == 0) {
		incoming = EOF;
		(*readHandler)(handlerArg);
	    } else if ((d & 0xc0) != 0x80) {
		/* Odd, drop */
		return;
	    } else {
		incoming = (c & 0x03) << 6 | d;
	    }
	}
    }

    if (cont)
	// schedule the next time to poll for a packet
	interrupt->Schedule(ConsoleReadPoll, this, ConsoleTime, 
			    ConsoleReadInt);

    if (n) {
	stats->numConsoleCharsRead++;
	(*readHandler)(handlerArg);
    }


}

//----------------------------------------------------------------------
// Console::WriteDone()
// 	Internal routine called when it is time to invoke the interrupt
//	handler to tell the Nachos kernel that the output character has
//	completed.
//----------------------------------------------------------------------

void
Console::WriteDone()
{
    putBusy = FALSE;
    stats->numConsoleCharsWritten++;
    (*writeHandler)(handlerArg);
}

//----------------------------------------------------------------------
// Console::RX()
// 	Read a character from the input buffer, if there is any there.
//	Either return the character, or EOF if none buffered or the end of the
//	input file was reached.
//----------------------------------------------------------------------

int
Console::RX()
{
   int ch = incoming;

   // We should not be reading anything if no character was received yet
   ASSERT(incoming != NOCHAR);

   incoming = NOCHAR;
   return ch;
}

//----------------------------------------------------------------------
// Console::TX()
// 	Write a character to the simulated display, schedule an interrupt 
//	to occur in the future, and return.
//----------------------------------------------------------------------

void
Console::TX(int ch)
{
    unsigned char c;

    // Make sure that we are not already transferring a character
    ASSERT(putBusy == FALSE);

    // Compensate when given a non-ascii latin1 character passed as signed char
    if (ch < 0 && ch >= -128)
	ch += 256;

    if (ch < 0x80 || strcmp(nl_langinfo(CODESET),"UTF-8")) {
	/* Not UTF-8 or ASCII, assume 8bit locale */
	c = ch;
        WriteFile(writeFileNo, &c, sizeof(c));
    } else if (ch < 0x100) {
	/* Non-ASCII UTF-8, thus two bytes */
	c = ((ch & 0xc0) >> 6) | 0xc0;
        WriteFile(writeFileNo, &c, sizeof(c));
	c = (ch & 0x3f) | 0x80;
        WriteFile(writeFileNo, &c, sizeof(c));
    } /* Else not latin1, drop */
    putBusy = TRUE;
    interrupt->Schedule(ConsoleWriteDone, this, ConsoleTime,
					ConsoleWriteInt);
}
