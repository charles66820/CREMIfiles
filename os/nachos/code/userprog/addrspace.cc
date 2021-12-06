// addrspace.cc
//      Routines to manage address spaces (executing user programs).
//
//      In order to run a user program, you must:
//
//      1. link with the -N -T 0 option
//      2. run coff2noff to convert the object file to Nachos format
//              (Nachos object code format is essentially just a simpler
//              version of the UNIX executable object code format)
//      3. load the NOFF file into the Nachos file system
//              (if you haven't implemented the file system yet, you
//              don't need to do this last step)
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation
// of liability and disclaimer of warranty provisions.

#include "addrspace.h"

#include "copyright.h"
#include "new"
#include "noff.h"
#include "syscall.h"
#include "system.h"

#ifdef CHANGED
#include "bitmap.h"
#include "synch.h"
#include "thread.h"
#endif  // CHANGED

//----------------------------------------------------------------------
// SwapHeader
//      Do little endian to big endian conversion on the bytes in the
//      object file header, in case the file was generated on a little
//      endian machine, and we're now running on a big endian machine.
//----------------------------------------------------------------------

static void SwapHeader(NoffHeader *noffH) {
  noffH->noffMagic = WordToHost(noffH->noffMagic);
  noffH->code.size = WordToHost(noffH->code.size);
  noffH->code.virtualAddr = WordToHost(noffH->code.virtualAddr);
  noffH->code.inFileAddr = WordToHost(noffH->code.inFileAddr);
  noffH->initData.size = WordToHost(noffH->initData.size);
  noffH->initData.virtualAddr = WordToHost(noffH->initData.virtualAddr);
  noffH->initData.inFileAddr = WordToHost(noffH->initData.inFileAddr);
  noffH->uninitData.size = WordToHost(noffH->uninitData.size);
  noffH->uninitData.virtualAddr = WordToHost(noffH->uninitData.virtualAddr);
  noffH->uninitData.inFileAddr = WordToHost(noffH->uninitData.inFileAddr);
}

#ifdef CHANGED
//-----------------------------------------------------------------------
// ReadAtVirtual
//      Read from the executable position and writes at a virtual address
//-----------------------------------------------------------------------
static void ReadAtVirtual(OpenFile *executable, int virtualaddr, int numBytes,
                          int position, TranslationEntry *pageTable,
                          unsigned numPages) {
  if (numBytes <= 0) return;

  char buffer[numBytes];
  // Read the executable data and store in the buffer
  executable->ReadAt(&buffer, numBytes, position);

  // Save state and set it to the new process
  TranslationEntry *savePageTable = machine->currentPageTable;
  unsigned int saveNumPages = machine->currentPageTableSize;
  machine->currentPageTable = pageTable;
  machine->currentPageTableSize = numPages;

  // Copy the data to the virtual memory of the futur process
  for (int i = 0; i < numBytes; i++)
    if (!machine->WriteMem(virtualaddr + i, 1, buffer[i])) break;

  // Restore state
  machine->currentPageTable = savePageTable;
  machine->currentPageTableSize = saveNumPages;
}
#endif  // CHANGED

//----------------------------------------------------------------------
// AddrSpaceList
//      List of all address spaces, for debugging
//----------------------------------------------------------------------
List AddrSpaceList;

//----------------------------------------------------------------------
// AddrSpace::AddrSpace
//      Create an address space to run a user program.
//      Load the program from a file "executable", and set everything
//      up so that we can start executing user instructions.
//
//      Assumes that the object code file is in NOFF format.
//
//      First, set up the translation from program memory to physical
//      memory.  For now, this is really simple (1:1), since we are
//      only uniprogramming, and we have a single unsegmented page table
//
//      "executable" is the file containing the object code to load into memory
//----------------------------------------------------------------------

AddrSpace::AddrSpace(OpenFile *executable) {
  unsigned int i, size;

  executable->ReadAt(&noffH, sizeof(noffH), 0);
  if ((noffH.noffMagic != NOFFMAGIC) &&
      (WordToHost(noffH.noffMagic) == NOFFMAGIC))
    SwapHeader(&noffH);
  /* Check that this is really a MIPS program */
  ASSERT(noffH.noffMagic == NOFFMAGIC);

  // how big is address space?
  size = noffH.code.size + noffH.initData.size + noffH.uninitData.size +
         UserStacksAreaSize;  // we need to increase the size
  // to leave room for the stack
  numPages = divRoundUp(size, PageSize);
  size = numPages * PageSize;

  // check we're not trying
  // to run anything too big --
  // at least until we have
  // virtual memory
  if (numPages > pageProvider->NumAvailPage()) throw std::bad_alloc();

  DEBUG('a', "Initializing address space, num pages %d, total size 0x%x\n",
        numPages, size);

  // get physical pages
  int physicalPages[numPages];
  for (i = 0; i < numPages; i++)
    physicalPages[i] = pageProvider->GetEmptyPage();

  bool randomizedPages = true;
  if (randomizedPages) {
    // rand the table
    // shuffle fonction from
    // https://stackoverflow.com/questions/6127503/shuffle-array-in-c
    if (numPages > 1) {
      for (uint k = 0; k < numPages - 1; k++) {
        uint j = k + rand() / (RAND_MAX / (numPages - k) + 1);
        uint t = physicalPages[j];
        physicalPages[j] = physicalPages[k];
        physicalPages[k] = t;
      }
    }
  }

  // first, set up the translation
  pageTable = new TranslationEntry[numPages];
  for (i = 0; i < numPages; i++) {
    pageTable[i].physicalPage = physicalPages[i];
    pageTable[i].valid = TRUE;
    pageTable[i].use = FALSE;
    pageTable[i].dirty = FALSE;
    pageTable[i].readOnly = FALSE;  // if the code segment was entirely on
                                    // a separate page, we could set its
                                    // pages to be read-only
  }

  // then, copy in the code and data segments into memory
  if (noffH.code.size > 0) {
    DEBUG('a', "Initializing code segment, at 0x%x, size 0x%x\n",
          noffH.code.virtualAddr, noffH.code.size);
#ifdef CHANGED
    ReadAtVirtual(executable, noffH.code.virtualAddr, noffH.code.size,
                  noffH.code.inFileAddr, pageTable, numPages);
#else
    executable->ReadAt(&(machine->mainMemory[noffH.code.virtualAddr]),
                       noffH.code.size, noffH.code.inFileAddr);
#endif  // CHANGED
  }
  if (noffH.initData.size > 0) {
    DEBUG('a', "Initializing data segment, at 0x%x, size 0x%x\n",
          noffH.initData.virtualAddr, noffH.initData.size);
#ifdef CHANGED
    ReadAtVirtual(executable, noffH.initData.virtualAddr, noffH.initData.size,
                  noffH.initData.inFileAddr, pageTable, numPages);
#else
    executable->ReadAt(&(machine->mainMemory[noffH.initData.virtualAddr]),
                       noffH.initData.size, noffH.initData.inFileAddr);

#endif  // CHANGED
  }

  DEBUG('a', "Area for stacks at 0x%x, size 0x%x\n", size - UserStacksAreaSize,
        UserStacksAreaSize);

  pageTable[0].valid = FALSE;  // Catch NULL dereference

  AddrSpaceList.Append(this);

#ifdef CHANGED
  // Init mutex for thread counter
  mutex = new Lock("CounterProtection");

  // Init thread list
  userThreadList = new List();

  // Allocate stack bitmap
  int nbrMaxStack = UserStacksAreaSize / UserStackSize;
  stackBitMap = new BitMap(nbrMaxStack);

  machine->DumpMem("addrspace.svg");
#endif  // CHANGED
}

//----------------------------------------------------------------------
// AddrSpace::~AddrSpace
//      Dealloate an address space.  Nothing for now!
//----------------------------------------------------------------------

AddrSpace::~AddrSpace() {
#ifdef CHANGED
  for (unsigned int i = 0; i < numPages; i++) {
    pageProvider->ReleasePage(pageTable[i].physicalPage);
  }
#endif  // CHANGED
  delete[] pageTable;
  pageTable = NULL;

  AddrSpaceList.Remove(this);

#ifdef CHANGED
  delete mutex;
  delete stackBitMap;
  delete userThreadList;
#endif  // CHANGED
}

//----------------------------------------------------------------------
// AddrSpace::InitRegisters
//      Set the initial values for the user-level register set.
//
//      We write these directly into the "machine" registers, so
//      that we can immediately jump to user code.  Note that these
//      will be saved/restored into the currentThread->userRegisters
//      when this thread is context switched out.
//----------------------------------------------------------------------

void AddrSpace::InitRegisters() {
  int i;

  for (i = 0; i < NumTotalRegs; i++) machine->WriteRegister(i, 0);

  // Initial program counter -- must be location of "Start"
  machine->WriteRegister(PCReg, USER_START_ADDRESS);

  // Need to also tell MIPS where next instruction is, because
  // of branch delay possibility
  machine->WriteRegister(NextPCReg, machine->ReadRegister(PCReg) + 4);

  // Set the stack register to the end of the address space, where we
  // allocated the stack; but subtract off a bit, to make sure we don't
  // accidentally reference off the end!
  machine->WriteRegister(StackReg, numPages * PageSize - 16);
  DEBUG('a', "Initializing stack register to 0x%x\n", numPages * PageSize - 16);
}

#ifdef CHANGED
//----------------------------------------------------------------------
// AddrSpace::AllocateUserStack
//      Determine where the stack of the thread will be allocated
//----------------------------------------------------------------------

int AddrSpace::AllocateUserStack(Thread *newThread) {
  mutex->Acquire();
  // Mark stack space used in stack bitmap
  int index = stackBitMap->Find();
  if (index == -1) {
    mutex->Release();
    return 0;
  }
  stackBitMap->Mark(index);
  userThreadList->Append(newThread);

  // Determine the thread stack address
  if (UserStackSize * index + 16 >= UserStacksAreaSize) {
    stackBitMap->Clear(index);
    mutex->Release();
    return -1;
  }
  newThread->SetStackIndex(index);

  mutex->Release();

  return 1;
}

//----------------------------------------------------------------------
// AddrSpace::DeallocateUserStack
//      Deallocate the stack of the thread and if it is the last, it finish the
//      process
//----------------------------------------------------------------------

bool AddrSpace::DeallocateUserStack(Thread *thread) {
  mutex->Acquire();
  stackBitMap->Clear(thread->GetStackIndex());

  userThreadList->Remove(thread);

  thread->SetStackIndex(-1);
  mutex->Release();

  return userThreadList->IsEmpty();
}

void AddrSpace::CloseAllUserThread() {
  if (userThreadList->IsEmpty()) return;

  bool isLastUserThread;
  Thread *t;
  do {
    t = (Thread *)userThreadList->FirstElement()->item;
    isLastUserThread = currentThread->space->DeallocateUserStack(t);
    if (t != currentThread) t->Finish();
  } while (!isLastUserThread);
}
#endif  // CHANGED

//----------------------------------------------------------------------
// AddrSpace::Dump
//      Dump program layout as SVG
//----------------------------------------------------------------------

static void DrawArea(FILE *output, unsigned sections_x, unsigned virtual_x,
                     unsigned y, unsigned blocksize, struct segment *segment,
                     const char *name) {
  if (segment->size == 0) return;

  ASSERT((segment->virtualAddr % PageSize == 0));
  ASSERT((segment->size % PageSize == 0));
  unsigned page = segment->virtualAddr / PageSize;
  unsigned end = (segment->virtualAddr + segment->size) / PageSize;

  fprintf(output,
          "<rect x=\"%u\" y=\"%u\" width=\"%u\" height=\"%u\" "
          "fill=\"#ffffff\" "
          "stroke=\"#000000\" stroke-width=\"1\"/>\n",
          sections_x, y - end * blocksize, virtual_x - sections_x,
          (end - page) * blocksize);

  fprintf(
      output,
      "<text x=\"%u\" y=\"%u\" fill=\"#000000\" font-size=\"%u\">%s</text>\n",
      sections_x, y - page * blocksize, blocksize, name);
}

unsigned AddrSpace::Dump(FILE *output, unsigned addr_x, unsigned sections_x,
                         unsigned virtual_x, unsigned virtual_width,
                         unsigned physical_x, unsigned virtual_y, unsigned y,
                         unsigned blocksize) {
  unsigned ret = machine->DumpPageTable(output, pageTable, numPages, addr_x,
                                        virtual_x, virtual_width, physical_x,
                                        virtual_y, y, blocksize);

  DrawArea(output, sections_x, virtual_x, virtual_y, blocksize, &noffH.code,
           "code");
  DrawArea(output, sections_x, virtual_x, virtual_y, blocksize, &noffH.initData,
           "data");
  DrawArea(output, sections_x, virtual_x, virtual_y, blocksize,
           &noffH.uninitData, "bss");

  DumpThreadsState(output, this, sections_x, virtual_x, virtual_y, blocksize);

  return ret;
}

//----------------------------------------------------------------------
// AddrSpace::AddrSpacesRoom
//      Return how much room is needed for showing address spaces
//----------------------------------------------------------------------

unsigned AddrSpacesRoom(unsigned blocksize) {
  ListElement *element;
  unsigned room = 0;

  for (element = AddrSpaceList.FirstElement(); element;
       element = element->next) {
    AddrSpace *space = (AddrSpace *)element->item;
    room += machine->PageTableRoom(space->NumPages(), blocksize);
  }

  return room;
}

//----------------------------------------------------------------------
// AddrSpace::DumpAddrSpaces
//      Dump all address spaces
//----------------------------------------------------------------------

void DumpAddrSpaces(FILE *output, unsigned addr_x, unsigned sections_x,
                    unsigned virtual_x, unsigned virtual_width,
                    unsigned physical_x, unsigned y, unsigned blocksize) {
  ListElement *element;
  unsigned virtual_y = y;

  /* TODO: sort by physical page addresses to avoid too much mess */
  for (element = AddrSpaceList.FirstElement(); element;
       element = element->next) {
    AddrSpace *space = (AddrSpace *)element->item;
    virtual_y -=
        space->Dump(output, addr_x, sections_x, virtual_x, virtual_width,
                    physical_x, virtual_y, y, blocksize);
  }
}

//----------------------------------------------------------------------
// AddrSpace::SaveState
//      On a context switch, save any machine state, specific
//      to this address space, that needs saving.
//
//      For now, nothing!
//----------------------------------------------------------------------

void AddrSpace::SaveState() {}

//----------------------------------------------------------------------
// AddrSpace::RestoreState
//      On a context switch, restore the machine state so that
//      this address space can run.
//
//      For now, tell the machine where to find the page table.
//----------------------------------------------------------------------

void AddrSpace::RestoreState() {
  machine->currentPageTable = pageTable;
  machine->currentPageTableSize = numPages;
}
