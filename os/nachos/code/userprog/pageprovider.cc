#ifdef CHANGED
// pageprovider.cc

#include "pageprovider.h"
#include "bitmap.h"
#include "synch.h"
#include "system.h"

//----------------------------------------------------------------------
// PageProvider::PageProvider
//      Initialize a pageprovider
//----------------------------------------------------------------------

PageProvider::PageProvider(int nbPhysicalPage) {
  physicalPage = new BitMap(nbPhysicalPage);
  mutex = new Lock("page mutex");
  cond = new Condition("page monitor");
}

//----------------------------------------------------------------------
// PageProvider::~PageProvider
//      De-allocate a pageprovider.
//----------------------------------------------------------------------

PageProvider::~PageProvider() {
  delete physicalPage;
  delete mutex;
  delete cond;
}

//----------------------------------------------------------------------
// PageProvider::GetEmptyPage
//      Return a physical page number and initialise to 0 with memset
//----------------------------------------------------------------------

int PageProvider::GetEmptyPage() {
  mutex->Acquire();
  int index;
  while ((index = physicalPage->Find()) == -1)
    cond->Wait(mutex);// wait
  physicalPage->Mark(index);
  mutex->Release();

  memset(&machine->mainMemory[index], 0, PageSize);

  return index;
}

//----------------------------------------------------------------------
// PageProvider::ReleasePage
//      Free a page
//----------------------------------------------------------------------

void PageProvider::ReleasePage(int pageNum) {
  mutex->Acquire();
  physicalPage->Clear(pageNum);
  cond->Signal(mutex);
  mutex->Release();
}

//----------------------------------------------------------------------
// PageProvider::NumAvailPage
//      Give the disponible pages number
//----------------------------------------------------------------------

int PageProvider::NumAvailPage() {
  return physicalPage->NumClear();
}

#endif  // CHANGED