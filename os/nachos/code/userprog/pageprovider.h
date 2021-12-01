#ifdef CHANGED
// pageprovider.h

#ifndef PAGEPROVIDER_H
#define PAGEPROVIDER_H

#include "bitmap.h"
#include "synch.h"

class PageProvider {
 public:
  PageProvider(int nbPhysicalPage);  // Initialise a page provider, with "nbPhysicalPage" bits
  ~PageProvider();           // De-allocate a pageprovider
  int GetEmptyPage(void);    // Return a physical page number
  void ReleasePage(int pageNum);  // Free a page
  uint NumAvailPage(void);         // Give the disponible pages number
 private:
  BitMap *physicalPage;
  Lock *mutex;
  Condition *cond;
};

#endif  // PAGEPROVIDER_H
#endif  // CHANGED