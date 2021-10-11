#ifdef CHANGED
#include "consoledriver.h"

#include "copyright.h"
#include "synch.h"
#include "system.h"

static Semaphore *readAvail;
static Semaphore *writeDone;

static void ReadAvailHandler(void *arg) {
  (void)arg;
  readAvail->V();
}

static void WriteDoneHandler(void *arg) {
  (void)arg;
  writeDone->V();
}

ConsoleDriver::ConsoleDriver(const char *in, const char *out) {
  readAvail = new Semaphore("read avail", 0);
  writeDone = new Semaphore("write done", 0);
  console = new Console(in, out, ReadAvailHandler, WriteDoneHandler, NULL);
}

ConsoleDriver::~ConsoleDriver() {
  delete console;
  delete writeDone;
  delete readAvail;
}

void ConsoleDriver::PutChar(int ch) {
  console->TX(ch);  // echo it!
  writeDone->P();   // wait for write to finish
}

int ConsoleDriver::GetChar() {
  readAvail->P();  // wait for character to arrive
  return console->RX();
}

void ConsoleDriver::PutString(const char s[]) {
  for (int i = 0; s[i] != '\0' ; i++) {
      console->TX(s[i]);
      writeDone->P();
  }
}

void ConsoleDriver::GetString(char *s, int n) {
  //Check if the size is correct
  if (n <= 0) {
    return;
  }

  int c = 0;
  for (int i = 0; i < n-1; i++) {
    readAvail->P();
    c = console->RX();
    //Reading stop after and EOF or a newline
    if (c == EOF) {
      s[i] = '\0';
      return;
    }
    if (c == '\n') {
      s[i] = '\n';
      s[i+1] = '\0';
      return;
    }
    s[i] = c;
  }
  //At the end of the buffer we add a termining null byte '\0'
  s[n] = '\0';
}

#endif  // CHANGED