#ifdef CHANGED

#ifndef USERTHREAD_H
#define USERTHREAD_H

extern int do_ThreadCreate(int f, int arg);
extern void do_ThreadExit(void);
static void StartUserThread(void *args);

#endif // USERTHREAD_H
#endif // CHANGED