// valgrind.h
//     Valgrind hooks to announce stack allocation/deallocation
//
// Copyright (c) 2009 Samuel Thibault
// All rights reserved.  See copyright.h for copyright notice and limitation
// of liability and disclaimer of warranty provisions.

#ifndef VALGRIND_H
#define VALGRIND_H

#ifdef HAVE_VALGRIND
#include <valgrind/valgrind.h>
#endif

#ifndef VALGRIND_STACK_REGISTER
#define VALGRIND_STACK_REGISTER(start, end) 0
#endif

#ifndef VALGRIND_STACK_DEREGISTER
#define VALGRIND_STACK_DEREGISTER(id) ((void)0)
#endif

#endif // VALGRIND_H
