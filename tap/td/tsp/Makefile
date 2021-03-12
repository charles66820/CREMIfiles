CC = gcc
CFLAGS = -O3 -Wall -g -Wno-unused-function -Wno-deprecated-declarations
LDLIBS = -lm

ifeq ($(shell uname -s), Darwin)
   LDLIBS += -framework OpenGL -framework GLUT
else
   LDLIBS += -lglut -lGLU -lGL
endif

LDLIBS += $(shell sdl2-config --libs)
LDFLAGS += $(shell sdl2-config --cflags)

tsp_src := $(wildcard tsp_*.c)
tsp_src := $(filter-out %.cor.c,$(tsp_src))
tsp_objs := $(patsubst %.c,%.o,$(tsp_src))

.PHONY: all clean
all: tsp_main

tsp_main: tools.o $(tsp_objs)

test_heap: test_heap.o heap.o

#a_star: a_star.o tools.o heap.o

clean:
	rm -f *.o
	rm -f tsp_main
	rm -f test_heap
	rm -f a_star
	rm -fr *.dSYM/
