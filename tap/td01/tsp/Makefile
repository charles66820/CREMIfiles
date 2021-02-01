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

# all: tsp test_heap a_star

# tsp_main: tsp_main.o tools.o tsp_brute_force.o tsp_prog_dyn.o tsp_heuristic.o tsp_mst.o

tsp: tsp_main.o tools.o tsp_brute_force.o

#test_heap: test_heap.o heap.o

#a_star: a_star.o tools.o heap.o

clean:
	rm -f *.o
	rm -f tsp_main tsp
	rm -f test_heap
	rm -f a_star a_star
	rm -fr *.dSYM/
