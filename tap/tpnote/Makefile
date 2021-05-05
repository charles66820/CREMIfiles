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

tp:  tools.o main.o tp.o util.o

.PHONY: clean
clean:
	rm -f *.o
	rm -f tp
	rm -fr *.dSYM/
