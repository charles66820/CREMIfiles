CC=gcc
CFLAGS=-std=c99 -O3 -Wall -g -DDEBUG -DSDL --coverage -I/usr/include/SDL2
LDFLAGS=-rdynamic -lm -lSDL2 -lSDL2_ttf -lSDL2_image

.PHONY : all
all : recolor_sdl

libgame.a : game.o game_io.o
	ar -cr $@ $^

recolor_sdl : recolor_sdl.o main_sdl.o fun_sdl.o game_rand.o load_game.o libgame.a
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

.PHONY : clean
clean :
	rm -f *~ *.o *.a *.gcno *.gcda recolor_sdl